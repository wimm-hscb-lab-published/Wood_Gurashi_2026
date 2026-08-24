"""Small, dependency-light helpers shared across the DNA pipeline.

Includes maintenance utilities for the MissionBio Varsome annotation cache.
That cache (``annotations_database.csv``) lives inside the installed
``missionbio.annotation`` package, so every process using the conda environment
appends to the *same* file with no locking. Two distinct failure modes result:

*Structural* — interleaved appends from concurrent notebook kernels or SLURM
array tasks produce mangled lines, which surface as::

    ParserError: Error tokenizing data. C error: Expected 18 fields in
    line 69011, saw 23

*Value formatting* — mosaic writes the cache with pandas, so a batch containing
a blank ``Position`` (legitimate for rows like ``chr13:28602226:*/A+AAG``) makes
that column float and every position in the batch gets a ``.0`` suffix, while
batches without blanks write bare ints. ``load_database`` then reads with
``low_memory=True``, infers a different dtype per chunk, and collapses the
column to ``object`` holding a mix of ints, floats and *strings*. A downstream
``int()`` on the string form raises::

    ValueError: invalid literal for int() with base 10: '148386525.0'

The cache is a cache, not a source of truth: dropped rows are simply re-pulled
from the API on the next annotation run, so repair costs API calls rather than
information.
"""

from __future__ import annotations

import copy
import csv
import gc
import math
import os
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ── Chromosome helpers ────────────────────────────────────────────────────────

def extract_chr_num(variant: str) -> int:
    """Return a sortable integer for the chromosome in a variant ID like
    ``'chr17:7577538:C/T'``.  chrX → 23, chrY → 24, chrM/chrMT → 25."""
    chr_part = variant.split(":")[0]
    mapping = {"chrX": 23, "chrY": 24, "chrM": 25, "chrMT": 25}
    if chr_part in mapping:
        return mapping[chr_part]
    try:
        return int(chr_part[3:])
    except ValueError:
        return 999


def extract_chr(col: str) -> str:
    """Return the chromosome token from a variant column name."""
    return col.split(":")[0]


def extract_position(variant: str) -> int:
    """Return the 1-based position from a variant ID like
    ``'chr17:7577538:C/T'``, or -1 if unparseable."""
    parts = str(variant).split(":")
    if len(parts) < 2:
        return -1
    try:
        return int(parts[1])
    except ValueError:
        return -1


def chrom_key(s: pd.Series) -> pd.Series:
    """Map chromosome labels to sortable integers (1‒22, X=23, Y=24, MT=25).

    Handles optional ``chr`` prefix, mixed case, and pushes non-standard
    contigs to the end.
    """
    s = s.astype(str).str.replace("chr", "", regex=False).str.upper().str.strip()
    mapping = {str(i): i for i in range(1, 23)}
    mapping.update({"X": 23, "Y": 24, "MT": 25, "M": 25})
    out = s.map(mapping)

    nonstd = out.isna()
    if nonstd.any():
        codes, _ = pd.factorize(s[nonstd])
        out.loc[nonstd] = 1_000_000 + codes

    return out.astype(int)


# ── Array / assay helpers ─────────────────────────────────────────────────────

def sanitize_row_attrs(assay):
    """Drop ``row_attrs`` that would break slicing (0-D or wrong length)."""
    n = assay.shape[0]
    bad = [
        k for k, v in list(assay.row_attrs.items())
        if np.asarray(v).ndim == 0 or (np.asarray(v).ndim == 1 and len(np.asarray(v)) != n)
    ]
    for k in bad:
        del assay.row_attrs[k]


def drop_scalar_row_attrs(assay):
    """Drop 0-D ``row_attrs`` from *assay* (in-place)."""
    if assay is None:
        return
    for k, v in list(assay.row_attrs.items()):
        if np.asarray(v).ndim == 0:
            del assay.row_attrs[k]


def heatmap_shape(fig) -> tuple:
    """Return ``(n_rows, n_cols)`` from the first heatmap-like trace in a
    Plotly figure."""
    for tr in fig.data:
        z = getattr(tr, "z", None)
        if z is None:
            continue
        try:
            n_rows = len(z)
            n_cols = len(z[0]) if n_rows else 0
            return n_rows, n_cols
        except Exception:
            pass
    return 0, 0


# ── Varsome cache: schema ─────────────────────────────────────────────────────

#: Exact header written by ``missionbio.annotation.varsome``. The first two
#: fields form the MultiIndex that ``load_database`` reads via ``index_col=[0, 1]``.
VARSOME_DB_COLUMNS = [
    "Genome",
    "Variant ID",
    "Position",
    "Ref allele",
    "Alt allele",
    "Varsome url",
    "Variant type",
    "RefSeq transcript id",
    "Gene",
    "Protein",
    "cDNA",
    "Coding impact",
    "Function",
    "Allele Freq (gnomAD)",
    "DANN",
    "dbSNP rsids",
    "ClinVar",
    "COSMIC ids",
]

#: Expected field count per line (18).
VARSOME_N_FIELDS = len(VARSOME_DB_COLUMNS)

#: Index of the ``Position`` field, which is the one prone to mixed formatting.
VARSOME_POSITION_IDX = VARSOME_DB_COLUMNS.index("Position")

#: Accepted values for the ``Genome`` field. A line whose first field is not one
#: of these is almost certainly the tail of an interleaved write.
VARSOME_VALID_GENOMES = {"hg19", "hg38", "hg37", "GRCh37", "GRCh38"}

# Some annotation fields (long rsid lists, multi-term ClinVar strings) are large;
# raise the csv field cap so the reader never truncates a legitimate row.
csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))


# ── Varsome cache: locating ───────────────────────────────────────────────────

def varsome_db_path() -> Path:
    """Return the path to the active Varsome annotation cache.

    Resolved from the installed package rather than hardcoded, so this stays
    correct if the conda env is rebuilt or the module is imported from a
    different environment.
    """
    from missionbio.annotation.varsome import Varsome

    v = Varsome()
    if not hasattr(v, "db_path"):
        raise AttributeError(
            "Varsome instance has no 'db_path' attribute — the annotation "
            "package layout has changed; locate the cache manually."
        )
    return Path(v.db_path)


# ── Varsome cache: locking ────────────────────────────────────────────────────

@contextmanager
def varsome_db_lock(path=None, *, timeout: float = 600.0, poll: float = 0.5,
                    verbose: bool = False):
    """Serialise access to the Varsome cache across processes.

    Takes an exclusive advisory lock on a sidecar ``.lock`` file next to the
    cache. This only helps if *every* process that annotates wraps its
    ``get_annotations()`` calls in this context manager — mosaic itself does no
    locking, so a single unwrapped kernel is enough to reintroduce corruption.

    Parameters
    ----------
    timeout
        Seconds to wait for the lock before raising. Annotating a large variant
        space can legitimately take several minutes, hence the generous default.
    poll
        Seconds between acquisition attempts.
    """
    import fcntl  # POSIX only; fine on the cluster and on macOS

    db = Path(path) if path is not None else varsome_db_path()
    lock_path = db.with_suffix(db.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    fh = open(lock_path, "w")
    deadline = time.monotonic() + timeout
    acquired = False

    try:
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Could not acquire Varsome cache lock within {timeout}s "
                        f"({lock_path}). Another annotation job is likely still "
                        f"running; if none is, remove the lock file."
                    )
                if verbose:
                    print("  … waiting for Varsome cache lock")
                time.sleep(poll)

        yield db

    finally:
        if acquired:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


# ── Varsome cache: row-level validation ───────────────────────────────────────

def _varsome_classify_row(row) -> str:
    """Return ``'ok'`` or a short reason string for why *row* should be dropped.

    Checks are structural, not semantic: a row is kept if it has the right field
    count and its two index fields look like a genome tag and a variant id.
    Interleaved writes that happen to land on exactly 18 fields are still caught
    by the index-field checks in most cases.

    Note this deliberately says nothing about *value* formatting — a row with a
    ``'148386525.0'`` position is structurally perfect. That class of problem is
    handled by :func:`_varsome_normalize_position` instead.
    """
    if len(row) != VARSOME_N_FIELDS:
        return f"field_count({len(row)})"

    genome, variant_id = row[0].strip(), row[1].strip()

    # A repeated header block means a process rewrote the file mid-append.
    if genome == "Genome" and variant_id == "Variant ID":
        return "repeat_header"

    if genome not in VARSOME_VALID_GENOMES:
        return f"bad_genome({genome[:20]!r})"

    # Variant ids are 'chrN:pos:ref/alt'; anything missing either separator is a
    # fragment of a torn write. Note that legitimate rows may have every field
    # after the id blank (e.g. 'chr13:28602226:*/A+AAG'), so blankness alone is
    # not grounds for dropping.
    if ":" not in variant_id or "/" not in variant_id:
        return f"bad_variant_id({variant_id[:30]!r})"

    return "ok"


def _varsome_n_populated(row) -> int:
    """Count non-empty fields — used to pick the better of two duplicate rows."""
    return sum(1 for f in row if str(f).strip() != "")


def _varsome_normalize_position(row) -> bool:
    """Rewrite *row*'s ``Position`` field to a bare integer string, in place.

    ``load_database`` reads the cache with ``low_memory=True``, so pandas infers
    a dtype per chunk. A cache mixing ``'148386525.0'``, ``'101917521'`` and
    blank positions makes that column collapse to ``object`` holding a mix of
    ints, floats and strings, and a downstream ``int()`` on the string form
    raises ``invalid literal for int() with base 10: '148386525.0'``.

    Normalising every position to one representation removes the heterogeneity.
    Blank positions are left blank — they are legitimate (see the ``*/A+AAG``
    style rows, which carry no resolved coordinate) and pandas reads them as
    NaN rather than as a string.

    Returns
    -------
    bool
        True if the field was changed.
    """
    raw = str(row[VARSOME_POSITION_IDX]).strip()

    if raw == "" or raw.lower() in {"nan", "none"}:
        if row[VARSOME_POSITION_IDX] != "":
            row[VARSOME_POSITION_IDX] = ""
            return True
        return False

    try:
        val = float(raw)
    except ValueError:
        return False  # unparseable — leave it; the structural checks own this

    if not val.is_integer():
        return False  # genuinely fractional; don't silently truncate

    canonical = str(int(val))
    if canonical != raw:
        row[VARSOME_POSITION_IDX] = canonical
        return True
    return False


def _varsome_position_style(row) -> str:
    """Classify a row's ``Position`` formatting as ``'int'``, ``'float'``,
    ``'blank'`` or ``'other'`` — used for reporting in :func:`inspect_varsome_db`."""
    raw = str(row[VARSOME_POSITION_IDX]).strip()
    if raw == "" or raw.lower() in {"nan", "none"}:
        return "blank"
    try:
        float(raw)
    except ValueError:
        return "other"
    return "float" if "." in raw else "int"


# ── Varsome cache: inspection ─────────────────────────────────────────────────

def inspect_varsome_db(path=None, *, max_examples: int = 10,
                       verbose: bool = True) -> dict:
    """Report on the health of the cache without modifying it.

    Covers both failure modes: structural damage (bad field counts, torn lines)
    and ``Position`` formatting heterogeneity. A cache that is structurally
    perfect but mixes ``'148386525.0'`` with ``'101917521'`` will still break
    annotation, so ``needs_repair`` accounts for both.

    Returns
    -------
    dict
        ``{'path', 'n_lines', 'n_ok', 'n_bad', 'n_duplicate_keys',
        'position_styles', 'position_mixed', 'needs_repair', 'reasons',
        'examples'}``
    """
    def vprint(msg):
        if verbose:
            print(msg)

    db = Path(path) if path is not None else varsome_db_path()

    empty_result = {
        "path": str(db), "n_lines": 0, "n_ok": 0, "n_bad": 0,
        "n_duplicate_keys": 0, "position_styles": {}, "position_mixed": False,
        "needs_repair": False, "reasons": {}, "examples": [],
    }

    if not db.exists():
        vprint(f"  Cache does not exist: {db}")
        return empty_result

    reasons, examples = {}, []
    seen_keys, dup_keys = set(), 0
    pos_styles = {}
    n_lines = n_ok = 0

    with open(db, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)

        try:
            header = next(reader)
        except StopIteration:
            vprint(f"  Cache is empty: {db}")
            return empty_result

        if header != VARSOME_DB_COLUMNS:
            vprint(f"  ⚠️  Header differs from the expected {VARSOME_N_FIELDS}-column "
                   f"schema (found {len(header)} fields)")

        for line_no, row in enumerate(reader, start=2):
            n_lines += 1
            reason = _varsome_classify_row(row)

            if reason == "ok":
                n_ok += 1
                key = (row[0].strip(), row[1].strip())
                if key in seen_keys:
                    dup_keys += 1
                seen_keys.add(key)

                style = _varsome_position_style(row)
                pos_styles[style] = pos_styles.get(style, 0) + 1
            else:
                reasons[reason] = reasons.get(reason, 0) + 1
                if len(examples) < max_examples:
                    examples.append((line_no, reason, ",".join(row)[:160]))

    n_bad = n_lines - n_ok

    # Only int-vs-float heterogeneity matters; blanks read back as NaN and do
    # not push the column to object dtype on their own.
    numeric_styles = {s for s in pos_styles if s in {"int", "float"}}
    position_mixed = len(numeric_styles) > 1 or "other" in pos_styles

    needs_repair = bool(n_bad or dup_keys or position_mixed)

    vprint(f"  Cache: {db}")
    vprint(f"  {n_lines} data line(s): {n_ok} valid, {n_bad} malformed, "
           f"{dup_keys} duplicate key(s)")
    vprint(f"  Position formatting: {pos_styles}")
    if position_mixed:
        vprint("  ⚠️  Mixed Position formatting — this causes "
               "\"invalid literal for int()\" during annotation")
    if reasons:
        vprint(f"  Drop reasons: {reasons}")
    for line_no, reason, preview in examples:
        vprint(f"    line {line_no}  [{reason}]  {preview}")

    return {
        "path": str(db),
        "n_lines": n_lines,
        "n_ok": n_ok,
        "n_bad": n_bad,
        "n_duplicate_keys": dup_keys,
        "position_styles": pos_styles,
        "position_mixed": position_mixed,
        "needs_repair": needs_repair,
        "reasons": reasons,
        "examples": examples,
    }


# ── Varsome cache: repair ─────────────────────────────────────────────────────

def repair_varsome_db(path=None, *, backup: bool = True, dry_run: bool = False,
                      dedupe: bool = True, normalize_positions: bool = True,
                      verbose: bool = True) -> dict:
    """Drop malformed lines, normalise value formatting, and rewrite atomically.

    Handles both failure modes described in the module docstring: structural
    damage from interleaved appends, and ``Position`` formatting heterogeneity.

    Retained rows are re-serialised with :mod:`csv` using minimal quoting, which
    reproduces the file's existing style: fields containing commas (e.g.
    ``"coding, splicing"``, ``"Likely Pathogenic, Pathogenic"``) are quoted and
    everything else is bare.

    Parameters
    ----------
    backup
        Copy the file to ``<name>.<timestamp>.bak`` before rewriting.
    dry_run
        Report what would change without writing anything.
    dedupe
        Collapse duplicate ``(Genome, Variant ID)`` keys, keeping whichever row
        has more populated fields. ``_search_db`` would otherwise return
        multiple rows for a single variant.
    normalize_positions
        Rewrite the ``Position`` field to a consistent bare-integer form.

    Returns
    -------
    dict
        ``{'path', 'backup', 'n_before', 'n_kept', 'n_dropped', 'n_deduped',
        'n_positions_fixed', 'reasons'}``
    """
    def vprint(msg):
        if verbose:
            print(msg)

    db = Path(path) if path is not None else varsome_db_path()

    if not db.exists():
        raise FileNotFoundError(f"Varsome cache not found: {db}")

    # ── Read and classify ────────────────────────────────────────────────────
    kept_rows, reasons = [], {}
    n_before = 0

    with open(db, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)

        try:
            header = next(reader)
        except StopIteration:
            header = list(VARSOME_DB_COLUMNS)

        for row in reader:
            n_before += 1
            reason = _varsome_classify_row(row)
            if reason == "ok":
                kept_rows.append(row)
            else:
                reasons[reason] = reasons.get(reason, 0) + 1

    n_dropped = n_before - len(kept_rows)

    # ── Normalise the Position field ─────────────────────────────────────────
    n_positions_fixed = 0
    if normalize_positions:
        for row in kept_rows:
            if _varsome_normalize_position(row):
                n_positions_fixed += 1

    # ── Deduplicate on the index key ─────────────────────────────────────────
    n_deduped = 0
    if dedupe:
        best = {}
        for row in kept_rows:
            key = (row[0].strip(), row[1].strip())
            if key not in best:
                best[key] = row
            else:
                n_deduped += 1
                # Keep whichever row carries more information.
                if _varsome_n_populated(row) > _varsome_n_populated(best[key]):
                    best[key] = row
        kept_rows = list(best.values())

    vprint(f"  Cache: {db}")
    vprint(f"  {n_before} data line(s) → {len(kept_rows)} kept, "
           f"{n_dropped} malformed dropped, {n_deduped} duplicate(s) collapsed, "
           f"{n_positions_fixed} position(s) reformatted")
    if reasons:
        vprint(f"  Drop reasons: {reasons}")

    if dry_run:
        vprint("  (dry run — nothing written)")
        return {"path": str(db), "backup": None, "n_before": n_before,
                "n_kept": len(kept_rows), "n_dropped": n_dropped,
                "n_deduped": n_deduped, "n_positions_fixed": n_positions_fixed,
                "reasons": reasons}

    nothing_to_do = (n_dropped == 0 and n_deduped == 0
                     and n_positions_fixed == 0 and header == VARSOME_DB_COLUMNS)
    if nothing_to_do:
        vprint("  ✓ Cache already clean — not rewriting")
        return {"path": str(db), "backup": None, "n_before": n_before,
                "n_kept": len(kept_rows), "n_dropped": 0, "n_deduped": 0,
                "n_positions_fixed": 0, "reasons": {}}

    # ── Back up ──────────────────────────────────────────────────────────────
    backup_path = None
    if backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = db.with_suffix(f"{db.suffix}.{stamp}.bak")
        shutil.copy2(db, backup_path)
        vprint(f"  ✓ Backup: {backup_path}")

    # ── Write to a temp file, then atomically replace ────────────────────────
    tmp_path = db.with_suffix(db.suffix + ".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(VARSOME_DB_COLUMNS)
        writer.writerows(kept_rows)

    os.replace(tmp_path, db)
    vprint(f"  ✓ Rewrote {db}")

    # ── Verify the repaired file parses the way mosaic reads it ──────────────
    check = pd.read_csv(db, index_col=[0, 1], low_memory=False)
    if check.shape[1] != VARSOME_N_FIELDS - 2:
        raise AssertionError(
            f"Repaired cache has {check.shape[1]} data columns, expected "
            f"{VARSOME_N_FIELDS - 2}. Restore from {backup_path} and investigate."
        )
    if check.shape[0] != len(kept_rows):
        raise AssertionError(
            f"Repaired cache parsed {check.shape[0]} rows but {len(kept_rows)} "
            f"were written. Restore from {backup_path} and investigate."
        )
    # Position must now infer as a single numeric dtype, not object — object
    # dtype here means the int() failure will recur on the next annotation run.
    if normalize_positions and "Position" in check.columns:
        if check["Position"].dtype == object:
            raise AssertionError(
                "Position still infers as object dtype after repair. Restore "
                f"from {backup_path} and investigate."
            )
        vprint(f"  ✓ Verified: parses as {check.shape[0]} variants × "
               f"{check.shape[1]} annotation columns, "
               f"Position dtype={check['Position'].dtype}")
    else:
        vprint(f"  ✓ Verified: parses as {check.shape[0]} variants × "
               f"{check.shape[1]} annotation columns")

    return {"path": str(db), "backup": str(backup_path) if backup_path else None,
            "n_before": n_before, "n_kept": len(kept_rows),
            "n_dropped": n_dropped, "n_deduped": n_deduped,
            "n_positions_fixed": n_positions_fixed, "reasons": reasons}


def reset_varsome_db(path=None, *, backup: bool = True,
                     verbose: bool = True) -> dict:
    """Clear the cache back to a bare header.

    Preferred over ``rm``-ing the file: the header-only form is guaranteed to
    parse, whereas a missing file depends on how ``load_database`` handles a
    non-existent path. Every variant is re-pulled from the API on the next run,
    so reach for :func:`repair_varsome_db` first.
    """
    def vprint(msg):
        if verbose:
            print(msg)

    db = Path(path) if path is not None else varsome_db_path()
    db.parent.mkdir(parents=True, exist_ok=True)

    backup_path = None
    if backup and db.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = db.with_suffix(f"{db.suffix}.{stamp}.bak")
        shutil.copy2(db, backup_path)
        vprint(f"  ✓ Backup: {backup_path}")

    tmp_path = db.with_suffix(db.suffix + ".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh, lineterminator="\n").writerow(VARSOME_DB_COLUMNS)
    os.replace(tmp_path, db)

    vprint(f"  ✓ Reset {db} to a bare header")
    return {"path": str(db), "backup": str(backup_path) if backup_path else None}


# ── Annotation ────────────────────────────────────────────────────────────────

def _is_varsome_cache_dtype_error(exc) -> bool:
    """True if *exc* is the mixed-``Position``-dtype failure.

    Deliberately narrow: matches only ``int()`` refusing a decimal *string*,
    which cannot arise from anything other than a Position field read back as
    text. Other ``ValueError``s (genuinely bad input) must propagate.
    """
    if not isinstance(exc, ValueError):
        return False
    msg = str(exc)
    return "invalid literal for int()" in msg and "." in msg


def annotate_with_repair(assay, *, max_attempts: int = 2, lock: bool = True,
                         repair_on_parse_error: bool = True,
                         verbose: bool = True):
    """Call ``assay.get_annotations()``, repairing the cache on a cache-shaped error.

    Two error classes originate from the on-disk cache rather than from the
    request, and both are worth repairing and retrying rather than surfacing as
    a lost chunk:

    * ``ParserError`` — torn lines from concurrent appends.
    * ``ValueError: invalid literal for int() with base 10: '148386525.0'`` —
      a ``Position`` column that infers as ``object`` because its formatting is
      inconsistent across the file.

    Any other exception propagates unchanged for the caller's retry logic.

    Parameters
    ----------
    lock
        Hold the advisory cache lock for the duration of the call. Set False
        only if the caller already holds it.
    repair_on_parse_error
        Repair and retry on the cache-shaped errors above. Set False to let
        them propagate.
    """
    def vprint(msg):
        if verbose:
            print(msg)

    from pandas.errors import ParserError

    for attempt in range(1, max_attempts + 1):
        try:
            if lock:
                with varsome_db_lock(verbose=verbose):
                    return assay.get_annotations()
            return assay.get_annotations()

        except (ParserError, ValueError) as exc:
            is_cache_error = (isinstance(exc, ParserError)
                              or _is_varsome_cache_dtype_error(exc))
            if not is_cache_error or not repair_on_parse_error or attempt == max_attempts:
                raise
            vprint(f"  ⚠️  Cache-derived failure ({exc}) — repairing and retrying")
            # The lock from the attempt above is already released here, so the
            # repair runs unlocked and the next iteration re-acquires it.
            repair_varsome_db(verbose=verbose)

    raise AssertionError("unreachable: retry loop exited without returning")


def annotate_assay_in_chunks(
    assay,
    *,
    chunk_size: int = 1000,
    max_retries: int = 3,
    backoff_sec: float = 5.0,
    add_col_attrs: bool = True,
    id_key: str = "id",
    use_cache_lock: bool = True,
    repair_on_parse_error: bool = True,
    verbose: bool = True,
) -> tuple:
    """Annotate every variant on *assay* via chunked ``get_annotations()`` calls.

    The MissionBio annotation endpoint caps the number of variants per request,
    so the variant axis is sliced into chunks. Results are re-keyed on variant
    id and reindexed against ``assay.col_attrs[id_key]`` before being written
    back, so a short or reordered chunk yields NaN for those variants rather
    than silently shifting every annotation by an offset.

    Each chunk goes through :func:`annotate_with_repair`, which serialises
    access to the shared on-disk Varsome cache and repairs it in place if a
    previous run left it corrupt or inconsistently formatted. The lock is taken
    and released per chunk rather than held across the whole loop, so a long
    annotation job does not block every other process in the environment for its
    full duration.

    Chunk-level failures are retried with linear backoff and then recorded
    rather than raised, so one API timeout does not discard the work already
    done. Mutates ``assay.col_attrs`` in place when ``add_col_attrs`` is True.

    Parameters
    ----------
    assay
        Mosaic DNA assay (annotated in place).
    chunk_size
        Variants per API call.
    max_retries, backoff_sec
        Retry policy; sleep is ``backoff_sec * attempt``.
    add_col_attrs
        Write the merged annotations back as ``col_attrs``.
    id_key
        ``col_attrs`` key holding variant ids.
    use_cache_lock
        Hold the advisory Varsome cache lock around each chunk. Only effective
        if every other annotating process does the same.
    repair_on_parse_error
        Repair the cache and retry when a chunk fails with a cache-derived
        error.

    Returns
    -------
    merged : pd.DataFrame
        Annotations indexed by variant id, reindexed to the assay's variant
        order (NaN where a variant was not annotated).
    failed : list of (start, stop, error_repr)
        Chunk ranges that exhausted their retries.
    """
    def vprint(msg):
        if verbose:
            print(msg)

    ids = np.asarray(assay.col_attrs[id_key], dtype=object)
    n_variants = len(ids)
    n_chunks = int(np.ceil(n_variants / chunk_size)) if n_variants else 0

    vprint(f"  Annotating {n_variants} variants in {n_chunks} chunk(s) of {chunk_size}")

    frames, failed = [], []
    for k, start in enumerate(range(0, n_variants, chunk_size), start=1):
        stop = min(start + chunk_size, n_variants)
        # Slice the assay directly — no need to copy the parent Sample.
        sub = assay[:, start:stop]

        ann = None
        for attempt in range(1, max_retries + 1):
            try:
                ann = annotate_with_repair(
                    sub,
                    lock=use_cache_lock,
                    repair_on_parse_error=repair_on_parse_error,
                    verbose=verbose,
                )
                break
            except Exception as exc:  # rate limit / timeout / transient API error
                if attempt == max_retries:
                    failed.append((start, stop, repr(exc)))
                    vprint(f"  ✗ chunk {k}/{n_chunks} [{start}:{stop}] failed: {exc}")
                else:
                    time.sleep(backoff_sec * attempt)

        if ann is None:
            continue

        ann = ann.copy()
        chunk_ids = ids[start:stop]
        if ann.shape[0] == len(chunk_ids) and not ann.index.isin(chunk_ids).all():
            ann.index = chunk_ids
        frames.append(ann)

        if verbose and (k % 10 == 0 or k == n_chunks):
            vprint(f"  … {k}/{n_chunks} chunks ({stop}/{n_variants} variants)")

    if not frames:
        raise RuntimeError("All annotation chunks failed — nothing to merge.")

    merged = pd.concat(frames, axis=0)
    merged = merged[~merged.index.duplicated(keep="first")]
    n_hit = int(merged.index.isin(ids).sum())
    merged = merged.reindex(ids)

    vprint(f"  Annotated {n_hit}/{n_variants} variants "
           f"({n_variants - n_hit} unannotated → NaN)")

    if add_col_attrs:
        for col, content in merged.items():
            assay.add_col_attr(col, np.asarray(content.values, dtype=object))
        vprint(f"  ✓ Added {merged.shape[1]} annotation col_attrs")

    if failed:
        vprint(f"  ⚠️  {len(failed)} chunk(s) failed; ranges: "
               f"{[(a, b) for a, b, _ in failed]}")

    return merged, failed


# ── Cell subsetting ───────────────────────────────────────────────────────────

def subset_assay_to_cells(
    assay,
    barcodes,
    *,
    preserve_assay_order: bool = True,
    require_any: bool = True,
    verbose: bool = True,
):
    """Return a copy of *assay* restricted to *barcodes*.

    Accepts a barcode iterable or any object with an ``.index`` (e.g. a
    per-cell DataFrame such as ``df_aberrancies``). Row attrs are sanitized
    first, since 0-D or wrong-length entries break mosaic slicing.

    Parameters
    ----------
    preserve_assay_order
        Keep the assay's own barcode ordering (recommended — keeps row_attrs
        interpretable and deduplicates). If False, use the order given.
    require_any
        Raise if no barcode overlaps; the usual cause is a table indexed by
        variant rather than by cell.
    """
    def vprint(msg):
        if verbose:
            print(msg)

    if hasattr(barcodes, "index") and not isinstance(barcodes, (list, tuple, np.ndarray)):
        barcodes = barcodes.index

    target = pd.Index(pd.Series(list(barcodes)).astype(str)).unique()
    bc_all = np.asarray(assay.barcodes(), dtype=str)
    present = set(bc_all)

    if preserve_assay_order:
        bars_keep = bc_all[np.isin(bc_all, list(target))]
    else:
        bars_keep = np.asarray([b for b in target if b in present], dtype=str)

    n_found, n_target = bars_keep.size, len(target)
    vprint(f"  Requested {n_target} unique barcodes; matched {n_found}")

    if n_found == 0 and require_any:
        raise ValueError(
            "No overlap between the requested barcodes and the assay barcodes. "
            "Is the source table indexed by variant rather than by cell?"
        )
    if n_found < n_target:
        missing = [b for b in target if b not in present]
        vprint(f"  ⚠️  {len(missing)} not present in assay (e.g. {missing[:3]})")

    sanitize_row_attrs(assay)
    out = copy.copy(assay)[np.asarray(bars_keep, dtype=object), :]
    vprint(f"  ✓ Subset: {len(out.barcodes())} cells × "
           f"{len(out.col_attrs['id'])} variants")
    return out


# ── Variant metadata ──────────────────────────────────────────────────────────

def variant_metadata_frame(
    assay,
    *,
    annotations=None,
    id_key: str = "id",
    include_subset_stats: bool = True,
    include_af_dp_stats: bool = False,
    ngt_layer: str = "NGT",
    sort_genomic: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build a per-variant metadata table for *assay*.

    Collects every 1-D ``col_attrs`` entry whose length matches the variant
    axis — which after :func:`annotate_assay_in_chunks` includes the API
    annotation columns alongside the native fields (amplicon, filtered, ADO
    score, and so on), so the annotation frame does not need to be passed in
    separately.

    Row order matches the exported matrix column order by default, so this
    table can be used directly as a column key for those matrices.

    Parameters
    ----------
    annotations
        Optional annotation DataFrame indexed by variant id, for the case where
        :func:`annotate_assay_in_chunks` was run with ``add_col_attrs=False``.
        Columns already present as ``col_attrs`` are suffixed ``_annot`` rather
        than overwriting.
    include_subset_stats
        Append per-variant genotype summaries computed over the cells present
        on *this* assay: WT/HET/HOM/missing counts, genotype rate, mutated
        fraction, and genotype-derived VAF. Holds one NGT frame in memory.
    include_af_dp_stats
        Also append mean AF and mean DP over genotyped cells only. This holds
        two full matrices in memory at once — avoid on the unfiltered variant
        space unless the cell subset is small.
    sort_genomic
        Sort by chromosome then position. Note this breaks the row-order
        correspondence with the exported matrix columns.

    Returns
    -------
    pd.DataFrame
        Indexed by variant id (index name ``id``).
    """
    def vprint(msg):
        if verbose:
            print(msg)

    ids = np.asarray(assay.col_attrs[id_key], dtype=object)
    n_variants = len(ids)

    # ── Native + annotation col_attrs ────────────────────────────────────────
    data, skipped = {}, []
    for key, val in assay.col_attrs.items():
        if key == id_key:
            continue
        arr = np.asarray(val)
        if arr.ndim == 1 and arr.shape[0] == n_variants:
            data[key] = arr
        else:
            skipped.append(key)

    meta = pd.DataFrame(data, index=pd.Index(ids, name=id_key))
    vprint(f"  {meta.shape[1]} col_attr field(s) for {n_variants} variants")
    if skipped:
        vprint(f"  ⚠️  Skipped non-1-D / length-mismatched col_attrs: {skipped}")

    # ── Parsed coordinates (handy for sorting and for joining to CNV output) ──
    if "chrom" not in meta.columns:
        meta.insert(0, "chrom", [extract_chr(str(v)) for v in ids])
    if "position" not in meta.columns:
        meta.insert(1, "position", [extract_position(v) for v in ids])

    # ── Optional externally supplied annotation frame ────────────────────────
    if annotations is not None and not annotations.empty:
        ann = annotations.reindex(ids)
        overlap = [c for c in ann.columns if c in meta.columns]
        if overlap:
            ann = ann.rename(columns={c: f"{c}_annot" for c in overlap})
            vprint(f"  ⚠️  {len(overlap)} annotation column(s) already present as "
                   f"col_attrs — suffixed '_annot'")
        meta = meta.join(ann)

    # ── Genotype summaries over the cells on this assay ──────────────────────
    if include_subset_stats:
        if ngt_layer not in assay.layers:
            vprint(f"  ⚠️  Layer '{ngt_layer}' absent — skipping genotype stats")
        else:
            ngt = assay.get_attribute(ngt_layer, constraint="row+col")
            n_cells = int(ngt.shape[0])

            n_wt = (ngt == 0).sum(axis=0)
            n_het = (ngt == 1).sum(axis=0)
            n_hom = (ngt == 2).sum(axis=0)
            n_missing = (ngt == 3).sum(axis=0)
            n_gt = (n_wt + n_het + n_hom).astype(float)
            gt_denom = n_gt.replace(0, np.nan)

            stats = pd.DataFrame({
                "n_cells_total": n_cells,
                "n_WT": n_wt,
                "n_HET": n_het,
                "n_HOM": n_hom,
                "n_missing": n_missing,
                "genotype_rate": (n_gt / n_cells) if n_cells else np.nan,
                "mutated_cell_frac": (n_het + n_hom) / gt_denom,
                "vaf_by_cell_count": 100.0 * (n_het + 2 * n_hom) / (2 * gt_denom),
            })
            stats.index = pd.Index(np.asarray(stats.index, dtype=object), name=id_key)

            if include_af_dp_stats:
                mask = (ngt != 3)
                for layer, out_col in (("AF", "mean_AF_genotyped"),
                                       ("DP", "mean_DP_genotyped")):
                    if layer not in assay.layers:
                        vprint(f"  ⚠️  Layer '{layer}' absent — skipping {out_col}")
                        continue
                    frame = assay.get_attribute(layer, constraint="row+col")
                    stats[out_col] = frame.where(mask).mean(axis=0).values
                    del frame
                    gc.collect()
                del mask

            del ngt
            gc.collect()

            meta = meta.join(stats.reindex(meta.index))
            vprint(f"  ✓ Added {stats.shape[1]} genotype summary column(s) over "
                   f"{n_cells} cells")

    if sort_genomic:
        order = np.lexsort((
            meta["position"].to_numpy(),
            np.array([extract_chr_num(str(v)) for v in meta.index]),
        ))
        meta = meta.iloc[order]
        vprint("  ⚠️  Sorted genomically — row order no longer matches matrix "
               "column order")

    return meta


# ── Matrix export ─────────────────────────────────────────────────────────────

def export_assay_layers(
    assay,
    out_dir: str,
    *,
    layers=None,
    suffix: str = "",
    label_key: str = "label",
    parquet: bool = False,
    formats: tuple = None,
    downcast: bool = True,
    size_warn_gb: float = 2.0,
    verbose: bool = True,
) -> dict:
    """Write ``cells × variants`` matrices for the requested layers.

    Each file carries a leading ``label`` column, matching the existing
    filtered-matrix export format.

    Parameters
    ----------
    layers
        Layer names to export; None exports every layer on the assay.
    suffix
        Appended to the layer name, e.g. ``'_all_variants'``.
    parquet
        Write ``.parquet`` instead of ``.csv`` — strongly preferred for the
        unfiltered variant space, where CSVs run to multiple GB. Ignored
        when `formats` is given.
    formats
        Explicit set of extensions to write, e.g. ``("csv", "parquet")``.
        The frame is fetched and downcast once and serialised to every
        format listed, rather than re-fetching per format. Overrides
        `parquet` when given; `parquet` remains the single-format default
        for backward compatibility. Note this must be a tuple — ``("csv")``
        is the string ``"csv"``, which iterates to ``('c', 's', 'v')``.
    downcast
        Cast NGT to int8 and float layers to float32 before writing.

    Returns
    -------
    dict
        ``{layer_name: path}`` when a single format is written, or
        ``{layer_name: {format: path}}`` when `formats` has more than one
        entry.
    """
    def vprint(msg):
        if verbose:
            print(msg)

    os.makedirs(out_dir, exist_ok=True)

    if layers is None:
        layers = list(assay.layers.keys())

    fmts = tuple(formats) if formats else (("parquet",) if parquet else ("csv",))
    multi = len(fmts) > 1

    n_cells = len(assay.barcodes())
    n_vars = len(assay.col_attrs["id"])

    try:
        labels = np.asarray(assay.get_labels()).astype(str)
    except Exception:
        vprint(f"  ⚠️  No '{label_key}' row attr — writing label='NA'")
        labels = np.full(n_cells, "NA", dtype=object)

    approx_gb = n_cells * n_vars * 6 / 1e9  # ≈6 bytes per value once serialised
    if approx_gb > size_warn_gb and "csv" in fmts:
        vprint(f"  ⚠️  Estimated ~{approx_gb:.1f} GB per CSV "
               f"({n_cells} × {n_vars}); consider formats=('parquet',)")

    written = {}
    for layer in layers:
        if layer not in assay.layers:
            vprint(f"  ⚠️  Layer '{layer}' not present — skipping "
                   f"(available: {list(assay.layers.keys())})")
            continue

        df = assay.get_attribute(layer, constraint="row+col")

        if downcast:
            try:
                if layer == "NGT":
                    df = df.astype(np.int8)
                elif all(pd.api.types.is_float_dtype(d) for d in df.dtypes):
                    df = df.astype(np.float32)
            except (TypeError, ValueError):
                pass  # leave dtype alone rather than fail the export

        df.insert(0, "label", labels)

        layer_paths = {}
        for fmt in fmts:
            path = os.path.join(out_dir, f"{layer}{suffix}.{fmt}")
            if fmt == "parquet":
                df.to_parquet(path, index=True)
            else:
                df.to_csv(path)
            layer_paths[fmt] = path
            vprint(f"  ✓ {path}  ({df.shape[0]} cells × {df.shape[1] - 1} variants)")

        written[layer] = layer_paths if multi else next(iter(layer_paths.values()))

        del df
        gc.collect()

    return written


def export_cell_subset_matrices(
    assay,
    barcodes,
    out_dir: str,
    *,
    annotate: bool = False,
    annot_chunk_size: int = 1000,
    use_cache_lock: bool = True,
    repair_on_parse_error: bool = True,
    layers=("NGT", "AF"),
    suffix: str = "_all_variants",
    parquet: bool = False,
    formats: tuple = None,
    downcast: bool = True,
    export_metadata: bool = True,
    metadata_stem: str = "variant_metadata",
    metadata_formats: tuple = ("csv",),
    include_subset_stats: bool = True,
    include_af_dp_stats: bool = False,
    verbose: bool = True,
) -> dict:
    """Annotate (optionally), subset to a cell set, and export the layer
    matrices together with the per-variant metadata table.

    Composes :func:`annotate_assay_in_chunks`, :func:`subset_assay_to_cells`,
    :func:`export_assay_layers` and :func:`variant_metadata_frame`. Annotation
    runs on the full assay *before* subsetting, since ``col_attrs`` live on the
    variant axis and the API cost is per variant, not per cell.

    The metadata table is written as CSV by default: it has one row per
    variant, so size is rarely the concern, and CSV is the format most likely
    to be opened by hand or joined downstream. Pass
    ``metadata_formats=("csv", "parquet")`` for both. Its row order matches
    the matrix column order.

    Parameters
    ----------
    use_cache_lock, repair_on_parse_error
        Forwarded to :func:`annotate_assay_in_chunks`; ignored when
        ``annotate=False``.

    Typical use — full variant space, cells from the aberrancy table::

        res = export_cell_subset_matrices(
            Sample_backup.dna,
            df_aberrancies,
            f"{folder_to_save}/dna/All_variants",
            annotate=True,
            parquet=True,
        )
        res["metadata"].head()

    Returns
    -------
    dict
        ``{'assay', 'annotations', 'annot_failed', 'metadata', 'paths'}``,
        where ``paths`` maps each layer name plus ``'metadata'`` to its
        file(s) — a single path unless multiple formats were requested for
        that item, per :func:`export_assay_layers`.
    """
    def vprint(msg):
        if verbose:
            print(msg)

    annotations, annot_failed = None, []

    if annotate:
        vprint("=" * 70)
        vprint("ANNOTATING VARIANTS")
        vprint("=" * 70)
        drop_scalar_row_attrs(assay)
        annotations, annot_failed = annotate_assay_in_chunks(
            assay,
            chunk_size=annot_chunk_size,
            use_cache_lock=use_cache_lock,
            repair_on_parse_error=repair_on_parse_error,
            verbose=verbose,
        )

    vprint("\n" + "=" * 70)
    vprint("SUBSETTING TO CELL SET")
    vprint("=" * 70)
    sub = subset_assay_to_cells(assay, barcodes, verbose=verbose)

    vprint("\n" + "=" * 70)
    vprint("EXPORTING MATRICES")
    vprint("=" * 70)
    paths = export_assay_layers(
        sub,
        out_dir,
        layers=layers,
        suffix=suffix,
        parquet=parquet,
        formats=formats,
        downcast=downcast,
        verbose=verbose,
    )

    metadata = None
    if export_metadata:
        vprint("\n" + "=" * 70)
        vprint("EXPORTING VARIANT METADATA")
        vprint("=" * 70)
        # Annotations are already col_attrs on `sub` when add_col_attrs ran, so
        # only pass the frame through if it would otherwise be lost.
        metadata = variant_metadata_frame(
            sub,
            annotations=None,
            include_subset_stats=include_subset_stats,
            include_af_dp_stats=include_af_dp_stats,
            verbose=verbose,
        )
        os.makedirs(out_dir, exist_ok=True)

        meta_paths = {}
        for fmt in metadata_formats:
            meta_path = os.path.join(out_dir, f"{metadata_stem}{suffix}.{fmt}")
            if fmt == "parquet":
                metadata.to_parquet(meta_path, index=True)
            else:
                metadata.to_csv(meta_path, index=True, index_label="id")
            meta_paths[fmt] = meta_path
            vprint(f"  ✓ {meta_path}  ({metadata.shape[0]} variants × "
                   f"{metadata.shape[1]} fields)")

        paths["metadata"] = meta_paths if len(metadata_formats) > 1 else next(iter(meta_paths.values()))

    vprint("=" * 70)

    return {
        "assay": sub,
        "annotations": annotations,
        "annot_failed": annot_failed,
        "metadata": metadata,
        "paths": paths,
    }


# ── Statistics ────────────────────────────────────────────────────────────────

def binomial_pval_right_tail(n: int, N: int, p: float) -> float:
    """Exact right-tail binomial p-value  P(X >= n)."""
    pval = 0.0
    for k in range(n, N + 1):
        pval += math.comb(N, k) * (p ** k) * ((1 - p) ** (N - k))
    return pval