"""Gene-level CNV exploration: map differential variants onto CNV amplicons and transcript regions.

Main entry points
-----------------
prepare_ploidy_region_plot_inputs
    Builds the inputs for a per-gene CNV ploidy plot: a clone x amplicon ploidy matrix restricted to
    the genes carrying differential variants, amplicon metadata labelled with the exon/intron of one
    selected transcript per gene, and the mutation labels placed on the amplicon covering each variant.
load_refseq_gtf_exons
    Reads the exon records of a RefSeq GTF/GFF file into the table expected by
    ``prepare_ploidy_region_plot_inputs(gtf_exons_refseq=...)``.
"""
from __future__ import annotations

import re
import time
from typing import Callable

import numpy as np
import pandas as pd

# Rank given to amplicons / variants without an exon or intron assignment (sorts them last).
_UNRANKED = 10**9

# Lower-cased strings treated as an empty label.
_EMPTY_LABEL_TOKENS = {"", "nan", "none", "na"}

# Chromosome order used for the ``chrom_rank`` column of the amplicon metadata.
_CHROM_ORDER = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]

_TRANSCRIPT_REGION_COLUMNS = [
    "gene_name",
    "CHROM_norm",
    "transcript_id_gtf_clean",
    "transcript_id_gtf_noversion_clean",
    "region_type",
    "region_number",
    "region_label",
    "region_start",
    "region_end",
    "region_rank",
]

_VARIANT_ID_COL_CANDIDATES = [
    "variant_id",
    "variant",
    "Variant",
    "id",
    "ID",
    "name",
    "Name",
    "condensed_format",
    "mutation_id",
    "locus",
]

_CHROM_COL_CANDIDATES = ["CHROM", "chrom", "chr", "Chromosome", "chromosome"]

_POS_COL_CANDIDATES = [
    "POS",
    "pos",
    "position",
    "Position",
    "start",
    "START",
    "start_pos",
    "genomic_position",
    "locus_pos",
]

_ANNOTATION_GENE_COL_CANDIDATES = [
    "gene_name",
    "gene",
    "Gene",
    "GENE",
    "Hugo_Symbol",
    "symbol",
    "gene_name_gtf",
]

_MUTATION_LABEL_COL_CANDIDATES = [
    "mutation",
    "Mutation",
    "protein_change",
    "Protein_Change",
    "HGVSp",
    "AAChange",
    "amino_acid_change",
    "condensed_format",
    "variant_name",
    "name",
]

# Annotation columns tried, in order, as keys when matching differential variant IDs.
_VARIANT_MATCH_COLS = [
    "variant_id",
    "_annotation_index_variant_id",
    "condensed_format",
    "mutation_id",
    "locus",
    "variant",
    "Variant",
    "name",
    "Name",
]

# Annotation columns carried into the variant lookup table (when present).
_VARIANT_LOOKUP_KEEP_COLS = [
    "variant_id",
    "_annotation_index_variant_id",
    "gene_name_plot",
    "CHROM_norm",
    "variant_pos",
    "exon",
    "mutation_label",
    "mutation_label_exon",
    "exon_match_status",
    "refseq_transcript_id",
    "refseq_transcript_id_noversion",
    "transcript_id_gtf",
    "gene_name_gtf",
    "exon_id",
    "exon_start",
    "exon_end",
]


# ============================================================
# Label / identifier parsing helpers
# ============================================================

def _clean_label(x):
    """Return ``str(x).strip()``, or ``""`` for NaN and 'nan'/'none'/'na' (case-insensitive)."""
    if pd.isna(x):
        return ""
    x = str(x).strip()
    if x.lower() in _EMPTY_LABEL_TOKENS:
        return ""
    return x


def _normalise_chrom(x):
    """Strip a 'chr' prefix (any case of CHR/Chr/chr) and map 'M' to 'MT'; NaN stays NaN."""
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    x = x.replace("CHR", "chr").replace("Chr", "chr")
    if x.startswith("chr"):
        x = x[3:]
    if x == "M":
        x = "MT"
    return x


def _strip_tx_version(x):
    """Drop a trailing '.N' version from a transcript ID; empty/missing values become NaN."""
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x.lower() in _EMPTY_LABEL_TOKENS:
        return np.nan
    return re.sub(r"\.\d+$", "", x)


def _as_variant_vector(x):
    """Convert a Series/Index/array/list/tuple/set (NaNs dropped) or a scalar to a list of str."""
    if isinstance(x, pd.Series):
        return x.dropna().astype(str).tolist()
    if isinstance(x, pd.Index):
        return x.dropna().astype(str).tolist()
    if isinstance(x, np.ndarray):
        return pd.Series(x).dropna().astype(str).tolist()
    if isinstance(x, (list, tuple, set)):
        return pd.Series(list(x)).dropna().astype(str).tolist()
    return [str(x)]


def _first_existing_col(df, candidates, required=False, label="column"):
    """Return the first of ``candidates`` present in ``df`` (None, or ValueError if ``required``)."""
    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise ValueError(
            f"Could not find required {label}. Tried: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def _parse_gene_from_variant_name(v):
    """Return the part before the first '_' of a 'GENE_...' style ID; NaN for 'chrN:pos' IDs."""
    v = str(v).strip()

    if re.match(r"^chr[\w]+:\d+", v, flags=re.IGNORECASE):
        return np.nan

    if "_" in v:
        return v.split("_")[0]

    return np.nan


def _parse_chrom_pos_from_variant_name(v):
    """Parse ``(chrom, pos)`` from a '[chr]N:pos...' ID, else ``(NaN, NaN)``."""
    v = str(v).strip()
    m = re.match(r"^(?:chr)?([0-9]{1,2}|X|Y|MT|M):(\d+)", v, flags=re.IGNORECASE)

    if m:
        return _normalise_chrom(m.group(1)), int(m.group(2))

    return np.nan, np.nan


def _pick_variant_id_col(df):
    """Return the first known variant-ID column present in ``df``, or None."""
    for c in _VARIANT_ID_COL_CANDIDATES:
        if c in df.columns:
            return c

    return None


# ============================================================
# Exon / intron label helpers
# ============================================================

def _sort_num_labels(vals):
    """Clean, de-duplicate and numerically sort labels; non-numeric labels sort last."""
    vals = [_clean_label(v) for v in vals]
    vals = [v for v in vals if v]

    def _sort_key(x):
        try:
            return int(float(x))
        except Exception:
            return _UNRANKED

    return sorted(set(vals), key=_sort_key)


def _region_rank_from_label(region_label):
    """Rank 'Exon N' as N and 'Intron N' as N + 0.5; anything else ranks last."""
    region_label = _clean_label(region_label)

    if region_label == "":
        return _UNRANKED

    m_exon = re.search(r"Exon\s+([0-9]+)", region_label, flags=re.IGNORECASE)
    if m_exon:
        return float(m_exon.group(1))

    m_intron = re.search(r"Intron\s+([0-9]+)", region_label, flags=re.IGNORECASE)
    if m_intron:
        return float(m_intron.group(1)) + 0.5

    return _UNRANKED


def _exon_rank_value(x):
    """
    Backward-compatible rank helper.

    Exon N   -> N
    Intron N -> N + 0.5
    "1,3"    -> smallest number (1)
    Blank    -> last
    """
    x = _clean_label(x)

    if x == "":
        return _UNRANKED

    if "exon" in x.lower() or "intron" in x.lower():
        return _region_rank_from_label(x)

    vals = []
    for part in str(x).split(","):
        part = part.strip()
        try:
            vals.append(int(float(part)))
        except Exception:
            pass

    if len(vals) == 0:
        return _UNRANKED

    return min(vals)


def _format_region_label(region_type, nums):
    """Format e.g. ``("exon", ["2", "3"])`` as ``"Exon 2,3"``; ``""`` if nothing to label."""
    nums = _sort_num_labels(nums)

    if len(nums) == 0:
        return ""

    nums_txt = ",".join(nums)

    if region_type == "exon":
        return f"Exon {nums_txt}"

    if region_type == "intron":
        return f"Intron {nums_txt}"

    return ""


# ============================================================
# Annotation-table column helpers
# ============================================================

def _fill_variant_id(ann, variant_id_col):
    """Set ``ann['variant_id']`` from ``variant_id_col``, falling back to the stripped index."""
    if variant_id_col is not None:
        ann["variant_id"] = ann[variant_id_col].astype(str).str.strip()
        empty_variant_id = ann["variant_id"].map(_clean_label) == ""
        ann.loc[empty_variant_id, "variant_id"] = ann.loc[
            empty_variant_id, "_annotation_index_variant_id"
        ]
    else:
        ann["variant_id"] = ann["_annotation_index_variant_id"].astype(str).str.strip()


def _add_chrom_norm(ann, log=None):
    """Add ``CHROM_norm`` from a chromosome column, else parse it from ``variant_id``."""
    chrom_col = _first_existing_col(
        ann,
        _CHROM_COL_CANDIDATES,
        required=False,
    )
    if log is not None:
        log(f"Chromosome column detected: {chrom_col}")

    if chrom_col is not None:
        ann["CHROM_norm"] = ann[chrom_col].map(_normalise_chrom)
    else:
        parsed = ann["variant_id"].map(_parse_chrom_pos_from_variant_name)
        ann["CHROM_norm"] = [x[0] for x in parsed]


def _add_variant_pos(ann, log=None):
    """Add numeric ``variant_pos`` from a position column, else parse it from ``variant_id``."""
    pos_col = _first_existing_col(
        ann,
        _POS_COL_CANDIDATES,
        required=False,
    )
    if log is not None:
        log(f"Position column detected: {pos_col}")

    if pos_col is not None:
        ann["variant_pos"] = pd.to_numeric(ann[pos_col], errors="coerce")
    else:
        parsed = ann["variant_id"].map(_parse_chrom_pos_from_variant_name)
        ann["variant_pos"] = [x[1] for x in parsed]


# ============================================================
# GTF helpers
# ============================================================

def _standardise_gtf_exons(gtf_exons, log):
    """
    Standardise gtf_exons_refseq column names and required fields.

    Returns None when the input is not a DataFrame or required columns are missing (logged), otherwise
    a copy with normalised chromosomes, integer coordinates/exon numbers and incomplete rows dropped.
    """
    if gtf_exons is None or not isinstance(gtf_exons, pd.DataFrame):
        return None

    gtf = gtf_exons.copy()

    if "exon_start" not in gtf.columns and "start" in gtf.columns:
        gtf = gtf.rename(columns={"start": "exon_start"})

    if "exon_end" not in gtf.columns and "end" in gtf.columns:
        gtf = gtf.rename(columns={"end": "exon_end"})

    if "gene_name_gtf" not in gtf.columns:
        gtf_gene_col = _first_existing_col(
            gtf,
            ["gene_name", "gene", "Gene", "GENE", "Hugo_Symbol", "symbol"],
            required=False,
        )
        if gtf_gene_col is not None:
            gtf["gene_name_gtf"] = gtf[gtf_gene_col].astype(str)

    if "CHROM_norm" not in gtf.columns:
        gtf_chrom_col = _first_existing_col(
            gtf,
            _CHROM_COL_CANDIDATES,
            required=False,
        )
        if gtf_chrom_col is not None:
            gtf["CHROM_norm"] = gtf[gtf_chrom_col].map(_normalise_chrom)

    if "transcript_id_gtf" not in gtf.columns:
        gtf_tx_col = _first_existing_col(
            gtf,
            ["transcript_id", "refseq_transcript_id", "transcript", "tx_id"],
            required=False,
        )
        if gtf_tx_col is not None:
            gtf["transcript_id_gtf"] = gtf[gtf_tx_col].astype(str)

    if "transcript_id_gtf_noversion" not in gtf.columns and "transcript_id_gtf" in gtf.columns:
        gtf["transcript_id_gtf_noversion"] = gtf["transcript_id_gtf"].map(_strip_tx_version)

    if "exon_id" not in gtf.columns:
        gtf["exon_id"] = ""

    required_gtf_cols = {
        "gene_name_gtf",
        "CHROM_norm",
        "exon_start",
        "exon_end",
        "exon_number",
        "transcript_id_gtf",
        "transcript_id_gtf_noversion",
    }

    missing = required_gtf_cols - set(gtf.columns)

    if missing:
        log(f"gtf_exons_refseq missing required columns: {sorted(missing)}")
        return None

    gtf["gene_name_gtf"] = gtf["gene_name_gtf"].astype(str)
    gtf["CHROM_norm"] = gtf["CHROM_norm"].map(_normalise_chrom)
    gtf["exon_start"] = pd.to_numeric(gtf["exon_start"], errors="coerce")
    gtf["exon_end"] = pd.to_numeric(gtf["exon_end"], errors="coerce")
    gtf["exon_number"] = pd.to_numeric(gtf["exon_number"], errors="coerce")
    gtf["transcript_id_gtf"] = gtf["transcript_id_gtf"].astype(str).map(_clean_label)
    gtf["transcript_id_gtf_noversion"] = (
        gtf["transcript_id_gtf_noversion"].astype(str).map(_clean_label)
    )

    gtf = gtf.dropna(
        subset=[
            "gene_name_gtf",
            "CHROM_norm",
            "exon_start",
            "exon_end",
            "exon_number",
            "transcript_id_gtf",
            "transcript_id_gtf_noversion",
        ]
    ).copy()

    if gtf.empty:
        return gtf

    gtf["exon_start"] = gtf["exon_start"].astype(int)
    gtf["exon_end"] = gtf["exon_end"].astype(int)
    gtf["exon_number"] = gtf["exon_number"].astype(int)

    return gtf


def _compute_annotation_exon_from_gtf(annotation_df, gtf_exons_refseq, log):
    """
    Compute exon/transcript annotation for variants from gtf_exons_refseq.

    Each variant row without both an ``exon`` and an ``exon_match_status`` is matched to the GTF exons
    of its (gene, chromosome) that contain its position. If the row names a transcript, hits on that
    transcript are preferred; ties are broken by sorting on transcript ID, exon number, start and end
    and taking the first. ``exon_match_status`` records the outcome ('matched_exon', 'missing_gene',
    'missing_chromosome', 'missing_position', 'no_gtf_gene_chrom_match', 'no_exon_overlap', or a
    GTF-unavailable/empty status).
    """
    ann = annotation_df.copy()

    if "_annotation_index_variant_id" not in ann.columns:
        ann["_annotation_index_variant_id"] = ann.index.astype(str).str.strip()

    variant_id_col = _pick_variant_id_col(ann)

    if "variant_id" not in ann.columns:
        _fill_variant_id(ann, variant_id_col)

    gene_col = _first_existing_col(
        ann,
        _ANNOTATION_GENE_COL_CANDIDATES,
        required=False,
    )

    if "gene_name_plot" not in ann.columns:
        if gene_col is not None:
            ann["gene_name_plot"] = ann[gene_col].astype(str)
        else:
            ann["gene_name_plot"] = ann["variant_id"].map(_parse_gene_from_variant_name)

    if "CHROM_norm" not in ann.columns:
        _add_chrom_norm(ann)

    if "variant_pos" not in ann.columns:
        _add_variant_pos(ann)

    for c, default in {
        "exon": "",
        "exon_match_status": "",
        "refseq_transcript_id": "",
        "refseq_transcript_id_noversion": "",
        "transcript_id_gtf": "",
        "gene_name_gtf": "",
        "exon_id": "",
        "exon_start": np.nan,
        "exon_end": np.nan,
    }.items():
        if c not in ann.columns:
            ann[c] = default

    gtf = _standardise_gtf_exons(gtf_exons_refseq, log)

    if gtf is None:
        ann["exon_match_status"] = ann["exon_match_status"].replace(
            "", "gtf_exons_refseq_not_available_or_invalid"
        )
        return ann

    if gtf.empty:
        ann["exon_match_status"] = ann["exon_match_status"].replace(
            "", "gtf_exons_refseq_empty"
        )
        return ann

    gtf_groups = {
        key: sub.copy()
        for key, sub in gtf.groupby(["gene_name_gtf", "CHROM_norm"])
    }

    for idx, row in ann.iterrows():
        existing_exon = _clean_label(row.get("exon", ""))
        existing_status = _clean_label(row.get("exon_match_status", ""))

        if existing_exon and existing_status:
            continue

        gene = _clean_label(row.get("gene_name_plot", ""))
        chrom = _normalise_chrom(row.get("CHROM_norm", ""))
        pos = pd.to_numeric(row.get("variant_pos", np.nan), errors="coerce")

        if gene == "":
            ann.loc[idx, "exon_match_status"] = "missing_gene"
            continue

        if _clean_label(chrom) == "":
            ann.loc[idx, "exon_match_status"] = "missing_chromosome"
            continue

        if pd.isna(pos):
            ann.loc[idx, "exon_match_status"] = "missing_position"
            continue

        hits = gtf_groups.get((gene, chrom))

        if hits is None or hits.empty:
            ann.loc[idx, "exon_match_status"] = "no_gtf_gene_chrom_match"
            continue

        hits = hits[
            (hits["exon_start"] <= int(pos))
            & (hits["exon_end"] >= int(pos))
        ].copy()

        if hits.empty:
            ann.loc[idx, "exon_match_status"] = "no_exon_overlap"
            continue

        tx_candidates = []

        for tx_col in ["transcript_id_gtf", "refseq_transcript_id"]:
            tx = _clean_label(row.get(tx_col, ""))
            if tx:
                tx_candidates.append(tx)
                tx_nover = _clean_label(_strip_tx_version(tx))
                if tx_nover:
                    tx_candidates.append(tx_nover)

        if tx_candidates:
            tx_hits = hits[
                hits["transcript_id_gtf"].isin(tx_candidates)
                | hits["transcript_id_gtf_noversion"].isin(tx_candidates)
            ].copy()

            if not tx_hits.empty:
                hits = tx_hits

        hits = hits.sort_values(
            [
                "transcript_id_gtf",
                "exon_number",
                "exon_start",
                "exon_end",
            ]
        )

        hit = hits.iloc[0]

        tx_full = _clean_label(hit["transcript_id_gtf"])
        tx_nover = _clean_label(hit["transcript_id_gtf_noversion"])

        ann.loc[idx, "exon"] = str(int(hit["exon_number"]))
        ann.loc[idx, "exon_match_status"] = "matched_exon"
        ann.loc[idx, "refseq_transcript_id"] = tx_full
        ann.loc[idx, "refseq_transcript_id_noversion"] = tx_nover
        ann.loc[idx, "transcript_id_gtf"] = tx_full
        ann.loc[idx, "gene_name_gtf"] = _clean_label(hit["gene_name_gtf"])
        ann.loc[idx, "exon_id"] = _clean_label(hit.get("exon_id", ""))
        ann.loc[idx, "exon_start"] = int(hit["exon_start"])
        ann.loc[idx, "exon_end"] = int(hit["exon_end"])

    return ann


def _transcript_ids_in(df):
    """Non-empty ``transcript_id_gtf`` then ``refseq_transcript_id`` values of ``df`` (when present)."""
    tx_candidates = []

    if "transcript_id_gtf" in df.columns:
        tx_candidates += (
            df["transcript_id_gtf"]
            .dropna()
            .astype(str)
            .map(_clean_label)
            .tolist()
        )

    if "refseq_transcript_id" in df.columns:
        tx_candidates += (
            df["refseq_transcript_id"]
            .dropna()
            .astype(str)
            .map(_clean_label)
            .tolist()
        )

    return [t for t in tx_candidates if t]


def _choose_gene_transcripts_from_diff(diff, ann=None, gtf_exons_refseq=None, log=None):
    """
    Choose one transcript per gene.

    Preference:
      1. most frequent transcript_id_gtf/refseq_transcript_id among the matched differential variants
      2. most frequent transcript_id_gtf/refseq_transcript_id in the full annotation table
      3. transcript with most distinct exon numbers in gtf_exons_refseq
      4. "" (no transcript)
    """
    gene_to_tx = {}

    genes = (
        diff["gene_name_plot"]
        .dropna()
        .astype(str)
        .map(_clean_label)
        .tolist()
    )
    genes = sorted(set([g for g in genes if g]))

    gtf = _standardise_gtf_exons(gtf_exons_refseq, log)

    for gene in genes:
        sub = diff[diff["gene_name_plot"].astype(str) == gene].copy()
        tx_candidates = _transcript_ids_in(sub)

        if tx_candidates:
            gene_to_tx[gene] = pd.Series(tx_candidates).value_counts().index[0]
            continue

        if ann is not None and isinstance(ann, pd.DataFrame) and "gene_name_plot" in ann.columns:
            ann_sub = ann[ann["gene_name_plot"].astype(str) == gene].copy()
            ann_tx_candidates = _transcript_ids_in(ann_sub)

            if ann_tx_candidates:
                gene_to_tx[gene] = pd.Series(ann_tx_candidates).value_counts().index[0]
                continue

        if gtf is not None and not gtf.empty:
            gtf_sub = gtf[gtf["gene_name_gtf"].astype(str) == gene].copy()

            if not gtf_sub.empty:
                tx_counts = (
                    gtf_sub.groupby("transcript_id_gtf")["exon_number"]
                    .nunique()
                    .sort_values(ascending=False)
                )

                if not tx_counts.empty:
                    gene_to_tx[gene] = _clean_label(tx_counts.index[0])
                    continue

        gene_to_tx[gene] = ""

    return gene_to_tx


def _build_transcript_regions_from_gtf(gtf_tx):
    """
    Build exon and intron intervals for selected transcripts.

    Intron N is the interval between Exon N and Exon N+1 (only for consecutive exon numbers and
    a non-empty gap).

    region_rank:
      Exon 1   -> 1.0
      Intron 1 -> 1.5
      Exon 2   -> 2.0
    """
    required = {
        "gene_name_gtf",
        "CHROM_norm",
        "exon_start",
        "exon_end",
        "exon_number",
        "transcript_id_gtf_clean",
        "transcript_id_gtf_noversion_clean",
    }

    missing = required - set(gtf_tx.columns)

    if missing:
        raise ValueError(f"gtf_tx missing required columns for transcript regions: {missing}")

    rows = []

    gtf_tx = gtf_tx.copy()
    gtf_tx["exon_number_int"] = pd.to_numeric(gtf_tx["exon_number"], errors="coerce")
    gtf_tx["exon_start"] = pd.to_numeric(gtf_tx["exon_start"], errors="coerce")
    gtf_tx["exon_end"] = pd.to_numeric(gtf_tx["exon_end"], errors="coerce")

    gtf_tx = gtf_tx.dropna(
        subset=[
            "exon_number_int",
            "exon_start",
            "exon_end",
            "gene_name_gtf",
            "CHROM_norm",
            "transcript_id_gtf_clean",
            "transcript_id_gtf_noversion_clean",
        ]
    )

    if gtf_tx.empty:
        return pd.DataFrame(columns=list(_TRANSCRIPT_REGION_COLUMNS))

    gtf_tx["exon_number_int"] = gtf_tx["exon_number_int"].astype(int)
    gtf_tx["exon_start"] = gtf_tx["exon_start"].astype(int)
    gtf_tx["exon_end"] = gtf_tx["exon_end"].astype(int)

    group_cols = [
        "gene_name_gtf",
        "CHROM_norm",
        "transcript_id_gtf_clean",
        "transcript_id_gtf_noversion_clean",
    ]

    for key, sub in gtf_tx.groupby(group_cols):
        gene, chrom, tx_full, tx_nover = key

        sub = (
            sub
            .sort_values(["exon_number_int", "exon_start", "exon_end"])
            .drop_duplicates(["exon_number_int", "exon_start", "exon_end"])
            .reset_index(drop=True)
        )

        for _, ex in sub.iterrows():
            exon_n = int(ex["exon_number_int"])

            rows.append({
                "gene_name": str(gene),
                "CHROM_norm": str(chrom),
                "transcript_id_gtf_clean": str(tx_full),
                "transcript_id_gtf_noversion_clean": str(tx_nover),
                "region_type": "exon",
                "region_number": exon_n,
                "region_label": f"Exon {exon_n}",
                "region_start": int(ex["exon_start"]),
                "region_end": int(ex["exon_end"]),
                "region_rank": float(exon_n),
            })

        for i in range(len(sub) - 1):
            ex_a = sub.iloc[i]
            ex_b = sub.iloc[i + 1]

            exon_a = int(ex_a["exon_number_int"])
            exon_b = int(ex_b["exon_number_int"])

            if exon_b != exon_a + 1:
                continue

            intron_n = exon_a

            intron_start = min(int(ex_a["exon_end"]), int(ex_b["exon_end"])) + 1
            intron_end = max(int(ex_a["exon_start"]), int(ex_b["exon_start"])) - 1

            if intron_start > intron_end:
                continue

            rows.append({
                "gene_name": str(gene),
                "CHROM_norm": str(chrom),
                "transcript_id_gtf_clean": str(tx_full),
                "transcript_id_gtf_noversion_clean": str(tx_nover),
                "region_type": "intron",
                "region_number": intron_n,
                "region_label": f"Intron {intron_n}",
                "region_start": intron_start,
                "region_end": intron_end,
                "region_rank": float(intron_n) + 0.5,
            })

    out = pd.DataFrame(rows)

    if out.empty:
        out = pd.DataFrame(columns=list(_TRANSCRIPT_REGION_COLUMNS))

    return out


# ============================================================
# Region assignment helpers
# ============================================================

def _select_best_region_hits(hits):
    """
    Prefer exon if an interval overlaps both exon and intron.
    Otherwise use intron.
    """
    if hits.empty:
        return hits

    exon_hits = hits[hits["region_type"] == "exon"].copy()
    if not exon_hits.empty:
        return exon_hits

    intron_hits = hits[hits["region_type"] == "intron"].copy()
    if not intron_hits.empty:
        return intron_hits

    return hits


def _empty_region_assignment():
    """Region fields for an amplicon with no exon/intron assignment."""
    return {
        "exon": "",
        "intron": "",
        "region_label": "",
        "region_type": "",
        "region_number": "",
        "region_rank": _UNRANKED,
        "exon_source": "",
    }


def _lookup_transcript_regions(
    transcript_region_groups_full,
    transcript_region_groups_nover,
    gene,
    chrom,
    tx,
    tx_nover,
):
    """Regions of the selected transcript, keyed by full ID first, then by unversioned ID."""
    hits = transcript_region_groups_full.get((gene, chrom, tx))

    if hits is None or hits.empty:
        hits = transcript_region_groups_nover.get((gene, chrom, tx_nover))

    return hits


def _assign_region_to_amplicon(
    amp,
    transcript_region_groups_full,
    transcript_region_groups_nover,
    gene_to_transcript,
):
    """Label one amplicon with the exon(s) (preferred) or intron(s) of its gene's selected transcript."""
    gene = str(amp["gene_name"])
    chrom = str(amp["CHROM_norm"])
    start = amp["start_pos"]
    end = amp["end_pos"]

    tx = _clean_label(gene_to_transcript.get(gene, ""))
    tx_nover = _clean_label(_strip_tx_version(tx))

    if tx == "":
        return _empty_region_assignment()

    hits = _lookup_transcript_regions(
        transcript_region_groups_full, transcript_region_groups_nover, gene, chrom, tx, tx_nover
    )

    if hits is None or hits.empty:
        return _empty_region_assignment()

    hits = hits[
        (hits["region_start"] <= end)
        & (hits["region_end"] >= start)
    ].copy()

    if hits.empty:
        return _empty_region_assignment()

    hits["overlap_bp"] = (
        np.minimum(hits["region_end"].values, end)
        - np.maximum(hits["region_start"].values, start)
        + 1
    )

    hits = hits[hits["overlap_bp"] > 0].copy()

    if hits.empty:
        return _empty_region_assignment()

    hits = _select_best_region_hits(hits)

    region_type = str(hits["region_type"].iloc[0])
    region_nums = _sort_num_labels(hits["region_number"].astype(str).tolist())
    region_label = _format_region_label(region_type, region_nums)
    region_rank = float(hits["region_rank"].min())

    exon = ",".join(region_nums) if region_type == "exon" else ""
    intron = ",".join(region_nums) if region_type == "intron" else ""

    return {
        "exon": exon,
        "intron": intron,
        "region_label": region_label,
        "region_type": region_type,
        "region_number": ",".join(region_nums),
        "region_rank": region_rank,
        "exon_source": f"gtf_selected_transcript:{tx}",
    }


# ============================================================
# Pipeline stages of prepare_ploidy_region_plot_inputs
# ============================================================

def _load_annotation_exon(annotation_df, annotation_exon, gtf_exons_refseq, log):
    """Step 1: copy the supplied exon annotation, or compute it from ``annotation_df`` + GTF."""
    if annotation_exon is not None:
        ann = annotation_exon.copy()
        log(f"Using supplied annotation_exon shape: {ann.shape}")
        return ann

    if annotation_df is None:
        raise ValueError(
            "Provide either annotation_df=... or annotation_exon=.... "
            "annotation_exon is no longer assumed to exist in notebook globals."
        )

    ann = _compute_annotation_exon_from_gtf(
        annotation_df=annotation_df,
        gtf_exons_refseq=gtf_exons_refseq,
        log=log,
    )

    log(f"Computed annotation_exon internally from annotation_df shape: {ann.shape}")
    return ann


def _standardise_annotation_table(ann, log):
    """Step 2 (in place): add variant_id, gene_name_plot, CHROM_norm, variant_pos and mutation labels."""
    ann["_annotation_index_variant_id"] = ann.index.astype(str).str.strip()

    variant_id_col = _pick_variant_id_col(ann)
    log(f"Variant ID column detected: {variant_id_col}")
    _fill_variant_id(ann, variant_id_col)

    gene_col = _first_existing_col(
        ann,
        _ANNOTATION_GENE_COL_CANDIDATES,
        required=False,
    )
    log(f"Gene column detected: {gene_col}")

    if gene_col is not None:
        ann["gene_name_plot"] = ann[gene_col].astype(str)
    elif "gene_name_plot" not in ann.columns:
        ann["gene_name_plot"] = ann["variant_id"].map(_parse_gene_from_variant_name)

    if "CHROM_norm" not in ann.columns:
        _add_chrom_norm(ann, log)

    if "variant_pos" not in ann.columns:
        _add_variant_pos(ann, log)

    if "mutation_label" not in ann.columns:
        label_col = _first_existing_col(
            ann,
            _MUTATION_LABEL_COL_CANDIDATES,
            required=False,
        )
        log(f"Mutation label column detected: {label_col}")

        if label_col is not None:
            ann["mutation_label"] = ann[label_col].astype(str)
        else:
            ann["mutation_label"] = ann["variant_id"].astype(str)

    ann["mutation_label"] = ann["mutation_label"].map(_clean_label)
    ann.loc[ann["mutation_label"] == "", "mutation_label"] = ann.loc[
        ann["mutation_label"] == "", "variant_id"
    ]

    if "exon" in ann.columns:
        ann["exon"] = ann["exon"].map(_clean_label)
    else:
        ann["exon"] = ""

    ann["mutation_label_exon"] = ann.apply(
        lambda r: (
            f"{r['mutation_label']} / Exon {r['exon']}"
            if _clean_label(r.get("exon", "")) != ""
            else str(r["mutation_label"])
        ),
        axis=1,
    )

    log("Standardised annotation table")


def _build_variant_lookup(ann, log):
    """Step 3: stack every ID-like annotation column into one key -> annotation table (first key wins)."""
    keep_cols = [c for c in _VARIANT_LOOKUP_KEEP_COLS if c in ann.columns]

    match_tables = []

    for c in _VARIANT_MATCH_COLS:
        if c not in ann.columns:
            continue

        tmp = ann[keep_cols].copy()
        tmp["_match_key"] = ann[c].astype(str).str.strip().values
        tmp["_match_source"] = c
        tmp = tmp[tmp["_match_key"].map(_clean_label) != ""]
        match_tables.append(tmp)

    if len(match_tables) == 0:
        raise ValueError("Could not build a variant matching table from annotation.")

    ann_match = (
        pd.concat(match_tables, axis=0, ignore_index=True)
        .drop_duplicates("_match_key", keep="first")
    )

    log(f"Variant lookup table built: {len(ann_match):,} keys")
    return ann_match


def _build_differential_table(differential_variants, ann_match, log):
    """Step 4: one row per differential variant, joined to the lookup and with gene/chrom/pos filled."""
    variant_vector = [_clean_label(v) for v in _as_variant_vector(differential_variants)]
    variant_vector = [v for v in variant_vector if v]

    log(f"Differential variants supplied: {len(variant_vector)}")
    log(str(variant_vector))

    diff = pd.DataFrame({"query_variant_id": variant_vector})
    diff["query_variant_id"] = diff["query_variant_id"].astype(str).str.strip()

    diff = diff.merge(
        ann_match,
        left_on="query_variant_id",
        right_on="_match_key",
        how="left",
        suffixes=("", "_ann"),
    )

    if "variant_id" in diff.columns:
        diff["matched_variant_id"] = diff["variant_id"]
    else:
        diff["matched_variant_id"] = np.nan

    diff["variant_id"] = diff["query_variant_id"]

    for c in [
        "gene_name_plot",
        "CHROM_norm",
        "variant_pos",
        "exon",
        "mutation_label",
        "mutation_label_exon",
        "exon_match_status",
        "refseq_transcript_id",
        "refseq_transcript_id_noversion",
        "transcript_id_gtf",
        "matched_variant_id",
    ]:
        if c not in diff.columns:
            diff[c] = np.nan

    missing_gene = (
        diff["gene_name_plot"].isna()
        | (diff["gene_name_plot"].astype(str).str.lower() == "nan")
        | (diff["gene_name_plot"].astype(str).str.strip() == "")
    )

    if missing_gene.any():
        diff.loc[missing_gene, "gene_name_plot"] = (
            diff.loc[missing_gene, "query_variant_id"].map(_parse_gene_from_variant_name)
        )

    for idx in diff.index:
        chrom_missing = (
            pd.isna(diff.loc[idx, "CHROM_norm"])
            or str(diff.loc[idx, "CHROM_norm"]).lower() == "nan"
            or str(diff.loc[idx, "CHROM_norm"]).strip() == ""
        )
        pos_missing = pd.isna(diff.loc[idx, "variant_pos"])

        if chrom_missing or pos_missing:
            chrom, pos = _parse_chrom_pos_from_variant_name(diff.loc[idx, "query_variant_id"])

            if chrom_missing:
                diff.loc[idx, "CHROM_norm"] = chrom

            if pos_missing:
                diff.loc[idx, "variant_pos"] = pos

    diff["mutation_label_exon"] = diff["mutation_label_exon"].map(_clean_label)
    diff.loc[diff["mutation_label_exon"] == "", "mutation_label_exon"] = diff.loc[
        diff["mutation_label_exon"] == "", "query_variant_id"
    ]

    diff["exon"] = diff["exon"].map(_clean_label)
    diff["variant_pos_numeric"] = pd.to_numeric(diff["variant_pos"], errors="coerce")

    log("Differential variant table built")
    return diff


def _build_amplicon_metadata(sample_all_vars, log):
    """Step 6: amplicon id/gene/chrom/start/end table from the CNV assay (start <= end enforced)."""
    cnv_attrs = sample_all_vars.cnv.col_attrs

    amp_meta = pd.DataFrame({
        "amplicon": cnv_attrs["id"],
        "gene_name": (
            cnv_attrs["gene_name"]
            if "gene_name" in cnv_attrs
            else sample_all_vars.cnv.get_gene_names()
        ),
        "CHROM": cnv_attrs["CHROM"],
        "start_pos": cnv_attrs["start_pos"],
    })

    if "end_pos" in cnv_attrs:
        amp_meta["end_pos"] = cnv_attrs["end_pos"]
    elif "end" in cnv_attrs:
        amp_meta["end_pos"] = cnv_attrs["end"]
    elif "stop_pos" in cnv_attrs:
        amp_meta["end_pos"] = cnv_attrs["stop_pos"]
    else:
        amp_meta["end_pos"] = amp_meta["start_pos"]

    amp_meta["gene_name"] = amp_meta["gene_name"].astype(str)
    amp_meta["CHROM_norm"] = amp_meta["CHROM"].map(_normalise_chrom)
    amp_meta["start_pos"] = pd.to_numeric(amp_meta["start_pos"], errors="coerce")
    amp_meta["end_pos"] = pd.to_numeric(amp_meta["end_pos"], errors="coerce")

    swap_mask = amp_meta["end_pos"] < amp_meta["start_pos"]
    amp_meta.loc[swap_mask, ["start_pos", "end_pos"]] = (
        amp_meta.loc[swap_mask, ["end_pos", "start_pos"]].values
    )

    amp_meta["chrom_rank"] = amp_meta["CHROM_norm"].map(
        {c: i for i, c in enumerate(_CHROM_ORDER)}
    )

    log(f"CNV amplicon metadata built: {len(amp_meta):,} amplicons")
    return amp_meta


def _restrict_amplicons_to_genes_of_interest(amp_meta, diff, df_ploidy_median, log):
    """Step 7: keep amplicons of the differential-variant genes that are columns of df_ploidy_median."""
    genes_of_interest = (
        diff["gene_name_plot"]
        .dropna()
        .astype(str)
        .replace("nan", np.nan)
        .dropna()
        .unique()
        .tolist()
    )
    genes_of_interest = [g for g in genes_of_interest if _clean_label(g) != ""]

    if len(genes_of_interest) == 0:
        raise ValueError(
            "Could not infer genes from differential_variants. "
            "Check diff display and annotation variant identifiers."
        )

    log(f"Genes of interest: {genes_of_interest}")

    amp_meta_interest = (
        amp_meta[amp_meta["gene_name"].astype(str).isin(genes_of_interest)]
        .copy()
        .reset_index(drop=True)
    )

    log(
        "CNV amplicons for genes of interest before df_ploidy filter: "
        f"{len(amp_meta_interest)}"
    )

    ploidy_cols = set(df_ploidy_median.columns)
    amp_ids = [a for a in amp_meta_interest["amplicon"].tolist() if a in ploidy_cols]

    amp_meta_interest = (
        amp_meta_interest[amp_meta_interest["amplicon"].isin(amp_ids)]
        .copy()
        .reset_index(drop=True)
    )

    if len(amp_ids) == 0:
        raise ValueError(
            "No CNV amplicons for differential-variant genes found in df_ploidy_median."
        )

    log(f"CNV amplicons retained in df_ploidy_median: {len(amp_meta_interest)}")

    for c, default in _empty_region_assignment().items():
        amp_meta_interest[c] = default

    return genes_of_interest, amp_meta_interest


def _build_selected_transcript_regions(gtf_exons_refseq, gene_to_transcript, genes_of_interest, log):
    """Step 8: exon/intron regions of the selected transcripts, plus lookups by full / unversioned ID."""
    log("Building selected-transcript exon/intron regions")

    transcript_regions = pd.DataFrame(columns=list(_TRANSCRIPT_REGION_COLUMNS))
    transcript_region_groups_full = {}
    transcript_region_groups_nover = {}

    gtf = _standardise_gtf_exons(gtf_exons_refseq, log)

    if gtf is None:
        log("gtf_exons_refseq not available or invalid; cannot build exon/intron regions")
    elif gtf.empty:
        log("gtf_exons_refseq is empty; cannot build exon/intron regions")
    else:
        gtf["transcript_id_gtf_clean"] = gtf["transcript_id_gtf"].astype(str).map(_clean_label)
        gtf["transcript_id_gtf_noversion_clean"] = (
            gtf["transcript_id_gtf_noversion"].astype(str).map(_clean_label)
        )

        selected_tx_full = [
            _clean_label(tx)
            for tx in gene_to_transcript.values()
            if _clean_label(tx) != ""
        ]

        selected_tx_nover = [
            _clean_label(_strip_tx_version(tx))
            for tx in selected_tx_full
            if _clean_label(tx) != ""
        ]

        selected_tx_full = set(selected_tx_full)
        selected_tx_nover = set([tx for tx in selected_tx_nover if _clean_label(tx) != ""])

        gtf_tx = gtf[
            (
                gtf["transcript_id_gtf_clean"].isin(selected_tx_full)
                | gtf["transcript_id_gtf_noversion_clean"].isin(selected_tx_nover)
            )
            & gtf["gene_name_gtf"].astype(str).isin(genes_of_interest)
        ].copy()

        log(f"GTF exon intervals after selected-transcript filter: {len(gtf_tx)}")

        transcript_regions = _build_transcript_regions_from_gtf(gtf_tx)

        n_exons = (
            int((transcript_regions["region_type"] == "exon").sum())
            if not transcript_regions.empty
            else 0
        )
        n_introns = (
            int((transcript_regions["region_type"] == "intron").sum())
            if not transcript_regions.empty
            else 0
        )

        log(f"Transcript regions built: {n_exons} exons, {n_introns} introns")

        if not transcript_regions.empty:
            transcript_region_groups_full = {
                key: sub.copy()
                for key, sub in transcript_regions.groupby(
                    ["gene_name", "CHROM_norm", "transcript_id_gtf_clean"]
                )
            }

            transcript_region_groups_nover = {
                key: sub.copy()
                for key, sub in transcript_regions.groupby(
                    ["gene_name", "CHROM_norm", "transcript_id_gtf_noversion_clean"]
                )
            }

    return transcript_regions, transcript_region_groups_full, transcript_region_groups_nover


def _label_differential_variant_regions(
    diff,
    gene_to_transcript,
    transcript_region_groups_full,
    transcript_region_groups_nover,
    log,
):
    """Step 10 (in place): region label/rank per variant; non-exonic variants get intron labels."""
    log("Adding intron labels to non-exonic differential variants where applicable")

    diff["intron"] = ""
    diff["region_label"] = ""
    diff["region_rank"] = diff["exon"].map(lambda x: _exon_rank_value(x))

    for idx, mut in diff.iterrows():
        exon = _clean_label(mut.get("exon", ""))

        if exon:
            diff.loc[idx, "region_label"] = f"Exon {exon}"
            diff.loc[idx, "region_rank"] = _exon_rank_value(exon)
            continue

        gene = str(mut["gene_name_plot"])
        chrom = str(mut["CHROM_norm"])
        pos = pd.to_numeric(mut["variant_pos"], errors="coerce")

        if pd.isna(pos):
            continue

        tx = _clean_label(gene_to_transcript.get(gene, ""))
        tx_nover = _clean_label(_strip_tx_version(tx))

        if tx == "":
            continue

        hits = _lookup_transcript_regions(
            transcript_region_groups_full, transcript_region_groups_nover, gene, chrom, tx, tx_nover
        )

        if hits is None or hits.empty:
            continue

        hits = hits[
            (hits["region_start"] <= pos)
            & (hits["region_end"] >= pos)
        ].copy()

        if hits.empty:
            continue

        hits = _select_best_region_hits(hits)

        region_type = str(hits["region_type"].iloc[0])
        region_num = str(int(hits["region_number"].iloc[0]))
        region_label = f"{region_type.capitalize()} {region_num}"

        diff.loc[idx, "region_label"] = region_label
        diff.loc[idx, "region_rank"] = float(hits["region_rank"].iloc[0])

        if region_type == "intron":
            diff.loc[idx, "intron"] = region_num

            base_label = _clean_label(diff.loc[idx, "mutation_label"])

            if base_label == "":
                base_label = _clean_label(diff.loc[idx, "query_variant_id"])

            diff.loc[idx, "mutation_label_exon"] = f"{base_label} / Intron {region_num}"


def _order_amplicons_and_build_plot_df(
    amp_meta_interest, df_ploidy_median, desired_plotting_order, log
):
    """Step 11: sort amplicons by gene then region rank/position; select and order the ploidy matrix."""
    log("Ordering amplicons by selected-transcript exon/intron order")

    amp_meta_interest["exon_rank"] = amp_meta_interest["region_rank"]

    amp_sel = (
        amp_meta_interest
        .sort_values(
            [
                "gene_name",
                "region_rank",
                "start_pos",
                "end_pos",
                "amplicon",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    amp_ids = amp_sel["amplicon"].tolist()
    plot_df = df_ploidy_median[amp_ids].copy()

    clone_order = [c for c in desired_plotting_order if c in plot_df.index]

    if len(clone_order) == 0:
        clone_order = list(plot_df.index)

    plot_df = plot_df.loc[clone_order]

    log(f"plot_df shape: {plot_df.shape}")
    return amp_sel, plot_df


def _assign_mutation_labels_to_amplicons(amp_sel, diff, log):
    """
    Step 12 (in place): append each variant's label to ``amp_sel['mutation_labels']``.

    The target is the first amplicon of the variant's gene covering its position (same chromosome
    preferred), else the gene amplicon with the nearest midpoint; without a position, the gene's
    first amplicon.
    """
    log("Assigning mutation labels to covering amplicons")

    amp_sel["mutation_labels"] = ""

    for n, (_, mut) in enumerate(diff.iterrows(), start=1):
        log(f"Assigning mutation label {n}/{len(diff)}: {mut.get('query_variant_id', '')}")

        gene = str(mut["gene_name_plot"])
        chrom = str(mut["CHROM_norm"])
        pos = pd.to_numeric(mut["variant_pos"], errors="coerce")
        mut_label = _clean_label(mut["mutation_label_exon"])

        gene_amps = amp_sel[amp_sel["gene_name"].astype(str) == gene].copy()

        if gene_amps.empty:
            log(f"  - skipped: no amplicons for gene {gene}")
            continue

        if pd.notna(pos):
            candidates = gene_amps[
                (gene_amps["start_pos"] <= pos)
                & (gene_amps["end_pos"] >= pos)
            ]

            candidates_chr = candidates[candidates["CHROM_norm"].astype(str) == chrom]

            if not candidates_chr.empty:
                candidates = candidates_chr

            if candidates.empty:
                tmp = gene_amps.copy()
                tmp_chr = tmp[tmp["CHROM_norm"].astype(str) == chrom]

                if not tmp_chr.empty:
                    tmp = tmp_chr

                tmp["amp_mid"] = (tmp["start_pos"] + tmp["end_pos"]) / 2
                tmp["dist"] = (tmp["amp_mid"] - pos).abs()
                candidates = tmp.sort_values("dist").head(1)
        else:
            candidates = gene_amps.head(1)

        if candidates.empty:
            log(f"  - skipped: no candidate amplicon for {mut_label}")
            continue

        amp_idx = candidates.index[0]
        old = _clean_label(amp_sel.loc[amp_idx, "mutation_labels"])

        amp_sel.loc[amp_idx, "mutation_labels"] = (
            mut_label if old == "" else f"{old}; {mut_label}"
        )

        log(
            f"  - assigned to {amp_sel.loc[amp_idx, 'amplicon']} "
            f"/ {amp_sel.loc[amp_idx, 'region_label']}"
        )


# ============================================================
# Public API
# ============================================================

def prepare_ploidy_region_plot_inputs(
    sample_all_vars,
    df_ploidy_median: pd.DataFrame,
    differential_variants,
    annotation_df: pd.DataFrame | None = None,
    annotation_exon: pd.DataFrame | None = None,
    desired_plotting_order=None,
    gtf_exons_refseq: pd.DataFrame | None = None,
    verbose: bool = True,
) -> dict:
    """
    Prepare CNV ploidy plotting inputs for the genes carrying differential variants.

    Differential variants are matched to an exon-annotated variant table, one transcript is chosen
    per gene, the CNV amplicons of those genes are labelled with the exon/intron of that transcript
    they overlap, ordered along the transcript, and each variant's label is attached to the amplicon
    that covers it.

    Either pass ``annotation_exon`` (an already exon-annotated table, used as is) or, preferably,
    ``annotation_df`` together with ``gtf_exons_refseq`` so exon annotation is computed internally.

    Parameters
    ----------
    sample_all_vars : mosaic/tapestri sample-like object
        Object whose ``.cnv.col_attrs`` holds ``id``, ``CHROM``, ``start_pos`` and optionally
        ``gene_name`` (else ``.cnv.get_gene_names()`` is used) and ``end_pos``/``end``/``stop_pos``
        (else end = start).
    df_ploidy_median : pandas.DataFrame
        Clone x CNV amplicon ploidy matrix (columns are amplicon IDs).
    differential_variants : list, tuple, set, pandas.Series, pandas.Index, numpy.ndarray or scalar
        Differential variant identifiers. They are matched against the annotation's ID-like columns
        (``variant_id``, the index, ``condensed_format``, ``mutation_id``, ``locus``, ``variant``,
        ``Variant``, ``name``, ``Name``); genes/positions missing after matching are parsed from
        'GENE_...' or 'chrN:pos...' style identifiers.
    annotation_df : pandas.DataFrame, optional
        Base variant annotation table, exon-annotated internally from ``gtf_exons_refseq``.
    annotation_exon : pandas.DataFrame, optional
        Already exon-annotated variant table. Takes precedence over ``annotation_df``.
    desired_plotting_order : list-like, optional
        Preferred clone order for the rows of ``plot_df``; clones not in ``df_ploidy_median`` are
        ignored. If none match, the original row order is kept.
    gtf_exons_refseq : pandas.DataFrame, optional
        RefSeq exon table (e.g. from :func:`load_refseq_gtf_exons`) used for exon and intron
        assignment. Without it amplicons are left unlabelled.
    verbose : bool, default True
        Print timestamped progress logs.

    Returns
    -------
    dict
        ``"plot_df"``: clone x selected-amplicon ploidy matrix;
        ``"amp_sel"``: selected amplicon metadata with exon/intron/region and ``mutation_labels``;
        ``"diff"``: differential variant mapping table;
        ``"gene_to_transcript"``: selected transcript per gene (``""`` if none);
        ``"transcript_regions"``: exon/intron intervals of the selected transcripts;
        ``"genes_of_interest"``: genes inferred from the differential variants;
        ``"annotation_exon"``: the (standardised) exon annotation table used.

    Raises
    ------
    ValueError
        If neither ``annotation_df`` nor ``annotation_exon`` is given, no variant lookup table can be
        built, no gene can be inferred from the variants, or none of their amplicons is a column of
        ``df_ploidy_median``.

    Notes
    -----
    Transcript choice per gene: most frequent transcript among the matched variants, else in the
    whole annotation table, else the GTF transcript with the most distinct exon numbers.
    Intron N spans the gap between consecutive Exon N and N+1. An amplicon overlapping both exons and
    introns is labelled with the exon(s). Region rank is N for Exon N and N + 0.5 for Intron N;
    unlabelled amplicons rank last (10**9).
    """

    def _log(msg):
        if verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    t0 = time.perf_counter()
    _log("Starting CNV ploidy region prep")

    if desired_plotting_order is None:
        desired_plotting_order = []
    else:
        desired_plotting_order = list(desired_plotting_order)

    # 1-2. Exon-annotated variant table with standard columns.
    ann = _load_annotation_exon(annotation_df, annotation_exon, gtf_exons_refseq, _log)
    _standardise_annotation_table(ann, _log)

    # 3-4. Match differential variants to the annotation.
    ann_match = _build_variant_lookup(ann, _log)
    diff = _build_differential_table(differential_variants, ann_match, _log)

    # 5. One transcript per gene.
    gene_to_transcript = _choose_gene_transcripts_from_diff(
        diff=diff,
        ann=ann,
        gtf_exons_refseq=gtf_exons_refseq,
        log=_log,
    )

    _log("Transcript selected per gene:")
    for gene, tx in gene_to_transcript.items():
        _log(f"  {gene}: {tx}")

    # 6-7. CNV amplicons of the genes of interest present in the ploidy matrix.
    amp_meta = _build_amplicon_metadata(sample_all_vars, _log)
    genes_of_interest, amp_meta_interest = _restrict_amplicons_to_genes_of_interest(
        amp_meta, diff, df_ploidy_median, _log
    )

    # 8. Exon/intron intervals of the selected transcripts.
    (
        transcript_regions,
        transcript_region_groups_full,
        transcript_region_groups_nover,
    ) = _build_selected_transcript_regions(
        gtf_exons_refseq, gene_to_transcript, genes_of_interest, _log
    )

    # 9. Assign each amplicon to a selected-transcript exon/intron region.
    _log("Assigning each amplicon to selected-transcript exon/intron region")

    for i, amp in amp_meta_interest.iterrows():
        if verbose and (i % 25 == 0 or i == len(amp_meta_interest) - 1):
            _log(f"Region assignment: amplicon {i + 1}/{len(amp_meta_interest)}")

        assigned = _assign_region_to_amplicon(
            amp=amp,
            transcript_region_groups_full=transcript_region_groups_full,
            transcript_region_groups_nover=transcript_region_groups_nover,
            gene_to_transcript=gene_to_transcript,
        )

        for k, v in assigned.items():
            amp_meta_interest.loc[i, k] = v

    _log(
        "Region assignment complete: "
        f"{(amp_meta_interest['region_type'] == 'exon').sum()} exon-labelled, "
        f"{(amp_meta_interest['region_type'] == 'intron').sum()} intron-labelled, "
        f"{(amp_meta_interest['region_type'] == '').sum()} unlabelled"
    )

    # 10. Exon/intron labels for the differential variants.
    _label_differential_variant_regions(
        diff,
        gene_to_transcript,
        transcript_region_groups_full,
        transcript_region_groups_nover,
        _log,
    )

    # 11-12. Order amplicons, build plot_df, attach mutation labels.
    amp_sel, plot_df = _order_amplicons_and_build_plot_df(
        amp_meta_interest, df_ploidy_median, desired_plotting_order, _log
    )
    _assign_mutation_labels_to_amplicons(amp_sel, diff, _log)

    elapsed = time.perf_counter() - t0

    _log(f"Finished CNV ploidy region prep in {elapsed:.1f} sec")
    _log(f"Genes: {genes_of_interest}")
    _log(f"Amplicons: {len(amp_sel)}")
    _log(f"Clones: {plot_df.shape[0]}")

    return {
        "plot_df": plot_df,
        "amp_sel": amp_sel,
        "diff": diff,
        "gene_to_transcript": gene_to_transcript,
        "transcript_regions": transcript_regions,
        "genes_of_interest": genes_of_interest,
        "annotation_exon": ann,
    }


def _gtf_attr(attr, keys):
    """Return the first ``key "value"`` (GTF) or ``key=value`` (GFF) attribute among ``keys``, else NaN."""
    attr = str(attr)

    for key in keys:
        m = re.search(rf'{re.escape(key)}\s+"([^"]+)"', attr)
        if m:
            return m.group(1)

        m = re.search(rf'{re.escape(key)}=([^;]+)', attr)
        if m:
            return m.group(1).strip()

    return np.nan


def load_refseq_gtf_exons(gtf_path) -> pd.DataFrame:
    """
    Load the exon records of a RefSeq GTF/GFF file.

    Parameters
    ----------
    gtf_path : str or path-like
        Tab-separated GTF/GFF file; lines starting with '#' are skipped. Only columns 1 (seqname),
        3 (feature), 4-5 (start/end), 7 (strand) and 9 (attributes) are read.

    Returns
    -------
    pandas.DataFrame
        One row per ``exon`` feature with columns ``gene_name_gtf``, ``CHROM_norm``, ``exon_start``,
        ``exon_end``, ``exon_number``, ``transcript_id_gtf``, ``transcript_id_gtf_noversion``,
        ``exon_id`` and ``strand`` (original row order). Rows without chromosome, gene, transcript or
        coordinates are dropped.

    Notes
    -----
    - Attributes are read in GTF (``key "value"``) or GFF (``key=value``) style. Gene name is taken
      from ``gene_name``, ``gene``, ``gene_id`` or ``Name`` (first found); transcript from
      ``transcript_id``, ``transcriptId`` or ``transcript``.
    - Chromosomes are normalised by stripping 'chr' (any case) and mapping 'M' to 'MT'.
    - Missing exon numbers are inferred per (gene, transcript, chromosome, strand) by ordering exons
      by start/end, ascending on '+' and descending on '-' strand.
    """
    gtf = pd.read_csv(
        gtf_path,
        sep="\t",
        comment="#",
        header=None,
        usecols=[0, 2, 3, 4, 6, 8],
        names=[
            "CHROM",
            "feature",
            "exon_start",
            "exon_end",
            "strand",
            "attribute",
        ],
        dtype={
            "CHROM": str,
            "feature": str,
            "strand": str,
            "attribute": str,
        },
    )

    gtf = gtf[gtf["feature"] == "exon"].copy()

    gtf["CHROM_norm"] = gtf["CHROM"].map(_normalise_chrom)

    gtf["gene_name_gtf"] = gtf["attribute"].map(
        lambda x: _gtf_attr(x, ["gene_name", "gene", "gene_id", "Name"])
    )

    gtf["transcript_id_gtf"] = gtf["attribute"].map(
        lambda x: _gtf_attr(x, ["transcript_id", "transcriptId", "transcript"])
    )

    gtf["transcript_id_gtf_noversion"] = (
        gtf["transcript_id_gtf"]
        .astype(str)
        .str.replace(r"\.\d+$", "", regex=True)
    )

    gtf["exon_id"] = gtf["attribute"].map(
        lambda x: _gtf_attr(x, ["exon_id", "exonId"])
    )

    gtf["exon_number"] = gtf["attribute"].map(
        lambda x: _gtf_attr(x, ["exon_number", "exonNumber"])
    )

    gtf["exon_start"] = pd.to_numeric(gtf["exon_start"], errors="coerce")
    gtf["exon_end"] = pd.to_numeric(gtf["exon_end"], errors="coerce")
    gtf["exon_number"] = pd.to_numeric(gtf["exon_number"], errors="coerce")

    gtf = gtf.dropna(
        subset=[
            "CHROM_norm",
            "gene_name_gtf",
            "transcript_id_gtf",
            "exon_start",
            "exon_end",
        ]
    ).copy()

    gtf["exon_start"] = gtf["exon_start"].astype(int)
    gtf["exon_end"] = gtf["exon_end"].astype(int)

    # If exon_number is absent in the GTF, infer transcript-order exon numbers.
    missing_exon_number = gtf["exon_number"].isna()

    if missing_exon_number.any():
        inferred = []

        group_cols = [
            "gene_name_gtf",
            "transcript_id_gtf",
            "CHROM_norm",
            "strand",
        ]

        for _, sub in gtf.groupby(group_cols, dropna=False):
            sub = sub.copy()

            ascending = not (str(sub["strand"].iloc[0]) == "-")

            sub = sub.sort_values(
                ["exon_start", "exon_end"],
                ascending=[ascending, ascending],
            )

            sub["_inferred_exon_number"] = range(1, len(sub) + 1)
            inferred.append(sub)

        gtf = pd.concat(inferred, axis=0).sort_index()

        gtf.loc[gtf["exon_number"].isna(), "exon_number"] = gtf.loc[
            gtf["exon_number"].isna(),
            "_inferred_exon_number",
        ]

        gtf = gtf.drop(columns=["_inferred_exon_number"])

    gtf["exon_number"] = gtf["exon_number"].astype(int)

    gtf_exons_refseq = gtf[
        [
            "gene_name_gtf",
            "CHROM_norm",
            "exon_start",
            "exon_end",
            "exon_number",
            "transcript_id_gtf",
            "transcript_id_gtf_noversion",
            "exon_id",
            "strand",
        ]
    ].copy()

    return gtf_exons_refseq
