"""Unit tests for DNA_Utils.utils (pure helpers, Varsome cache tools, small assays)."""
import csv
import os
import time

import numpy as np
import pandas as pd
import pytest

from DNA_Utils import utils as U


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_dna(ngt=None, barcodes=None, ids=None, labels=None, extra_col_attrs=None):
    from missionbio.mosaic.dna import Dna

    if ngt is None:
        ngt = np.array([[0, 1, 2], [1, 1, 3], [0, 0, 1], [2, 1, 0]])
    ngt = np.asarray(ngt)
    n, m = ngt.shape
    barcodes = np.array(barcodes if barcodes is not None else [f"BC{i}" for i in range(n)])
    ids = np.array(ids if ids is not None else ["chr1:100:A/T", "chr2:200:C/G", "chrX:5:G/A"][:m])
    labels = np.array(labels if labels is not None else ["x"] * n)
    col_attrs = {"id": ids}
    if extra_col_attrs:
        col_attrs.update(extra_col_attrs)
    return Dna(
        name="dna",
        metadata={"sample_name": np.array(["s1"])},
        layers={"NGT": ngt, "AF": ngt * 50.0, "DP": np.full(ngt.shape, 10)},
        row_attrs={"barcode": barcodes, "sample_name": np.array(["s1"] * n), "label": labels},
        col_attrs=col_attrs,
    )


def varsome_row(genome="hg19", vid="chr1:100:A/T", position="100", **kw):
    row = [""] * U.VARSOME_N_FIELDS
    row[0] = genome
    row[1] = vid
    row[U.VARSOME_POSITION_IDX] = position
    for i, (k, v) in enumerate(kw.items()):
        row[U.VARSOME_DB_COLUMNS.index(k)] = v
    return row


def write_db(path, rows, header=None):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header if header is not None else U.VARSOME_DB_COLUMNS)
        w.writerows(rows)


# ── Chromosome helpers ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "variant, expected",
    [
        ("chr1:100:A/T", 1),
        ("chr17:7577538:C/T", 17),
        ("chr22:1:A/G", 22),
        ("chrX:5:G/A", 23),
        ("chrY:5:G/A", 24),
        ("chrM:5:G/A", 25),
        ("chrMT:5:G/A", 25),
        ("chrUn_gl000220:5:G/A", 999),
        ("1:100:A/T", 999),  # no 'chr' prefix -> slicing yields '' -> 999
        ("", 999),
    ],
)
def test_extract_chr_num(variant, expected):
    assert U.extract_chr_num(variant) == expected


@pytest.mark.parametrize(
    "col, expected",
    [("chr1:100:A/T", "chr1"), ("chrX", "chrX"), ("", ""), ("TP53:chr17:1:A/T", "TP53")],
)
def test_extract_chr(col, expected):
    assert U.extract_chr(col) == expected


@pytest.mark.parametrize(
    "variant, expected",
    [
        ("chr17:7577538:C/T", 7577538),
        ("chr1:0:A/T", 0),
        ("chr1", -1),
        ("chr1:abc:A/T", -1),
        ("chr1:100.0:A/T", -1),
        (None, -1),
        (12345, -1),
    ],
)
def test_extract_position(variant, expected):
    assert U.extract_position(variant) == expected


def test_chrom_key_standard_and_prefix():
    s = pd.Series(["chr1", "2", "chrX", "y", "chrM", "MT", " 22 "])
    assert U.chrom_key(s).tolist() == [1, 2, 23, 24, 25, 25, 22]


def test_chrom_key_nonstandard_pushed_to_end():
    s = pd.Series(["chr1", "chrUn_A", "chrUn_B", "chrUn_A", "chr2"])
    out = U.chrom_key(s).tolist()
    assert out[0] == 1 and out[4] == 2
    assert out[1] == out[3] == 1_000_000
    assert out[2] == 1_000_001
    assert out.index(1_000_000) < out.index(1_000_001)


def test_chrom_key_empty_and_dtype():
    out = U.chrom_key(pd.Series([], dtype=object))
    assert len(out) == 0
    assert U.chrom_key(pd.Series(["chr3"])).dtype.kind == "i"


def test_chrom_key_sorts_genomically():
    s = pd.Series(["chr10", "chr2", "chrX", "chr1"])
    order = s.iloc[np.argsort(U.chrom_key(s).values)].tolist()
    assert order == ["chr1", "chr2", "chr10", "chrX"]


@pytest.mark.xfail(
    strict=True,
    reason="chrom_key strips lowercase 'chr' BEFORE upper-casing, so a mixed-case "
    "'Chr1'/'CHR1' prefix is not removed and is pushed to the non-standard tail "
    "despite the docstring promising mixed-case handling (utils.py:85).",
)
def test_chrom_key_mixed_case_prefix():
    assert U.chrom_key(pd.Series(["Chr1", "CHRX"])).tolist() == [1, 23]


# ── Statistics ───────────────────────────────────────────────────────────────

# ── Plotly heatmap shape ─────────────────────────────────────────────────────

# ── row_attrs sanitizers (fake assay) ────────────────────────────────────────

class _FakeAssay:
    def __init__(self, n, row_attrs):
        self.shape = (n, 2)
        self.row_attrs = row_attrs


def test_sanitize_row_attrs_drops_scalar_and_wrong_length():
    a = _FakeAssay(3, {"ok": np.arange(3), "scalar": np.array(5), "short": np.arange(2),
                       "two_d": np.zeros((3, 2))})
    U.sanitize_row_attrs(a)
    assert set(a.row_attrs) == {"ok", "two_d"}


def test_drop_scalar_row_attrs():
    a = _FakeAssay(3, {"ok": np.arange(3), "scalar": np.array(5), "short": np.arange(2)})
    U.drop_scalar_row_attrs(a)
    assert set(a.row_attrs) == {"ok", "short"}
    U.drop_scalar_row_attrs(None)  # no-op


# ── Varsome: row-level helpers ───────────────────────────────────────────────

def test_varsome_constants():
    assert U.VARSOME_N_FIELDS == 18
    assert U.VARSOME_DB_COLUMNS[U.VARSOME_POSITION_IDX] == "Position"


@pytest.mark.parametrize(
    "row, expected",
    [
        (varsome_row(), "ok"),
        (varsome_row(genome=" hg38 "), "ok"),
        (varsome_row(vid="chr13:28602226:*/A+AAG", position=""), "ok"),
        (list(U.VARSOME_DB_COLUMNS), "repeat_header"),
        (varsome_row(genome="foo"), "bad_genome('foo')"),
        (varsome_row(vid="chr1100A/T"), "bad_variant_id('chr1100A/T')"),
        (varsome_row(vid="chr1:100:AT"), "bad_variant_id('chr1:100:AT')"),
        (["hg19", "chr1:1:A/T"], "field_count(2)"),
        (varsome_row() + ["extra"], "field_count(19)"),
        ([], "field_count(0)"),
    ],
)
def test_varsome_classify_row(row, expected):
    assert U._varsome_classify_row(row) == expected


def test_varsome_classify_row_truncates_long_genome():
    r = varsome_row(genome="x" * 50)
    assert U._varsome_classify_row(r) == f"bad_genome({'x' * 20!r})"


def test_varsome_n_populated():
    assert U._varsome_n_populated(["a", "", " ", "b", None]) == 3  # None -> 'None'
    assert U._varsome_n_populated([]) == 0


@pytest.mark.parametrize(
    "raw, changed, after",
    [
        ("148386525.0", True, "148386525"),
        ("101917521", False, "101917521"),
        (" 100 ", False, " 100 "),  # stripped form already canonical -> untouched
        ("", False, ""),
        ("nan", True, ""),
        ("None", True, ""),
        ("  ", True, ""),
        ("1e3", True, "1000"),
        ("12.5", False, "12.5"),
        ("abc", False, "abc"),
    ],
)
def test_varsome_normalize_position(raw, changed, after):
    row = varsome_row(position=raw)
    assert U._varsome_normalize_position(row) is changed
    assert row[U.VARSOME_POSITION_IDX] == after


@pytest.mark.parametrize(
    "exc, expected",
    [
        (ValueError("invalid literal for int() with base 10: '148386525.0'"), True),
        (ValueError("invalid literal for int() with base 10: 'abc'"), False),
        (ValueError("something else."), False),
        (TypeError("invalid literal for int() with base 10: '1.0'"), False),
    ],
)
def test_is_varsome_cache_dtype_error(exc, expected):
    assert U._is_varsome_cache_dtype_error(exc) is expected


# ── Varsome: lock ────────────────────────────────────────────────────────────

def test_varsome_db_lock_yields_path_and_creates_lock(tmp_path):
    db = tmp_path / "sub" / "varsome.csv"
    with U.varsome_db_lock(db) as got:
        assert got == db
        assert (tmp_path / "sub" / "varsome.csv.lock").exists()


def test_varsome_db_lock_reentrant_after_release(tmp_path):
    db = tmp_path / "v.csv"
    with U.varsome_db_lock(db):
        pass
    with U.varsome_db_lock(db, timeout=0.1):
        pass


def test_varsome_db_lock_times_out_when_held(tmp_path):
    db = tmp_path / "v.csv"
    with U.varsome_db_lock(db):
        # flock locks are per open file description, so a second open in the
        # same process contends just like another process would.
        t0 = time.monotonic()
        with pytest.raises(TimeoutError):
            with U.varsome_db_lock(db, timeout=0.2, poll=0.05):
                pass
        assert time.monotonic() - t0 < 5


def test_varsome_db_lock_released_on_exception(tmp_path):
    db = tmp_path / "v.csv"
    with pytest.raises(RuntimeError):
        with U.varsome_db_lock(db):
            raise RuntimeError("boom")
    with U.varsome_db_lock(db, timeout=0.1):
        pass


def test_varsome_db_lock_default_path_uses_varsome_db_path(tmp_path, monkeypatch):
    target = tmp_path / "cache.csv"
    monkeypatch.setattr(U, "varsome_db_path", lambda: target)
    with U.varsome_db_lock() as got:
        assert got == target


# ── Varsome: inspect / repair / reset (all on tmp_path) ──────────────────────

def test_repair_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        U.repair_varsome_db(tmp_path / "missing.csv", verbose=False)


def test_repair_dry_run_does_not_write(tmp_path):
    db = tmp_path / "v.csv"
    write_db(db, [varsome_row(position="1.0"), ["bad"]])
    before = db.read_bytes()
    r = U.repair_varsome_db(db, dry_run=True, verbose=False)
    assert db.read_bytes() == before
    assert r["n_before"] == 2 and r["n_kept"] == 1 and r["n_dropped"] == 1
    assert r["n_positions_fixed"] == 1
    assert r["backup"] is None


def test_repair_clean_not_rewritten(tmp_path):
    db = tmp_path / "v.csv"
    write_db(db, [varsome_row()])
    mtime = os.stat(db).st_mtime_ns
    r = U.repair_varsome_db(db, verbose=False)
    assert r["n_dropped"] == 0 and r["backup"] is None
    assert os.stat(db).st_mtime_ns == mtime
    assert list(tmp_path.glob("*.bak")) == []


def test_repair_full(tmp_path):
    db = tmp_path / "v.csv"
    sparse = varsome_row(vid="chr2:5:C/G", position="5")
    rich = varsome_row(vid="chr2:5:C/G", position="5.0", Gene="TP53", Function="coding, splicing")
    write_db(db, [
        varsome_row(vid="chr1:1:A/T", position="1.0"),
        sparse,
        rich,
        ["torn", "line"],
        varsome_row(genome="junk"),
    ])
    r = U.repair_varsome_db(db, verbose=False)
    assert r["n_before"] == 5
    assert r["n_dropped"] == 2
    assert r["n_deduped"] == 1
    assert r["n_kept"] == 2
    assert r["n_positions_fixed"] == 2
    assert r["backup"] is not None and os.path.exists(r["backup"])
    assert not (tmp_path / "v.csv.tmp").exists()

    df = pd.read_csv(db, index_col=[0, 1])
    assert df.shape == (2, U.VARSOME_N_FIELDS - 2)
    assert df.loc[("hg19", "chr2:5:C/G"), "Gene"] == "TP53"  # richer row kept
    assert df.loc[("hg19", "chr2:5:C/G"), "Function"] == "coding, splicing"
    assert df["Position"].dtype != object


def test_repair_no_backup_no_dedupe(tmp_path):
    db = tmp_path / "v.csv"
    write_db(db, [varsome_row(), varsome_row(), ["x"]])
    r = U.repair_varsome_db(db, backup=False, dedupe=False, verbose=False)
    assert r["n_kept"] == 2 and r["n_deduped"] == 0 and r["backup"] is None
    assert list(tmp_path.glob("*.bak")) == []
    assert len(pd.read_csv(db)) == 2


# ── Small real mosaic Dna assay ──────────────────────────────────────────────

def test_subset_assay_to_cells_preserve_order():
    dna = make_dna()
    out = U.subset_assay_to_cells(dna, ["BC2", "BC0", "BC2", "missing"], verbose=False)
    assert list(out.barcodes()) == ["BC0", "BC2"]
    assert out.shape == (2, 3)


@pytest.mark.xfail(
    strict=True,
    reason="preserve_assay_order=False is documented to 'use the order given', but "
    "mosaic's barcode slicing (utils.py:892) re-sorts rows to assay order, so the "
    "requested order is silently lost.",
)
def test_subset_assay_to_cells_given_order():
    dna = make_dna()
    out = U.subset_assay_to_cells(dna, ["BC3", "BC1"], preserve_assay_order=False, verbose=False)
    assert list(out.barcodes()) == ["BC3", "BC1"]


def test_subset_assay_to_cells_index_input():
    dna = make_dna()
    df = pd.DataFrame(index=["BC1"])
    assert list(U.subset_assay_to_cells(dna, df, verbose=False).barcodes()) == ["BC1"]


def test_subset_assay_to_cells_no_overlap_raises():
    with pytest.raises(ValueError):
        U.subset_assay_to_cells(make_dna(), ["zzz"], verbose=False)


def test_variant_metadata_frame_stats():
    dna = make_dna(extra_col_attrs={"amplicon": np.array(["a1", "a2", "a3"])})
    meta = U.variant_metadata_frame(dna, verbose=False, include_af_dp_stats=True)
    assert meta.index.name == "id"
    assert list(meta.index) == ["chr1:100:A/T", "chr2:200:C/G", "chrX:5:G/A"]
    assert meta["amplicon"].tolist() == ["a1", "a2", "a3"]
    assert meta["chrom"].tolist() == ["chr1", "chr2", "chrX"]
    assert meta["position"].tolist() == [100, 200, 5]
    # NGT column 3: [2,3,1,0] -> WT1 HET1 HOM1 missing1
    r = meta.loc["chrX:5:G/A"]
    assert (r.n_WT, r.n_HET, r.n_HOM, r.n_missing) == (1, 1, 1, 1)
    assert r.genotype_rate == pytest.approx(0.75)
    assert r.mutated_cell_frac == pytest.approx(2 / 3)
    assert r.vaf_by_cell_count == pytest.approx(100 * 3 / 6)
    # mean_AF over genotyped (non-3) cells: AF = NGT*50 -> (100+50+0)/3
    assert r.mean_AF_genotyped == pytest.approx(50.0)
    assert r.mean_DP_genotyped == pytest.approx(10.0)


def test_variant_metadata_frame_annotations_and_sort():
    dna = make_dna(ids=["chrX:5:G/A", "chr2:200:C/G", "chr2:100:C/G"],
                   extra_col_attrs={"Gene": np.array(["g1", "g2", "g3"])})
    ann = pd.DataFrame({"Gene": ["A", "B"], "Impact": ["hi", "lo"]},
                       index=["chr2:200:C/G", "chrX:5:G/A"])
    meta = U.variant_metadata_frame(dna, annotations=ann, include_subset_stats=False,
                                    sort_genomic=True, verbose=False)
    assert list(meta.index) == ["chr2:100:C/G", "chr2:200:C/G", "chrX:5:G/A"]
    assert "Gene_annot" in meta.columns and "Gene" in meta.columns
    assert meta.loc["chr2:200:C/G", "Impact"] == "hi"
    assert pd.isna(meta.loc["chr2:100:C/G", "Impact"])
    assert "n_WT" not in meta.columns
