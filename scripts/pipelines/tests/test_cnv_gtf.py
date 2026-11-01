import pandas as pd
import pytest

from CNV_Utils import explore_gene_cnv as eg

EXPECTED_COLS = [
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


def _write_gtf(path, rows):
    lines = ["#!genome-build test", "#comment line"]
    for chrom, feature, start, end, strand, attr in rows:
        lines.append("\t".join([chrom, "RefSeq", feature, str(start), str(end), ".", strand, ".", attr]))
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_gtf_quoted_attributes_and_chrom_normalisation(tmp_path):
    gtf = _write_gtf(
        tmp_path / "a.gtf",
        [
            ("chr5", "gene", 1, 1000, "+", 'gene_id "TET2";'),
            ("chr5", "exon", 100, 200, "+",
             'gene_id "TET2"; transcript_id "NM_001127208.3"; exon_number "1"; exon_id "e1";'),
            ("CHR7", "exon", 300, 400, "-",
             'gene_name "EZH2"; transcript_id "NM_004456.5"; exon_number "2";'),
            ("chrM", "exon", 5, 10, "+", 'gene_name "MT-ND1"; transcript_id "NC_1"; exon_number "1";'),
            ("X", "exon", 5, 10, "+", 'gene_name "BCOR"; transcript_id "NM_2"; exon_number "3";'),
        ],
    )
    df = eg.load_refseq_gtf_exons(gtf)

    assert list(df.columns) == EXPECTED_COLS
    assert len(df) == 4  # 'gene' feature dropped
    assert list(df["CHROM_norm"]) == ["5", "7", "MT", "X"]
    assert list(df["gene_name_gtf"]) == ["TET2", "EZH2", "MT-ND1", "BCOR"]
    assert list(df["transcript_id_gtf_noversion"]) == ["NM_001127208", "NM_004456", "NC_1", "NM_2"]
    assert list(df["exon_number"]) == [1, 2, 1, 3]
    assert df["exon_number"].dtype.kind == "i"
    assert df["exon_start"].dtype.kind == "i"
    assert df.iloc[0]["exon_id"] == "e1"
    assert pd.isna(df.iloc[1]["exon_id"])


def test_gtf_gff_style_key_value_attributes(tmp_path):
    gtf = _write_gtf(
        tmp_path / "b.gtf",
        [("1", "exon", 10, 20, "+", "gene=NPM1;transcript_id=NM_002520.7;exon_number=4")],
    )
    df = eg.load_refseq_gtf_exons(gtf)
    row = df.iloc[0]
    assert row["gene_name_gtf"] == "NPM1"
    assert row["transcript_id_gtf"] == "NM_002520.7"
    assert row["exon_number"] == 4


@pytest.mark.parametrize(
    "strand, expected",
    [("+", [1, 2, 3]), ("-", [3, 2, 1])],
)
def test_gtf_infers_exon_numbers_by_strand(tmp_path, strand, expected):
    attr = 'gene_name "G"; transcript_id "NM_9.1";'
    gtf = _write_gtf(
        tmp_path / "c.gtf",
        [
            ("chr1", "exon", 100, 150, strand, attr),
            ("chr1", "exon", 300, 350, strand, attr),
            ("chr1", "exon", 500, 550, strand, attr),
        ],
    )
    df = eg.load_refseq_gtf_exons(gtf)
    assert list(df["exon_start"]) == [100, 300, 500]  # original row order kept
    assert list(df["exon_number"]) == expected


def test_gtf_drops_rows_missing_gene_or_transcript(tmp_path):
    gtf = _write_gtf(
        tmp_path / "d.gtf",
        [
            ("chr1", "exon", 1, 2, "+", 'transcript_id "NM_1"; exon_number "1";'),  # no gene
            ("chr1", "exon", 1, 2, "+", 'gene_name "G"; exon_number "1";'),         # no transcript
            ("chr1", "exon", 1, 2, "+", 'gene_name "G"; transcript_id "NM_1"; exon_number "1";'),
        ],
    )
    df = eg.load_refseq_gtf_exons(gtf)
    assert len(df) == 1


def test_gtf_gene_id_fallback(tmp_path):
    gtf = _write_gtf(
        tmp_path / "e.gtf",
        [("chr2", "exon", 1, 2, "+", 'gene_id "DNMT3A"; transcript_id "NM_3"; exon_number "1";')],
    )
    assert eg.load_refseq_gtf_exons(gtf).iloc[0]["gene_name_gtf"] == "DNMT3A"


def test_gtf_no_exons(tmp_path):
    gtf = _write_gtf(tmp_path / "f.gtf", [("chr1", "gene", 1, 2, "+", 'gene_name "G";')])
    df = eg.load_refseq_gtf_exons(gtf)
    assert df.empty
    assert list(df.columns) == EXPECTED_COLS
