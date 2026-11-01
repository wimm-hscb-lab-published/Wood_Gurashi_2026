import pandas as pd
import pytest

from CNV_Utils import variant_grouping as vg


@pytest.fixture
def cytobands():
    # Two bands per arm so the min/max aggregation is exercised.
    return pd.DataFrame(
        {
            "chromosome": ["1", "1", "1", "1", "5", "5", "X"],
            "arm": ["p", "p", "q", "q", "p", "q", "p"],
            "start": [0, 50, 101, 150, 0, 48_000_001, 0],
            "end": [49, 100, 149, 200, 48_000_000, 181_000_000, 60_000_000],
            "cytoband": ["p36", "p11", "q11", "q44", "p15", "q35", "p22"],
        }
    )


def test_groups_by_chromosome_and_arm(cytobands):
    variants = [
        "chr1:10:A/T",
        "chr1:100:C/G",   # boundary: end of p arm (inclusive)
        "chr1:101:G/A",   # boundary: start of q arm (inclusive)
        "chr1:200:T/C",
        "chr5:1000:A/C",
        "chr5:150000000:A/C",
        "chrX:5:A/G",
    ]
    by_chr, by_arm = vg.group_informative_variants_by_chr_arm(variants, cytobands, verbose=False)

    assert dict(by_chr) == {
        "chr1": variants[:4],
        "chr5": variants[4:6],
        "chrX": [variants[6]],
    }
    assert by_arm == {
        "chr1p": ["chr1:10:A/T", "chr1:100:C/G"],
        "chr1q": ["chr1:101:G/A", "chr1:200:T/C"],
        "chr5p": ["chr5:1000:A/C"],
        "chr5q": ["chr5:150000000:A/C"],
        "chrXp": ["chrX:5:A/G"],
    }


def test_unmapped_variants_are_dropped(cytobands):
    variants = [
        "chr2:10:A/T",       # no cytoband data for chr2
        "chr1:999999:A/T",   # beyond every band on chr1
        "chr1:notanint:A/T", # unparsable position
        "chr1:20:A/T",
    ]
    by_chr, by_arm = vg.group_informative_variants_by_chr_arm(variants, cytobands, verbose=False)
    assert set(by_chr) == {"chr1", "chr2"}
    assert by_arm == {"chr1p": ["chr1:20:A/T"]}


def test_empty_input(cytobands):
    by_chr, by_arm = vg.group_informative_variants_by_chr_arm([], cytobands, verbose=False)
    assert dict(by_chr) == {}
    assert by_arm == {}


def test_verbose_prints(cytobands, capsys):
    vg.group_informative_variants_by_chr_arm(["chr1:10:A/T"], cytobands, verbose=True)
    out = capsys.readouterr().out
    assert "GROUPING COMPLETE" in out


def test_quiet_is_silent(cytobands, capsys):
    vg.group_informative_variants_by_chr_arm(["chr1:10:A/T"], cytobands, verbose=False)
    assert capsys.readouterr().out == ""


@pytest.mark.xfail(
    strict=True,
    reason=(
        "variant_grouping.py builds arm_lookup keyed on the raw cytobands['chromosome'] "
        "values but looks it up with the string chr_short ('1'); if the cytoband "
        "table has an integer chromosome column (pd.read_csv on autosome-only bands) "
        "every variant is silently unmapped. visualization.py uses str(row['chromosome'])."
    ),
)
def test_integer_chromosome_column_is_mapped():
    cyto = pd.DataFrame(
        {"chromosome": [1, 1], "arm": ["p", "q"], "start": [0, 101], "end": [100, 200]}
    )
    _, by_arm = vg.group_informative_variants_by_chr_arm(["chr1:10:A/T"], cyto, verbose=False)
    assert by_arm == {"chr1p": ["chr1:10:A/T"]}
