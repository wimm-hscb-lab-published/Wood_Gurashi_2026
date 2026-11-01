"""CNV_Utils: copy-number and allelic-imbalance analysis for single-cell DNA.

Modules
-------
variant_grouping  Group informative variants by chromosome arm.
informativeness   Test which variants are informative between clones (AF shifts vs wild type).
cnv_extraction    Per-amplicon ploidy / read depth per clone.
explore_gene_cnv  Inputs for per-gene ploidy region plots; RefSeq GTF exon loading.
visualization     Variant-by-clone and integrated CNV/BAF plots.

See README.md for details.
"""


from .variant_grouping import (
    group_informative_variants_by_chr_arm,
)

from .informativeness import (
    test_variant_informativeness,
    expand_informative_variants,
)

from .cnv_extraction import (
    get_cnv_info,
    extract_cnv_data,
)

from .explore_gene_cnv import (
    prepare_ploidy_region_plot_inputs,
    load_refseq_gtf_exons,
)

from .visualization import (
    visualise_variants_by_clone,
    plot_integrated_cnv_baf_with_significance,
)