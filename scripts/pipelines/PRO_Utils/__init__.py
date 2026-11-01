"""PRO_Utils: protein (antibody-derived tag) analysis for Tapestri samples.

Modules
-------
update_labels           Copy cell-type labels from a DataFrame onto the protein assay.
colors                  Cell-type colour map.
plot_umap               Protein UMAP coloured by cell type.
homogeneity_silhouette  Homogeneity and silhouette scores of protein clusters.
hto_calling             Hashtag (HTO) positive/negative calling.

See README.md for details.
"""

from .update_labels import update_protein_labels_from_dataframe
from .colors import get_celltype_color, CELLTYPE_COLOR_MAP
from .plot_umap import plot_protein_umap
from .homogeneity_silhouette import plot_protein_homogeneity_silhouette
from .hto_calling import (
    call_hto_present_absent,
    call_single_hashtag,
    plot_hashtag_distribution,
    HTOResult,
)

__all__ = [
    "update_protein_labels_from_dataframe",
    "get_celltype_color",
    "CELLTYPE_COLOR_MAP",
    "plot_protein_umap",
    "plot_protein_homogeneity_silhouette",
    "call_hto_present_absent",
    "call_single_hashtag",
    "plot_hashtag_distribution",
    "HTOResult",
]