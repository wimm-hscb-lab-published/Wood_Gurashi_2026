"""Shared colour maps and helpers for protein UMAP visualisations."""


# Canonical cell-type → hex colour mapping
CELLTYPE_COLOR_MAP = {
    'Erythroid':      '#E15759',
    'Pro-Myeloid':    '#F28E2D',
    'HSC_MPP':        '#063E8A',
    'Progenitor':     '#8c564b',
    'CD14_monocyte':  '#F1CE63',
    'B':              '#227C9D',
    'NK':             '#3A5A41',
    'Mast':           '#b5817d',
    'CD16_monocyte':  '#B6992D',
    'Other_T':        '#84BCB5',
    'CD8_T':          '#E9AFA3',
    'CD4_T':          '#231843',
    'Plasma':         '#22D19D',
    'pDC':            '#8CA141',
    'cDC':            '#808080',
}

DEFAULT_COLOR = '#808080'


def get_celltype_color(celltype_name: str) -> str:
    """Return the hex colour for *celltype_name*, falling back to grey."""

    # Exact match
    if celltype_name in CELLTYPE_COLOR_MAP:
        return CELLTYPE_COLOR_MAP[celltype_name]

    # Case-insensitive / normalised match
    normalised = celltype_name.lower().replace(' ', '_').replace('-', '_')
    for key, color in CELLTYPE_COLOR_MAP.items():
        if key.lower().replace(' ', '_').replace('-', '_') == normalised:
            return color

    return DEFAULT_COLOR
