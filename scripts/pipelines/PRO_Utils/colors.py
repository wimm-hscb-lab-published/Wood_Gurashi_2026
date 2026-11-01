"""Canonical cell-type colours for protein UMAP visualisations.

Main entry points
-----------------
CELLTYPE_COLOR_MAP
    Cell-type name → hex colour.
get_celltype_color
    Look up a colour with exact, then normalised, matching and a grey fallback.
"""

#: Canonical cell-type → hex colour mapping.
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

#: Colour returned for cell types not found in :data:`CELLTYPE_COLOR_MAP`.
DEFAULT_COLOR = '#808080'


def get_celltype_color(celltype_name: str) -> str:
    """Return the hex colour for a cell-type name.

    Parameters
    ----------
    celltype_name : str
        Cell-type label, e.g. ``'CD14_monocyte'`` or ``'cd14 monocyte'``.

    Returns
    -------
    str
        Hex colour from :data:`CELLTYPE_COLOR_MAP`, or :data:`DEFAULT_COLOR`
        when no key matches.

    Notes
    -----
    An exact key match is tried first. Otherwise both the query and every key
    are lower-cased with spaces and hyphens replaced by underscores, and the
    first key (in mapping order) whose normalised form equals the normalised
    query wins. ``celltype_name`` must support ``.lower()``; non-string labels
    (e.g. numpy integers) raise ``AttributeError`` rather than falling back.
    """
    if celltype_name in CELLTYPE_COLOR_MAP:
        return CELLTYPE_COLOR_MAP[celltype_name]

    normalised = celltype_name.lower().replace(' ', '_').replace('-', '_')
    for key, color in CELLTYPE_COLOR_MAP.items():
        if key.lower().replace(' ', '_').replace('-', '_') == normalised:
            return color

    return DEFAULT_COLOR
