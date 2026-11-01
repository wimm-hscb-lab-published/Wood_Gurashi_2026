"""Highlight UMAPs: one Plotly figure per clone or per somatic variant.

Main entry points
-----------------
:func:`plot_individual_clone_umaps`
    One UMAP per clone label, the clone coloured by its annotation
    (see :func:`_get_clone_color`) over grey background cells.
:func:`plot_somatic_variants`
    One UMAP per somatic variant, cells with a non-zero NGT highlighted.

Both optionally write PNG + PDF files (via kaleido) and show the figures.
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Horizontal rule used in the verbose banners.
_RULE = "=" * 60


def _make_vprint(verbose: bool):
    """Return a flushing ``print`` that only prints when ``verbose`` is true."""
    def vprint(msg):
        if verbose:
            print(msg)
            sys.stdout.flush()
    return vprint


# ── Clone colour logic ────────────────────────────────────────────────────────

def _get_clone_color(clone_name: str) -> str:
    """Determine clone colour from the annotation string (case-insensitive).

    Priority:
    1. exactly ``WT`` → blue (#007da5)
    2. contains TP53 and a whole ``hem``/``hom`` token, or TP53 twice or more → red (#a50000)
    3. contains TP53 and a whole ``het`` token → yellow (#f3b900)
    4. no TP53, whole ``het`` token → green (#00A77F)
    5. no TP53, whole ``hom`` token → purple (#4B0DD2)
    6. anything else → dark purple (#6f00a5)

    Tokens are delimited by whitespace, ``_`` or the string ends.
    """
    name_l = str(clone_name).lower()

    if name_l == "wt":
        return "#007da5"

    tp53_count = name_l.count("tp53")
    _sep = r"(?:^|[\s_])"
    _end = r"(?:[\s_]|$)"

    if tp53_count > 0:
        if bool(re.search(f"{_sep}(hem|hom){_end}", name_l)) or tp53_count >= 2:
            return "#a50000"
        if bool(re.search(f"{_sep}het{_end}", name_l)):
            return "#f3b900"

    if tp53_count == 0:
        if re.search(f"{_sep}het{_end}", name_l):
            return "#00A77F"
        if re.search(f"{_sep}hom{_end}", name_l):
            return "#4B0DD2"

    return "#6f00a5"


# ── Shared figure helpers ─────────────────────────────────────────────────────

def _highlight_umap_figure(umap, mask, color, size, background_color, background_size,
                           width, height) -> go.Figure:
    """Two-trace UMAP: background cells (``~mask``) in grey, highlighted cells on top."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=umap[~mask, 0], y=umap[~mask, 1], mode="markers",
        marker=dict(size=background_size, color=background_color, line=dict(width=0)),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=umap[mask, 0], y=umap[mask, 1], mode="markers",
        marker=dict(size=size, color=color, line=dict(width=0)),
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        width=width, height=height,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
    )
    return fig


def _write_png_and_pdf(fig, png_path, width, height):
    """Write ``fig`` to ``png_path`` and to the same path with ``.png`` replaced by ``.pdf``."""
    fig.write_image(png_path, width=width, height=height, scale=2)
    fig.write_image(png_path.replace('.png', '.pdf'), width=width, height=height, scale=2)


# ── Per-clone UMAP ────────────────────────────────────────────────────────────

def plot_individual_clone_umaps(
    Sample,
    clone_column: str = "Clone",
    outdir: str = None,
    output_prefix: str = "Clone_UMAP",
    umap_attribute: str = "umap",
    background_color: str = "#D3D3D3",
    clone_size: int = 10,
    background_size: int = 3,
    width: int = 800,
    height: int = 800,
    show_plots: bool = True,
    verbose: bool = True,
) -> dict:
    """Draw one UMAP per clone, coloured by the clone-annotation rules.

    Parameters
    ----------
    Sample : mosaic Sample
        Sample whose dna assay has the ``umap_attribute`` row attribute and
        the ``clone_column`` row attribute.
    clone_column : str
        Row attribute with clone labels (one figure per unique value, sorted).
    outdir : str, optional
        If given, it is created and each figure is written as
        ``<output_prefix>_<clone>.png`` and ``.pdf`` (spaces and slashes in
        the clone name replaced by ``_``).
    output_prefix : str
        File-name prefix.
    umap_attribute : str
        Row attribute holding the 2-D embedding.
    background_color, background_size : str, int
        Style of the non-clone cells.
    clone_size : int
        Marker size of the clone's cells; colour from :func:`_get_clone_color`.
    width, height : int
        Figure size in pixels.
    show_plots : bool
        Call ``fig.show()`` for every figure.
    verbose : bool
        Print progress (flushed).

    Returns
    -------
    figures : dict[str, go.Figure]
        Keyed by clone label; ``{}`` if the UMAP or labels cannot be read
        (the error is printed when ``verbose``).
    """
    vprint = _make_vprint(verbose)

    vprint("\n" + _RULE)
    vprint("CREATING INDIVIDUAL CLONE UMAP PLOTS")
    vprint(_RULE)

    try:
        umap_df = Sample.dna.get_attribute(umap_attribute, constraint="row")
        umap = umap_df.values if isinstance(umap_df, pd.DataFrame) else np.array(umap_df)
        vprint(f"  ✓ UMAP: {umap.shape}")
    except Exception as e:
        vprint(f"  ❌ UMAP error: {e}")
        return {}

    try:
        clone_labels = np.array(Sample.dna.row_attrs[clone_column])
    except Exception as e:
        vprint(f"  ❌ Label error: {e}")
        return {}

    unique_clones = np.unique(clone_labels)
    vprint(f"  Clones: {len(unique_clones)}")

    if outdir:
        os.makedirs(outdir, exist_ok=True)

    figures = {}
    for idx, clone in enumerate(unique_clones, 1):
        mask = clone_labels == clone
        color = _get_clone_color(clone)
        vprint(f"\n  [{idx}/{len(unique_clones)}] {clone}  {color}  n={mask.sum()}")

        fig = _highlight_umap_figure(umap, mask, color, clone_size, background_color,
                                     background_size, width, height)
        figures[clone] = fig

        if outdir:
            safe = str(clone).replace(" ", "_").replace("/", "_").replace("\\", "_")
            _write_png_and_pdf(fig, os.path.join(outdir, f"{output_prefix}_{safe}.png"),
                               width, height)

        if show_plots:
            fig.show()

    vprint("\n" + _RULE)
    vprint(f"✓ {len(figures)} clone plots created")
    vprint(_RULE)
    return figures


# ── Per-variant UMAP ──────────────────────────────────────────────────────────

def plot_somatic_variants(
    Sample,
    differential_variants,
    variant_metadata: pd.DataFrame = None,
    outdir: str = None,
    output_prefix: str = "Variant_UMAP",
    umap_attribute: str = "umap",
    variant_color: str = "#a50000",
    background_color: str = "#D3D3D3",
    variant_size: int = 10,
    background_size: int = 3,
    width: int = 800,
    height: int = 800,
    show_plots: bool = True,
    verbose: bool = True,
) -> dict:
    """Draw one UMAP per somatic variant, highlighting cells with NGT != 0.

    Parameters
    ----------
    Sample : mosaic Sample
        Sample whose dna assay has the ``umap_attribute`` row attribute, an
        ``NGT`` layer and labels.
    differential_variants : array-like
        Variant IDs (flattened, stripped, empty ones dropped); only IDs that
        are NGT columns are plotted, in the given order.
    variant_metadata : pd.DataFrame, optional
        Index = variant ID; an optional ``'Protein'`` column is appended to
        the progress line and file name.
    outdir : str, optional
        If given (and at least one variant is available), it is created and
        each figure is written as ``<output_prefix>_<variant>[_<protein>].png``
        and ``.pdf`` (``:``, ``/``, ``\\`` and, in the protein part, spaces
        replaced by ``_``).
    output_prefix : str
        File-name prefix.
    umap_attribute : str
        Row attribute holding the 2-D embedding.
    variant_color, variant_size : str, int
        Style of the carrier cells (NGT not 0 and not NaN; so NGT 3 =
        missing also counts as highlighted).
    background_color, background_size : str, int
        Style of the other cells.
    width, height : int
        Figure size in pixels.
    show_plots : bool
        Call ``fig.show()`` for every figure.
    verbose : bool
        Print progress (flushed).

    Returns
    -------
    figures : dict[str, go.Figure]
        Keyed by variant ID; ``{}`` if the UMAP/NGT cannot be read or no
        variant is available.
    """
    vprint = _make_vprint(verbose)

    vprint("\n" + _RULE)
    vprint("CREATING SOMATIC VARIANT UMAP PLOTS")
    vprint(_RULE)

    try:
        umap_df = Sample.dna.get_attribute(umap_attribute, constraint="row")
        umap = umap_df.values if isinstance(umap_df, pd.DataFrame) else np.array(umap_df)
    except Exception as e:
        vprint(f"  ❌ UMAP error: {e}")
        return {}

    try:
        df_NGT = Sample.dna.get_attribute("NGT", constraint="row+col")
        cols = list(df_NGT.columns.values)
        df_NGT.loc[:, "label"] = Sample.dna.get_labels()
        df_NGT = df_NGT.loc[:, ["label"] + cols]
    except Exception as e:
        vprint(f"  ❌ NGT error: {e}")
        return {}

    protein_dict = {}
    if variant_metadata is not None and "Protein" in variant_metadata.columns:
        protein_dict = dict(zip(
            variant_metadata.index.values, variant_metadata["Protein"].values
        ))

    variant_list = [str(v).strip() for v in np.ravel(differential_variants) if str(v).strip()]
    available = [v for v in variant_list if v in df_NGT.columns]
    vprint(f"  Variants to plot: {len(available)}")

    if not available:
        return {}
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    figures = {}
    for idx, variant in enumerate(available, 1):
        ngt = df_NGT[variant].values
        mask = (ngt != 0) & (~pd.isna(ngt))
        protein_ann = protein_dict.get(variant)
        vprint(f"  [{idx}/{len(available)}] {variant}  n={mask.sum()}"
               + (f"  ({protein_ann})" if protein_ann and not pd.isna(protein_ann) else ""))

        fig = _highlight_umap_figure(umap, mask, variant_color, variant_size, background_color,
                                     background_size, width, height)
        figures[variant] = fig

        if outdir:
            safe = variant.replace(":", "_").replace("/", "_").replace("\\", "_")
            if protein_ann and not pd.isna(protein_ann):
                safe += "_" + str(protein_ann).replace(":", "_").replace("/", "_").replace(" ", "_")
            _write_png_and_pdf(fig, os.path.join(outdir, f"{output_prefix}_{safe}.png"),
                               width, height)

        if show_plots:
            fig.show()

    vprint(f"\n✓ {len(figures)} variant plots created")
    return figures
