"""Protein UMAP scatter plot coloured by the canonical cell-type palette.

Main entry point
----------------
plot_protein_umap
    One plotly scatter trace per cell type, horizontal legend underneath,
    optional PNG + PDF export.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .colors import get_celltype_color


def plot_protein_umap(
    Sample,
    celltype_column: str = 'label',
    outdir: str = None,
    output_filename: str = "Protein_UMAP.png",
    umap_attribute: str = "umap",
    point_size: int = 5,
    width: int = 800,
    height: int = 1000,
    show_plot: bool = True,
    verbose: bool = True
) -> go.Figure | None:
    """Plot the protein UMAP with cells coloured by cell type.

    Each unique label (in ``np.unique`` order) becomes one ``go.Scatter``
    trace coloured by :func:`~PRO_Utils.colors.get_celltype_color`; axes are
    hidden, the aspect ratio is fixed 1:1 and the legend is horizontal below
    the plot.

    Parameters
    ----------
    Sample : object
        Sample object whose ``protein`` assay holds UMAP coordinates.
    celltype_column : str
        Row attribute of ``Sample.protein`` with cell-type labels
        (default ``'label'``).
    outdir : str, optional
        Directory to save into (created if needed). None saves nothing.
    output_filename : str
        PNG filename (default ``'Protein_UMAP.png'``); a PDF is also written
        with ``'.png'`` replaced by ``'.pdf'``.
    umap_attribute : str
        Name of the UMAP attribute read with
        ``get_attribute(umap_attribute, constraint='row')``.
    point_size : int
        Marker size (default 5).
    width, height : int
        Figure size in pixels (default 800 × 1000; the height includes room
        for the legend). Images are written at ``scale=2``.
    show_plot : bool
        Call ``fig.show()`` (default True).
    verbose : bool
        Print progress and the colour mapping.

    Returns
    -------
    plotly.graph_objects.Figure or None
        The figure, or None if the UMAP coordinates or the labels could not be
        loaded (the error is printed when ``verbose``).
    """
    def vprint(msg):
        if verbose:
            print(msg)
            sys.stdout.flush()

    vprint("\n" + "=" * 60)
    vprint("CREATING PROTEIN UMAP PLOT")
    vprint("=" * 60)

    try:
        umap_coords_df = Sample.protein.get_attribute(umap_attribute, constraint='row')
        umap_coords = (
            umap_coords_df.values
            if isinstance(umap_coords_df, pd.DataFrame)
            else np.array(umap_coords_df)
        )
        vprint(f"  ✓ Loaded UMAP coordinates: {umap_coords.shape}")
    except Exception as e:
        vprint(f"  ❌ ERROR: Could not load UMAP coordinates: {e}")
        return None

    try:
        celltype_labels = np.array(Sample.protein.row_attrs[celltype_column])
        vprint(f"  ✓ Loaded cell type labels: {len(celltype_labels)} cells")
    except Exception as e:
        vprint(f"  ❌ ERROR: Could not load cell type labels: {e}")
        return None

    unique_celltypes = np.unique(celltype_labels)
    vprint(f"\n  Unique cell types: {len(unique_celltypes)}")

    vprint("\n  Color mapping:")
    for ct in unique_celltypes:
        color = get_celltype_color(ct)
        n_cells = int((celltype_labels == ct).sum())
        vprint(f"    {ct:20s} → {color} ({n_cells} cells)")

    fig = go.Figure()

    for celltype in unique_celltypes:
        mask = celltype_labels == celltype
        color = get_celltype_color(celltype)

        fig.add_trace(go.Scatter(
            x=umap_coords[mask, 0],
            y=umap_coords[mask, 1],
            mode='markers',
            marker=dict(size=point_size, color=color, line=dict(width=0)),
            name=celltype,
            hoverinfo='skip',
            showlegend=True,
        ))

    fig.update_layout(
        width=width,
        height=height,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=0, b=200),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
            itemsizing='constant',
            tracegroupgap=10,
        ),
    )

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        output_path = os.path.join(outdir, output_filename)
        fig.write_image(output_path, width=width, height=height, scale=2)
        fig.write_image(output_path.replace('.png', '.pdf'), width=width, height=height, scale=2)
        vprint(f"\n  ✓ Saved: {output_path}")

    if show_plot:
        fig.show()

    vprint("\n" + "=" * 60)
    vprint("✓ PLOT COMPLETE")
    vprint("=" * 60)

    return fig
