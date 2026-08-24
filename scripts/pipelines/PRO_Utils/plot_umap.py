import os
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
) -> go.Figure:
    """
    Create a single protein UMAP plot with cell types coloured by a standard scheme.
    Includes a horizontal legend at the bottom.

    Parameters
    ----------
    Sample : object
        Sample object with protein attribute containing UMAP coordinates
    celltype_column : str
        Name of the cell type column in Sample.protein.row_attrs (default: 'label')
    outdir : str, optional
        Output directory for saving figure
    output_filename : str
        Filename for saved figure (default: 'Protein_UMAP.png')
    umap_attribute : str
        Name of UMAP attribute in Sample.protein
    point_size : int
        Size of points (default: 5)
    width : int
        Figure width in pixels
    height : int
        Figure height in pixels (default: 1000, includes space for legend)
    show_plot : bool
        Whether to display plot in notebook (default: True)
    verbose : bool
        Print progress messages

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    import sys

    def vprint(msg):
        if verbose:
            print(msg)
            sys.stdout.flush()

    vprint("\n" + "=" * 60)
    vprint("CREATING PROTEIN UMAP PLOT")
    vprint("=" * 60)

    # ---- UMAP coordinates ----
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

    # ---- Cell-type labels ----
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

    # ---- Build figure ----
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

    # ---- Save ----
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
