import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _safe_name(s: str) -> str:
    """Sanitise a label string for use in a filename."""
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s


def plot_draft_protein_umap_per_label(
    Sample,
    celltype_column: str = "label",
    outdir: str = None,
    umap_attribute: str = "umap",
    point_size: int = 5,
    width: int = 400,
    height: int = 400,
    show_plot: bool = False,
    verbose: bool = True,
    add_background: bool = True,
    background_color: str = "#A0A0A0",
    background_opacity: float = 0.5,
    highlight_opacity: float = 0.95,
    highlight_color: str = "#000000",
    output_prefix: str = "Protein_UMAP",
) -> list:
    """
    Create one UMAP plot per unique label, highlighting the relevant cells
    in black against a grey background.

    Parameters
    ----------
    Sample : object
        Sample object with protein attribute containing UMAP coordinates
    celltype_column : str
        Name of the label column in Sample.protein.row_attrs
    outdir : str, optional
        Output directory for saving figures
    umap_attribute : str
        Name of UMAP attribute in Sample.protein
    point_size : int
        Size of scatter points
    width, height : int
        Figure dimensions in pixels
    show_plot : bool
        Whether to call fig.show() for each figure
    verbose : bool
        Print progress messages
    add_background : bool
        Whether to draw all cells as a grey background layer
    background_color : str
        Hex colour for background cells
    background_opacity : float
        Opacity of background cells
    highlight_opacity : float
        Opacity of highlighted cells
    highlight_color : str
        Hex colour for highlighted cells (default: black)
    output_prefix : str
        Prefix for saved PNG filenames

    Returns
    -------
    figs : list[plotly.graph_objects.Figure]
    """
    import sys

    def vprint(msg):
        if verbose:
            print(msg)
            sys.stdout.flush()

    vprint("\n" + "=" * 60)
    vprint("CREATING PROTEIN UMAP PLOTS (ONE PER LABEL)")
    vprint("=" * 60)

    # ---- UMAP coordinates ----
    try:
        umap_coords_df = Sample.protein.get_attribute(umap_attribute, constraint="row")
        umap_coords = (
            umap_coords_df.values
            if isinstance(umap_coords_df, pd.DataFrame)
            else np.array(umap_coords_df)
        )
        if umap_coords.shape[1] < 2:
            raise ValueError(
                f"UMAP coords should have ≥2 columns; got {umap_coords.shape}"
            )
        vprint(f"  ✓ Loaded UMAP coordinates: {umap_coords.shape}")
    except Exception as e:
        vprint(f"  ❌ ERROR: Could not load UMAP coordinates: {e}")
        return []

    # ---- Labels ----
    try:
        celltype_labels = np.array(Sample.protein.row_attrs[celltype_column])
        vprint(f"  ✓ Loaded labels: {len(celltype_labels)} cells")
    except Exception as e:
        vprint(f"  ❌ ERROR: Could not load labels: {e}")
        return []

    unique_celltypes = np.unique(celltype_labels)
    vprint(f"  ✓ Found {len(unique_celltypes)} unique labels")

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    figs = []

    for ct in unique_celltypes:
        mask = celltype_labels == ct
        n_ct = int(mask.sum())
        if n_ct == 0:
            continue

        fig = go.Figure()

        # Background layer
        if add_background:
            fig.add_trace(go.Scatter(
                x=umap_coords[:, 0],
                y=umap_coords[:, 1],
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=background_color,
                    opacity=background_opacity,
                    line=dict(width=0),
                ),
                showlegend=False,
                hoverinfo="skip",
            ))

        # Highlighted cells
        fig.add_trace(go.Scatter(
            x=umap_coords[mask, 0],
            y=umap_coords[mask, 1],
            mode="markers",
            marker=dict(
                size=point_size,
                color=highlight_color,
                opacity=highlight_opacity,
                line=dict(width=0),
            ),
            name=f"{ct} (n={n_ct})",
            showlegend=True,
            hoverinfo="skip",
        ))

        fig.update_layout(
            width=width,
            height=height,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
            margin=dict(l=5, r=5, t=40, b=5),
            title=dict(text=f"{ct} (n={n_ct})", x=0.01, y=0.95, font=dict(size=14)),
        )

        # Save
        if outdir is not None:
            fname = f"{output_prefix}_{_safe_name(ct)}.png"
            outpath = os.path.join(outdir, fname)
            fig.write_image(outpath, width=width, height=height, scale=2)
            fig.write_image(outpath.replace('.png', '.pdf'), width=width, height=height, scale=2)
            vprint(f"  ✓ Saved {ct}: {outpath}")

        if show_plot:
            fig.show()

        figs.append(fig)

    vprint("\n" + "=" * 60)
    vprint("✓ ALL LABEL PLOTS COMPLETE")
    vprint("=" * 60)

    return figs
