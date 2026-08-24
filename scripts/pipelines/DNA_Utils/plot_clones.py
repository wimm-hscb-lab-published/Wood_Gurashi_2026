"""UMAP plots for individual clones and individual somatic variants."""

import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ── Clone colour logic ────────────────────────────────────────────────────────

def _get_clone_color(clone_name: str) -> str:
    """Determine clone colour from the annotation string.

    Priority:
    1. WT → blue (#007da5)
    2. TP53 Hem/Hom or multiple TP53 → red (#a50000)
    3. TP53 Het (single) → yellow (#f3b900)
    4. Non-TP53 Het → green (#00A77F)
    5. Non-TP53 Hom → purple (#4B0DD2)
    6. Default → dark purple (#6f00a5)
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
    """One UMAP per clone, coloured by clone-annotation rules.

    Returns
    -------
    figures : dict[str, go.Figure]
    """
    import sys

    def vprint(msg):
        if verbose:
            print(msg)
            sys.stdout.flush()

    vprint("\n" + "=" * 60)
    vprint("CREATING INDIVIDUAL CLONE UMAP PLOTS")
    vprint("=" * 60)

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

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=umap[~mask, 0], y=umap[~mask, 1], mode="markers",
            marker=dict(size=background_size, color=background_color, line=dict(width=0)),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=umap[mask, 0], y=umap[mask, 1], mode="markers",
            marker=dict(size=clone_size, color=color, line=dict(width=0)),
            hoverinfo="skip", showlegend=False,
        ))
        fig.update_layout(
            width=width, height=height,
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
            margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        )
        figures[clone] = fig

        if outdir:
            safe = str(clone).replace(" ", "_").replace("/", "_").replace("\\", "_")
            _png = os.path.join(outdir, f"{output_prefix}_{safe}.png")
            fig.write_image(_png, width=width, height=height, scale=2)
            fig.write_image(_png.replace('.png', '.pdf'), width=width, height=height, scale=2)

        if show_plots:
            fig.show()

    vprint("\n" + "=" * 60)
    vprint(f"✓ {len(figures)} clone plots created")
    vprint("=" * 60)
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
    """One UMAP per somatic variant (NGT != 0 highlighted).

    Parameters
    ----------
    variant_metadata : pd.DataFrame, optional
        Index = variant ID; must have a ``'Protein'`` column.

    Returns
    -------
    figures : dict[str, go.Figure]
    """
    import sys

    def vprint(msg):
        if verbose:
            print(msg)
            sys.stdout.flush()

    vprint("\n" + "=" * 60)
    vprint("CREATING SOMATIC VARIANT UMAP PLOTS")
    vprint("=" * 60)

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

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=umap[~mask, 0], y=umap[~mask, 1], mode="markers",
            marker=dict(size=background_size, color=background_color, line=dict(width=0)),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=umap[mask, 0], y=umap[mask, 1], mode="markers",
            marker=dict(size=variant_size, color=variant_color, line=dict(width=0)),
            hoverinfo="skip", showlegend=False,
        ))
        fig.update_layout(
            width=width, height=height,
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
            margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        )
        figures[variant] = fig

        if outdir:
            safe = variant.replace(":", "_").replace("/", "_").replace("\\", "_")
            if protein_ann and not pd.isna(protein_ann):
                safe += "_" + str(protein_ann).replace(":", "_").replace("/", "_").replace(" ", "_")
            _png = os.path.join(outdir, f"{output_prefix}_{safe}.png")
            fig.write_image(_png, width=width, height=height, scale=2)
            fig.write_image(_png.replace('.png', '.pdf'), width=width, height=height, scale=2)

        if show_plots:
            fig.show()

    vprint(f"\n✓ {len(figures)} variant plots created")
    return figures
