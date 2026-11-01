"""Clone-level plots of germline variant allele frequencies (BAF) and CNV ploidy.

Main entry points
-----------------
visualise_variants_by_clone
    One heatmap per clone (cells x variants, cells hierarchically clustered) of variant allele
    frequencies, with chromosome-arm or cytoband boundaries; saved as PNG + PDF.
plot_integrated_cnv_baf_with_significance
    One figure per clone with stacked BAF, CNV ploidy, and optionally ALT / REF read-depth panels,
    highlighting Wilcoxon-informative and differential (somatic) variants; saved as PNG + PDF.

matplotlib, seaborn and scipy.cluster are imported inside the plotting functions.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd

# BAF colour ramp used for single-cell jitter points: value -> RGB (linear interpolation).
_BAF_POINTS = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
_BAF_COLORS_HEX = ["#FFFFFF", "#dfe7fd", "#5e548e", "#eae4e9", "#FFFFFF"]

# Default face colours for cytoband types (Giemsa stain) in the ploidy-panel background.
_DEFAULT_CYTOBAND_TYPE_TO_COLOR = {
    "gneg": "#ffffff",
    "gpos25": "#d9d9d9",
    "gpos50": "#bdbdbd",
    "gpos75": "#969696",
    "gpos100": "#636363",
    "gvar": "#f0f0f0",
    "stalk": "#f5f5f5",
    "acen": "#f4cccc",
    "NA": "#ffffff",
    "": "#ffffff",
}

# Alternating chromosome-arm background colours.
_ARM_BG_COLORS = ["#eaeaea", "#ffffff"]

# Ploidy values are clipped at this value (shown as "4+").
_PLOIDY_CLIP = 4.0

# Mean depth (ALT + REF) below which a variant's mean markers are drawn faded.
_LOW_DP_THRESHOLD = 5.0

# Half-width of the horizontal jitter of single-cell points.
_JITTER_WIDTH = 0.18


def _make_vprint(verbose):
    """Return a ``print`` + flush function that is silent unless ``verbose``."""
    def vprint(msg):
        if verbose:
            print(msg)
            sys.stdout.flush()

    return vprint


def _subsample_cells(col_vals, max_cells):
    """Randomly keep ``max_cells`` values (global NumPy RNG, no replacement) if there are more."""
    if col_vals.size > max_cells:
        col_vals = col_vals[
            np.random.choice(
                col_vals.size,
                max_cells,
                replace=False,
            )
        ]
    return col_vals


# =============================================================================
# visualise_variants_by_clone helpers
# =============================================================================

def _categorize_af(data):
    """Convert continuous AF (%) to categories: 0 (<=10), 1 (10-90), 2 (>=90)."""
    categorized = np.zeros_like(data)
    categorized[data <= 10] = 0
    categorized[(data > 10) & (data < 90)] = 1
    categorized[data >= 90] = 2
    return categorized


def _build_arm_lookup(cytobands):
    """Map chromosome (str) -> list of ``{'arm', 'start', 'end'}`` spanning each arm's cytobands."""
    chrom_arm_ranges = (
        cytobands
        .groupby(['chromosome', 'arm'])
        .agg({'start': 'min', 'end': 'max'})
        .reset_index()
    )

    arm_lookup = {}
    for _, row in chrom_arm_ranges.iterrows():
        arm_lookup.setdefault(str(row['chromosome']), []).append({
            'arm': row['arm'],
            'start': row['start'],
            'end': row['end']
        })
    return arm_lookup


def _assign_variants_to_arms(variant_cols, arm_lookup, cytobands, split_by_cytobands):
    """
    Map 'chrN:pos:...' variants to 'chrN<arm>' (or 'chrN?') and, if requested, to 'chrN<cytoband>'.

    A variant outside every cytoband keeps its arm label as cytoband label.
    """
    variant_to_arm = {}
    variant_to_cytoband = {}

    for col in variant_cols:
        chr_part = col.split(':')[0]
        pos_part = int(col.split(':')[1])
        chr_short = chr_part.replace('chr', '')

        if chr_short in arm_lookup:
            assigned = False
            for arm_info in arm_lookup[chr_short]:
                if arm_info['start'] <= pos_part <= arm_info['end']:
                    variant_to_arm[col] = f"{chr_part}{arm_info['arm']}"
                    assigned = True
                    break
            if not assigned:
                variant_to_arm[col] = f"{chr_part}?"
        else:
            variant_to_arm[col] = f"{chr_part}?"

        if split_by_cytobands:
            matching_bands = cytobands[
                (cytobands['chromosome'].astype(str) == chr_short) &
                (cytobands['start'] <= pos_part) &
                (cytobands['end'] >= pos_part)
            ]

            if len(matching_bands) > 0:
                cytoband_name = matching_bands.iloc[0]['cytoband']
                variant_to_cytoband[col] = f"{chr_part}{cytoband_name}"
            else:
                variant_to_cytoband[col] = variant_to_arm[col]

    return variant_to_arm, variant_to_cytoband


def _heatmap_sort_key(col, variant_to_arm):
    """Sort key ``(chromosome number, arm, position)``; chrX -> 23, unparsable chromosome -> 0."""
    chr_arm = variant_to_arm.get(col, 'chr0p')
    try:
        chr_num = 23 if 'chrX' in chr_arm else int(chr_arm[3:-1])
    except Exception:
        chr_num = 0
    arm = chr_arm[-1]
    pos = int(col.split(':')[1])
    return (chr_num, arm, pos)


def _save_png_and_pdf(fig, output_path, dpi):
    """Save ``fig`` to ``output_path`` and to the same path with '.png' replaced by '.pdf'."""
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    fig.savefig(output_path.replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight')


def visualise_variants_by_clone(
    df: pd.DataFrame,
    variants_to_plot: list,
    cytobands: pd.DataFrame,
    output_dir: str,
    output_prefix: str = "Single_cell_clones_Variable_AF_Germline_by_chr_arm",
    label_col: str = 'label',
    max_height_per_subplot: float = 8.0,
    max_total_height: float = 200.0,
    max_subplots_per_figure: int = 25,
    dpi: int = 150,
    save_figures: bool = True,
    plot_categorical: bool = False,
    split_by_cytobands: bool = False,
    desired_clone_plotting_order: list = None,
    plot_separate_clones: bool = False,
    font_title: int = 14,
    verbose: bool = True
) -> None:
    """
    Visualize variant allele frequencies by clone with chromosome arm annotations.

    Creates seaborn heatmaps (cells x variants) of allele frequencies, one subplot per clone label.
    Within each clone, cells are ordered by average-linkage hierarchical clustering; variants are
    ordered by chromosome, arm and position, with dashed vertical lines and text labels at each
    chromosome-arm (or cytoband) boundary. Too many clones are split over several figures.

    Parameters
    ----------
    df : pd.DataFrame
        Cells as rows, variants as columns (AF in %, IDs 'chrN:pos:...'), plus a label column.
    variants_to_plot : list
        Variant IDs (columns of ``df``) to include.
    cytobands : pd.DataFrame
        Cytoband reference with columns: chromosome, arm, start, end, cytoband.
    output_dir : str
        Directory to save output figures (created if missing).
    output_prefix : str
        Prefix for output filenames.
    label_col : str
        Name of column containing clone labels.
    max_height_per_subplot : float
        Maximum height (inches) per subplot; also the height of each figure in separate-clone mode.
    max_total_height : float
        Maximum total figure height (inches).
    max_subplots_per_figure : int
        Maximum subplots per figure in combined mode (splits into ``_partN`` figures if exceeded).
    dpi : int
        Resolution for saved figures.
    save_figures : bool
        Only honoured when ``plot_separate_clones=True``; combined figures are always saved.
    plot_categorical : bool
        If True, uses categorical coloring: <=10 = dark blue, 10-90 = white, >=90 = dark red.
        If False, uses continuous RdYlBu_r colormap (0-100).
    split_by_cytobands : bool
        If True, boundaries separate cytobands (more granular than chromosome arms).
        If False, only chromosome arm boundaries are shown.
    desired_clone_plotting_order : list, optional
        Custom order for clone labels. Clones not in the list are appended at the end, sorted.
    plot_separate_clones : bool
        If True, creates one figure per clone (file name suffixed with the clone label).
        If False, creates a single figure with all clones stacked (default behavior).
    font_title : int
        Unused (kept for backwards compatibility).
    verbose : bool
        Print progress messages.

    Returns
    -------
    None
        Saves ``<output_prefix>{_by_chr_arm|_by_cytoband}[_partN|_<clone>].png`` and ``.pdf`` to
        ``output_dir`` and calls ``plt.show()`` for each figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage, leaves_list

    vprint = _make_vprint(verbose)

    vprint("\n" + "="*60)
    vprint("VISUALIZING INFORMATIVE VARIANTS BY CLONE")
    vprint("="*60)

    if plot_separate_clones:
        vprint("MODE: Separate figures per clone (clone name as title)")
    else:
        vprint("MODE: Single figure with all clones stacked")

    if plot_categorical:
        vprint("  Color scheme: CATEGORICAL (≤10=dark blue, 10-90=white, ≥90=dark red)")
    else:
        vprint("  Color scheme: CONTINUOUS (RdYlBu_r)")

    if split_by_cytobands:
        vprint("  Boundaries: CYTOBANDS (individual cytobands separated)")
    else:
        vprint("  Boundaries: CHROMOSOME ARMS (p/q arms separated)")

    os.makedirs(output_dir, exist_ok=True)

    df_selected = df.loc[:, variants_to_plot].copy()
    df_selected[label_col] = df[label_col]

    vprint(f"\n  Data shape: {df_selected.shape[0]} cells × {len(variants_to_plot)} variants")

    # -------------------------------------------------------------------------
    # Variant -> chromosome arm / cytoband assignment and column order
    # -------------------------------------------------------------------------
    vprint("\n  Building chromosome arm lookup...")
    arm_lookup = _build_arm_lookup(cytobands)
    vprint(f"    ✓ Lookup table built for {len(arm_lookup)} chromosomes")

    vprint(f"\n  Assigning variants to chromosome arms{' and cytobands' if split_by_cytobands else ''}...")

    variant_cols = [col for col in df_selected.columns if col != label_col]
    variant_to_arm, variant_to_cytoband = _assign_variants_to_arms(
        variant_cols, arm_lookup, cytobands, split_by_cytobands
    )

    vprint(f"    ✓ Assigned {len(variant_to_arm)} variants to chromosome arms")
    if split_by_cytobands:
        n_cytobands = len(set(variant_to_cytoband.values()))
        vprint(f"    ✓ Assigned variants to {n_cytobands} unique cytobands")

    sorted_cols = sorted(variant_cols, key=lambda col: _heatmap_sort_key(col, variant_to_arm))
    df_selected = df_selected[sorted_cols + [label_col]]

    if split_by_cytobands:
        boundary_list = pd.Index(sorted_cols).map(lambda col: variant_to_cytoband.get(col, ''))
    else:
        boundary_list = pd.Index(sorted_cols).map(lambda col: variant_to_arm.get(col, ''))

    # -------------------------------------------------------------------------
    # Clone order
    # -------------------------------------------------------------------------
    unique_labels_in_data = df_selected[label_col].unique()

    if desired_clone_plotting_order is not None:
        vprint(f"\n  Applying custom clone plotting order...")

        ordered_labels = [label for label in desired_clone_plotting_order
                          if label in unique_labels_in_data]

        remaining_labels = [label for label in unique_labels_in_data
                            if label not in desired_clone_plotting_order]

        unique_labels = ordered_labels + sorted(remaining_labels)

        vprint(f"    Ordered clones: {len(ordered_labels)}")
        if remaining_labels:
            vprint(f"    Additional clones (appended): {len(remaining_labels)}")
            vprint(f"      {', '.join(remaining_labels)}")
    else:
        unique_labels = unique_labels_in_data

    n_labels = len(unique_labels)

    vprint(f"\n  Unique clone labels: {n_labels}")
    vprint(f"  Final plotting order:")
    for idx, label in enumerate(unique_labels, 1):
        n_cells = (df_selected[label_col] == label).sum()
        vprint(f"    {idx}. {label} (n={n_cells})")

    # -------------------------------------------------------------------------
    # Colormap
    # -------------------------------------------------------------------------
    if plot_categorical:
        colors_list = ['#00008B', 'white', '#8B0000']
        cmap = mcolors.ListedColormap(colors_list)
        bounds = [0, 1, 2, 3]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        vmin, vmax = 0, 2
    else:
        cmap = 'RdYlBu_r'
        norm = None
        vmin, vmax = 0, 100

    def plot_single_clone(ax, label, show_ylabel=True):
        """Plot the clustered heatmap of one clone on ``ax``."""
        df_label = df_selected[df_selected[label_col] == label].drop(columns=label_col)

        if len(df_label) == 0:
            vprint(f"    Skipping {label}: no cells")
            ax.text(0.5, 0.5, f'{label}\n(No cells)',
                    ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        try:
            linkage_matrix = linkage(df_label, method='average')
            ordered_indices = leaves_list(linkage_matrix)
            df_label = df_label.iloc[ordered_indices, :]
        except Exception as e:
            vprint(f"    Clustering failed for {label}: {e}")

        if plot_categorical:
            plot_data = _categorize_af(df_label.values)
            plot_data = pd.DataFrame(
                plot_data,
                index=df_label.index,
                columns=df_label.columns
            )
        else:
            plot_data = df_label

        sns.heatmap(plot_data, cbar=True, ax=ax, yticklabels=False,
                    cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)

        # Boundary line (and label, except at x=0) wherever the arm/cytoband changes.
        prev_boundary = None
        for j, current_boundary in enumerate(boundary_list):
            if current_boundary != prev_boundary:
                line_width = 4 if split_by_cytobands else 2
                line_color = 'black' if plot_categorical else 'white'
                ax.axvline(x=j, color=line_color, linestyle='--', linewidth=line_width)

                if j > 0:
                    fontsize = 6 if split_by_cytobands else 8
                    ax.text(j, ax.get_ylim()[1] - 2, current_boundary,
                            ha='center', va='bottom',
                            color='black', fontsize=fontsize, fontweight='bold',
                            rotation=90)
                prev_boundary = current_boundary

        ax.set_yticklabels([])
        ax.tick_params(axis='x', labelsize=8, labelrotation=90)

        if show_ylabel:
            ax.set_ylabel(f'{label} (n={len(df_label)})',
                          fontsize=12, fontweight='bold', rotation=90, labelpad=15)
        else:
            ax.set_ylabel('')

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=9)

        if plot_categorical:
            cbar.set_ticks([0.5, 1.5, 2.5])
            cbar.set_ticklabels(['≤10%', '10-90%', '≥90%'])
            cbar.set_label('AF Category', fontsize=9)
        else:
            cbar.set_label('AF (%)', fontsize=9)

    suffix = "_by_cytoband" if split_by_cytobands else "_by_chr_arm"

    if plot_separate_clones:
        # One single-panel figure per clone.
        vprint("\n  Creating separate figures for each clone...")

        for i, label in enumerate(unique_labels, 1):
            vprint(f"\n  Plotting clone {i}/{n_labels}: {label}")

            fig, ax = plt.subplots(1, 1, figsize=(18, max_height_per_subplot))
            plot_single_clone(ax, label, show_ylabel=False)
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            clone_filename = f"{output_prefix}{suffix}_{label.replace(' ', '_').replace('/', '_')}.png"
            output_path = os.path.join(output_dir, clone_filename)
            if save_figures:
                _save_png_and_pdf(fig, output_path, dpi)
            vprint(f"    ✓ Saved: {clone_filename}")
            plt.show()
            plt.close(fig)

    else:
        # All clones stacked, split over several figures if there are too many.
        height_per_subplot = min(max_height_per_subplot, max_total_height / n_labels)
        total_height = height_per_subplot * n_labels

        def _plot_stacked_figure(labels_subset, fig_height, output_path):
            n_subplots = len(labels_subset)
            fig, axes = plt.subplots(
                n_subplots, 1,
                figsize=(18, fig_height),
                sharex=True
            )

            if n_subplots == 1:
                axes = [axes]

            for ax, label in zip(axes, labels_subset):
                plot_single_clone(ax, label, show_ylabel=True)

            plt.tight_layout()

            _save_png_and_pdf(fig, output_path, dpi)
            vprint(f"    ✓ Saved: {os.path.basename(output_path)}")
            plt.show()
            plt.close(fig)

        if n_labels > max_subplots_per_figure:
            vprint(f"\n  ⚠️  Too many labels ({n_labels}). Splitting into multiple figures...")
            n_figures = int(np.ceil(n_labels / max_subplots_per_figure))

            for fig_idx in range(n_figures):
                start_idx = fig_idx * max_subplots_per_figure
                end_idx = min((fig_idx + 1) * max_subplots_per_figure, n_labels)
                labels_subset = unique_labels[start_idx:end_idx]
                n_subplots = len(labels_subset)

                vprint(f"\n  Creating figure {fig_idx + 1}/{n_figures} with {n_subplots} subplots...")

                _plot_stacked_figure(
                    labels_subset,
                    min(8 * n_subplots, max_total_height),
                    os.path.join(output_dir, f"{output_prefix}{suffix}_part{fig_idx + 1}.png"),
                )

        else:
            vprint(f"\n  Creating single figure with {n_labels} subplots...")

            _plot_stacked_figure(
                unique_labels,
                min(total_height, max_total_height),
                os.path.join(output_dir, f"{output_prefix}{suffix}.png"),
            )

    vprint("\n" + "="*60)
    vprint("VISUALIZATION COMPLETE")
    vprint("="*60)


# =============================================================================
# plot_integrated_cnv_baf_with_significance helpers
# =============================================================================

def _extract_chr_arm_pos(col, variant_to_arm):
    """Sort key ``(chromosome number, arm, position)``; chrX -> 23, unparsable parts -> 0 / '?'."""
    chr_arm = variant_to_arm.get(col, "chr0p")

    try:
        chr_num = 23 if "chrX" in chr_arm else int(chr_arm[3:-1])
    except Exception:
        chr_num = 0

    arm = chr_arm[-1] if isinstance(chr_arm, str) and len(chr_arm) > 0 else "?"

    try:
        pos = int(str(col).split(":")[1])
    except Exception:
        pos = 0

    return chr_num, arm, pos


def _chr_from_arm(chr_arm):
    """'chr7q' -> 'chr7' (input returned as str if it does not match)."""
    match = re.match(r"^(chr(?:\d+|X|Y))", str(chr_arm))
    return match.group(1) if match else str(chr_arm)


def _summarise_values(values, stat="mean"):
    """Mean or median of the non-NaN values (NaN if none)."""
    values = pd.Series(values).dropna().astype(float)

    if len(values) == 0:
        return np.nan

    if stat == "mean":
        return float(values.mean())

    if stat == "median":
        return float(values.median())

    raise ValueError("ploidy_line_stat must be either 'mean' or 'median'.")


def _summarise_interval(values, method="percentile", percentiles=(25, 75)):
    """(low, high) interval of the non-NaN values: percentiles, mean +/- SD, or mean +/- 1.96 SEM."""
    values = pd.Series(values).dropna().astype(float)

    if len(values) == 0:
        return np.nan, np.nan

    if method == "percentile":
        lo, hi = np.percentile(values, percentiles)
        return float(lo), float(hi)

    if method == "sd":
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        return mean - sd, mean + sd

    if method == "sem":
        mean = float(values.mean())
        sem = float(values.sem(ddof=1)) if len(values) > 1 else 0.0
        return mean - 1.96 * sem, mean + 1.96 * sem

    raise ValueError("ploidy_ci_method must be 'percentile', 'sd', or 'sem'.")


def _safe_label_for_filename(label):
    """Replace spaces, '/', '\\' and ':' with '_'."""
    return (
        str(label)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def _baf_to_rgb(vals, baf_colors_rgb):
    """Map BAF values (clipped to [0, 1]) to RGB along the ``_BAF_POINTS`` colour ramp."""
    v = np.clip(np.asarray(vals, dtype=float), 0.0, 1.0)
    r = np.interp(v, _BAF_POINTS, baf_colors_rgb[:, 0])
    g = np.interp(v, _BAF_POINTS, baf_colors_rgb[:, 1])
    b = np.interp(v, _BAF_POINTS, baf_colors_rgb[:, 2])
    return np.stack([r, g, b], axis=-1)


def _cytoband_blocks(cytoband_list, chr_arm_list):
    """Yield ``(start, stop)`` index ranges of consecutive variants sharing cytoband and arm."""
    s = 0

    for i in range(1, len(cytoband_list) + 1):
        is_new_block = (
            i == len(cytoband_list)
            or cytoband_list[i] != cytoband_list[s]
            or chr_arm_list[i] != chr_arm_list[s]
        )

        if is_new_block:
            yield s, i
            s = i


def _arm_blocks(chr_arm_list):
    """``[(arm, first_index, last_index), ...]`` for runs of consecutive equal arms."""
    arm_blocks = []

    if chr_arm_list:
        s = 0

        for j in range(1, len(chr_arm_list) + 1):
            if j == len(chr_arm_list) or chr_arm_list[j] != chr_arm_list[s]:
                arm_blocks.append((chr_arm_list[s], s, j - 1))
                s = j

    return arm_blocks


def _add_top_chr_arm_labels(ax_baf, arm_blocks, rotation, fontsize, pad):
    """Label each chromosome-arm block at its centre on a secondary top x-axis."""
    if not arm_blocks:
        return

    secax = ax_baf.secondary_xaxis("top")
    centers = [(s + e) / 2.0 for _, s, e in arm_blocks]
    labels = [
        arm_name if isinstance(arm_name, str) else ""
        for arm_name, _, _ in arm_blocks
    ]

    secax.set_xticks(centers)
    secax.set_xticklabels(labels, rotation=rotation, fontsize=fontsize)
    secax.tick_params(axis="x", pad=pad)


def _add_bottom_cytoband_labels(
    ax_bottom,
    cytoband_list,
    chr_arm_list,
    fontsize=7,
    rotation=90,
    pad=3,
    min_label_spacing=1.5,
):
    """Label cytoband blocks at their centres on the bottom axis, skipping overlapping labels."""
    if not cytoband_list:
        return

    centers = []
    labels = []

    for s, i in _cytoband_blocks(cytoband_list, chr_arm_list):
        lab = cytoband_list[s]

        if lab not in ("", "NA", None):
            centers.append((s + i - 1) / 2.0)
            labels.append(lab)

    # Hide labels that would overlap: greedily keep a label only if
    # its center is far enough from the last kept label.
    if len(centers) > 1:
        kept_centers = [centers[0]]
        kept_labels = [labels[0]]

        for c, lab in zip(centers[1:], labels[1:]):
            if c - kept_centers[-1] >= min_label_spacing:
                kept_centers.append(c)
                kept_labels.append(lab)

        centers = kept_centers
        labels = kept_labels

    ax_bottom.set_xticks(centers)
    ax_bottom.set_xticklabels(labels, rotation=rotation, fontsize=fontsize)
    ax_bottom.tick_params(
        axis="x",
        bottom=True,
        labelbottom=True,
        top=False,
        labeltop=False,
        pad=pad,
    )


def _style_panel_spines(ax):
    """Show black left/bottom/top spines (width 1.0, above data); hide the right spine."""
    for name, spine in ax.spines.items():
        if name in ("left", "bottom", "top"):
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_edgecolor("black")
            spine.set_zorder(10)
        else:
            spine.set_visible(False)


def _draw_alt_arm_background(ax, chr_arm_list, arm_bg_colors):
    """Shade consecutive chromosome-arm blocks with alternating background colours."""
    current_arm = None
    arm_index = 0
    start_idx = 0

    for j, arm_name in enumerate(chr_arm_list + ["__END__"]):
        if arm_name != current_arm:
            if current_arm is not None:
                ax.axvspan(
                    start_idx - 0.5,
                    j - 0.5,
                    facecolor=arm_bg_colors[arm_index % 2],
                    alpha=0.8,
                    zorder=0,
                )
                arm_index += 1

            current_arm = arm_name
            start_idx = j


def _draw_ci_band_and_caps(
    ax,
    x_start,
    x_end,
    ci_lo,
    ci_hi,
    ci_color,
    ci_alpha,
    plot_caps,
    cap_color,
    cap_linewidth,
    cap_linestyle,
    cap_alpha,
):
    """Draw a CI band between ``ci_lo`` and ``ci_hi`` and optional dashed caps (skipped if NaN)."""
    if pd.isna(ci_lo) or pd.isna(ci_hi):
        return

    ax.fill_between(
        [x_start, x_end],
        [ci_lo, ci_lo],
        [ci_hi, ci_hi],
        color=ci_color,
        alpha=ci_alpha,
        linewidth=0,
        zorder=3.1,
    )

    if plot_caps:
        ax.hlines(
            y=ci_lo,
            xmin=x_start,
            xmax=x_end,
            colors=cap_color,
            linewidth=cap_linewidth,
            linestyles=cap_linestyle,
            alpha=cap_alpha,
            zorder=3.25,
        )
        ax.hlines(
            y=ci_hi,
            xmin=x_start,
            xmax=x_end,
            colors=cap_color,
            linewidth=cap_linewidth,
            linestyles=cap_linestyle,
            alpha=cap_alpha,
            zorder=3.25,
        )


def _build_cytoband_lookups(cytobands):
    """
    Validate ``cytobands`` and build the arm-level lookups.

    Returns ``cb_intervals`` (arm -> sorted ``[(start, end, cytoband, type), ...]``) and
    ``chrom_to_armranges`` (chromosome without 'chr' -> ``[(arm, start, end), ...]``).
    """
    required_cb_cols = {
        "chromosome arm",
        "start",
        "end",
        "cytoband",
        "cytoband type",
    }

    missing = required_cb_cols - set(cytobands.columns)
    if missing:
        raise KeyError(f"cytobands missing required columns: {missing}")

    cb = cytobands.copy()
    cb["chromosome arm"] = cb["chromosome arm"].astype(str)
    cb["start"] = pd.to_numeric(cb["start"], errors="coerce")
    cb["end"] = pd.to_numeric(cb["end"], errors="coerce")
    cb = cb.dropna(subset=["start", "end"])
    cb["start"] = cb["start"].astype(int)
    cb["end"] = cb["end"].astype(int)
    cb["cytoband"] = cb["cytoband"].astype(str)
    cb["cytoband type"] = cb["cytoband type"].astype(str)

    cb_intervals = {}

    for ca, sub in cb.groupby("chromosome arm", sort=False):
        sub = sub.sort_values(["start", "end"])

        cb_intervals[str(ca)] = list(
            zip(
                sub["start"],
                sub["end"],
                sub["cytoband"],
                sub["cytoband type"],
            )
        )

    arm_ranges = (
        cb.groupby("chromosome arm", as_index=False)
        .agg({"start": "min", "end": "max"})
    )

    chrom_to_armranges = {}

    for _, r in arm_ranges.iterrows():
        ca = str(r["chromosome arm"])
        chrom = ca.replace("chr", "")[:-1]

        chrom_to_armranges.setdefault(chrom, []).append(
            (ca, int(r["start"]), int(r["end"]))
        )

    return cb_intervals, chrom_to_armranges


def _variant_to_chr_arm(var, chrom_to_armranges):
    """'chrN:pos...' -> the chromosome arm containing pos, else 'chrN?'."""
    chr_part = str(var).split(":")[0]
    chrom = chr_part.replace("chr", "")

    try:
        pos = int(str(var).split(":")[1])
    except Exception:
        return f"{chr_part}?"

    for ca, s0, e0 in chrom_to_armranges.get(chrom, []):
        if s0 <= pos <= e0:
            return ca

    return f"{chr_part}?"


def _band_and_type(cb_intervals, chr_arm, pos):
    """(cytoband, cytoband type) containing ``pos`` on ``chr_arm``, else ('NA', 'NA')."""
    for s0, e0, b0, t0 in cb_intervals.get(str(chr_arm), []):
        if s0 <= pos <= e0:
            return b0, t0

    return "NA", "NA"


def _build_informative_lookup(wilcoxon_results_var):
    """``{(label, variant): True}`` for rows with informative=True; ``{}`` if anything fails."""
    informative_lookup = {}

    try:
        for arm_name, res_df in wilcoxon_results_var.items():
            if res_df is None or res_df.empty:
                continue

            required_cols = {"label", "variant", "informative"}

            if not required_cols.issubset(res_df.columns):
                continue

            for _, row in res_df.iterrows():
                if bool(row.get("informative", False)):
                    informative_lookup[
                        (str(row["label"]).strip(), str(row["variant"]))
                    ] = True

    except Exception:
        informative_lookup = {}

    return informative_lookup


def _ploidy_cytoband_segments(
    df_cnv_all_num,
    label_extreme_variants,
    cytoband_list,
    chr_arm_list,
    sorted_cols,
    band_to_cnv_cols,
    ploidy_line_stat,
):
    """One segment per displayed cytoband block with the per-amplicon mean ploidies of that band."""
    cytoband_segments = []

    for s, i in _cytoband_blocks(cytoband_list, chr_arm_list):
        chr_arm = chr_arm_list[s]
        cytoband_name = cytoband_list[s]
        chromosome = _chr_from_arm(chr_arm)

        band_key = (chr_arm, cytoband_name)
        band_cnv_cols = band_to_cnv_cols.get(band_key, [])

        amplicon_means = pd.Series(dtype=float)
        n_amplicons = 0

        if band_cnv_cols:
            available = [
                c for c in band_cnv_cols
                if c in df_cnv_all_num.columns
            ]

            if available:
                amplicon_means = (
                    df_cnv_all_num[available]
                    .mean(axis=0)
                    .dropna()
                    .astype(float)
                )
                n_amplicons = len(amplicon_means)

        y = _summarise_values(amplicon_means, stat=ploidy_line_stat)

        band_variants = [str(v) for v in sorted_cols[s:i]]

        n_extreme_variants = sum(
            v in label_extreme_variants
            for v in band_variants
        )

        state = "extreme" if n_extreme_variants > 0 else "neutral"

        cytoband_segments.append(
            {
                "chromosome": chromosome,
                "chr_arm": chr_arm,
                "cytoband": cytoband_name,
                "x_start": s - 0.5,
                "x_end": (i - 1) + 0.5,
                "y": y,
                "state": state,
                "n_amplicons": n_amplicons,
                "n_extreme_variants": n_extreme_variants,
                "amplicon_values": amplicon_means.tolist(),
            }
        )

    return cytoband_segments


def _merge_ploidy_segments(cytoband_segments):
    """
    Merge adjacent segments on the same chromosome with the same extreme/neutral state.

    Segments without amplicon values are dropped and break the run.
    """
    merged_segments = []
    current = None

    for seg in cytoband_segments:
        if len(seg["amplicon_values"]) == 0:
            if current is not None:
                merged_segments.append(current)
                current = None

            continue

        can_merge = (
            current is not None
            and seg["chromosome"] == current["chromosome"]
            and seg["state"] == current["state"]
        )

        if can_merge:
            current["x_end"] = seg["x_end"]
            current["cytobands"].append(seg["cytoband"])
            current["n_amplicons"] += seg["n_amplicons"]
            current["n_extreme_variants"] += seg["n_extreme_variants"]
            current["amplicon_values"].extend(seg["amplicon_values"])

        else:
            if current is not None:
                merged_segments.append(current)

            current = {
                "chromosome": seg["chromosome"],
                "state": seg["state"],
                "x_start": seg["x_start"],
                "x_end": seg["x_end"],
                "cytobands": [seg["cytoband"]],
                "n_amplicons": seg["n_amplicons"],
                "n_extreme_variants": seg["n_extreme_variants"],
                "amplicon_values": list(seg["amplicon_values"]),
            }

    if current is not None:
        merged_segments.append(current)

    return merged_segments


def _plot_log_count_panel(
    ax,
    df_num,
    low_dp_mask,
    x_positions,
    n_cols,
    facecolor,
    color,
    ylabel,
    show_ylabel,
    font_ylabel,
    font_yticks,
    draw_separators,
):
    """Plot per-variant mean log10(count + 1) dots (faded where depth is low) for the ALT/REF panel."""
    _style_panel_spines(ax)
    ax.set_facecolor(facecolor)

    log_vals = (
        np.log10(df_num + 1.0)
        .mean(axis=0, numeric_only=True)
        .values
        .astype(float)
    )

    good = (~low_dp_mask) & (~np.isnan(log_vals))
    low = low_dp_mask & (~np.isnan(log_vals))

    ax.scatter(
        x_positions[good],
        log_vals[good],
        c=color,
        s=20,
        alpha=1.0,
        edgecolors="black",
        linewidths=0.3,
        zorder=2,
    )

    ax.scatter(
        x_positions[low],
        log_vals[low],
        c=color,
        s=20,
        alpha=0.4,
        edgecolors="black",
        linewidths=0.3,
        zorder=2,
    )

    draw_separators(ax)

    ax.set_xlim(-0.5, n_cols - 0.5)

    v_max = np.nanmax(log_vals) if np.any(~np.isnan(log_vals)) else 2.0
    v_ceil = np.ceil(v_max * 10) / 10
    v_ceil = max(v_ceil, 0.5)

    ax.set_ylim(0.0, v_ceil)

    ax.set_yticks(
        np.arange(
            0,
            v_ceil + 0.5,
            max(1.0, np.floor(v_ceil / 3)),
        )
    )

    if show_ylabel:
        ax.set_ylabel(
            ylabel,
            fontsize=font_ylabel,
            fontweight="bold",
            rotation=90,
        )
    else:
        ax.set_ylabel("")

    ax.tick_params(axis="y", labelsize=font_yticks)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)


def plot_integrated_cnv_baf_with_significance(
    df_AF: pd.DataFrame,
    df_cnv_vars: pd.DataFrame,
    df_ALT: pd.DataFrame,
    df_REF: pd.DataFrame,
    cytobands: pd.DataFrame,
    wilcoxon_results_var: dict,
    df_mut_to_amplicon,
    differential_variants: list,
    desired_plotting_order: list,
    output_path: str,
    chr_arms_to_plot: list = None,
    max_jitter_cells: int = 100,
    dpi: int = 300,
    height_baf: float = 2.5,
    height_cnv: float = 2.5,
    height_alt: float = 1.2,
    height_ref: float = 1.2,
    height_spacer: float = 0.04,
    figure_width: float = 18.0,
    height_multiplier: float = 1.4,
    hspace: float = 1.0,
    font_ylabel: int = 11,
    font_genotype: int = 9,
    font_yticks: int = 9,
    font_chr_arms: int = 12,
    plot_separate_clones: bool = False,
    font_title: int = 14,
    plot_ploidy_median: bool = False,
    plot_alt_ref: bool = True,
    show_ylabel: bool = True,
    plot_title: bool = True,
    rotation_chr_labels: int = 90,
    verbose: bool = True,
    top_chr_arm_pad: int = 10,
    cytoband_fontsize: int = 7,
    cytoband_rotation: int = 90,
    cytoband_pad: int = 3,
    chr_arm_linewidth: float = 2.2,
    chr_arm_alpha: float = 0.95,
    cytoband_linewidth: float = 1.0,
    cytoband_alpha: float = 0.9,
    color_cytoband_background_in_ploidy: bool = False,
    cytoband_bg_alpha: float = 0.35,
    cytoband_type_to_color: dict = None,
    df_cnv_all: pd.DataFrame = None,
    show_chr_arm_labels: bool = True,
    ploidy_line_stat: str = "mean",
    merge_contiguous_ploidy_lines: bool = True,
    hide_cnv_points_when_plotting_lines: bool = True,
    plot_ploidy_ci: bool = True,
    ploidy_ci_alpha: float = 0.18,
    ploidy_ci_color: str = "grey",
    ploidy_ci_method: str = "percentile",
    ploidy_ci_percentiles: tuple = (25, 75),
    ploidy_line_color: str = "black",
    ploidy_line_width: float = 2.2,
    plot_ploidy_ci_caps: bool = True,
    ploidy_ci_cap_color: str = "grey",
    ploidy_ci_cap_alpha: float = 0.55,
    ploidy_ci_cap_linewidth: float = 1.0,
    ploidy_ci_cap_linestyle: str = "--",
    cytoband_min_label_spacing: float = 1.5,
) -> None:
    """
    Create integrated BAF, CNV ploidy, ALT, and REF plots per clone.

    For every clone in ``desired_plotting_order`` one figure is written with, from top to bottom:

    - **BAF**: single-cell AF/100 jitter (at most ``max_jitter_cells`` random cells per variant,
      coloured by a BAF ramp) and the clone-mean BAF per variant. Means of Wilcoxon-informative
      variants for the clone are big circles (blue < 0.5, red > 0.5, purple = 0.5), other variants
      small grey circles, ``differential_variants`` yellow diamonds. Markers are faded (alpha 0.4)
      when mean depth (ALT + REF) < 5 or unknown. Dashed lines at 0.25 / 0.75; right axis AA/AB/BB.
    - **Ploidy**: CNV values clipped at 4; single-cell jitter and per-variant medians, and/or
      per-cytoband ploidy lines (see below). Dashed lines at 1 and 3.
    - **ALT / REF** (if ``plot_alt_ref``): mean log10(count + 1) per variant.

    Variants are the ``chr*`` columns of ``df_AF``, ordered by chromosome, arm and position; solid
    lines mark chromosome-arm changes and dashed grey lines cytoband changes. Arm names are shown on
    a top axis and cytoband names on the bottom panel.

    Parameters
    ----------
    df_AF : pandas.DataFrame
        Cells x variants AF (%) plus a ``label`` column.
    df_cnv_vars : pandas.DataFrame
        Cells x variants CNV ploidy plus a ``label`` column.
    df_ALT, df_REF : pandas.DataFrame
        Cells x variants ALT / REF read counts (labels taken from ``df_AF`` by barcode).
    cytobands : pandas.DataFrame
        Columns 'chromosome arm' (e.g. 'chr1p'), 'start', 'end', 'cytoband', 'cytoband type'.
    wilcoxon_results_var : dict
        arm -> DataFrame with 'label', 'variant', 'informative'; source of informative variants.
    df_mut_to_amplicon
        Unused.
    differential_variants : list
        Variants drawn as diamonds in the BAF panel.
    desired_plotting_order : list or None
        Clones to plot, in order (labels absent from ``df_AF`` are skipped); None = all labels.
    output_path : str
        Template path; each clone is saved as '<name>_<clone><ext>' plus a '.pdf' copy (only when
        the extension is '.png' is the PDF a separate file).
    chr_arms_to_plot : list, optional
        Restrict variants (and ploidy-line amplicons) to these chromosome arms.
    max_jitter_cells : int
        Maximum single-cell points per variant (random subsample, global NumPy RNG).
    plot_separate_clones : bool
        Must be True; False raises NotImplementedError (after logging and data preparation).
    plot_ploidy_median : bool
        Draw per-cytoband ploidy lines (and hide single-cell CNV points if
        ``hide_cnv_points_when_plotting_lines``).
    df_cnv_all : pandas.DataFrame, optional
        All-amplicon CNV table (with ``label``) used for ploidy lines; defaults to ``df_cnv_vars``.
    ploidy_line_stat : {'mean', 'median'}
        Statistic of the per-amplicon means used for each line.
    ploidy_ci_method : {'percentile', 'sd', 'sem'}
        CI band: ``ploidy_ci_percentiles`` of amplicon means, mean +/- SD, or mean +/- 1.96 SEM.
    height_spacer, font_title, ...
        Remaining arguments control layout, fonts and colours; ``height_spacer`` is unused.

    Returns
    -------
    None
        Returns early (after printing a message) when no ``chr*`` variants, no variants on the
        selected arms, or no labels to plot are found.

    Raises
    ------
    KeyError
        If ``cytobands`` lacks a required column.
    ValueError
        For an invalid ``ploidy_line_stat`` or ``ploidy_ci_method`` (when ploidy lines are drawn).
    NotImplementedError
        If ``plot_separate_clones`` is False.

    Notes
    -----
    Ploidy-line behaviour (``plot_ploidy_median=True``):

    1. For each displayed cytoband block, all CNV amplicons in that cytoband are identified.
    2. For the current clone, each amplicon gets one mean ploidy value across cells.
    3. Cytobands are "extreme" if ``wilcoxon_results_var`` marks at least one displayed variant
       in that cytoband as informative=True for the clone, "neutral" otherwise.
    4. If ``merge_contiguous_ploidy_lines``, adjacent cytobands are merged when they are on the same
       chromosome and have the same state (bands without amplicons break a run).
    5. The final line for each (merged) block is ``ploidy_line_stat`` of all amplicon means in it,
       always drawn in ``ploidy_line_color``; with ``plot_ploidy_ci`` a grey CI band (and dashed
       caps if ``plot_ploidy_ci_caps``) is drawn behind it.

    Extreme/neutral status is NOT recomputed from ploidy values; it is used for merging only.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    vprint = _make_vprint(verbose)

    baf_colors_rgb = np.array([mcolors.to_rgb(h) for h in _BAF_COLORS_HEX])

    type_to_color = dict(_DEFAULT_CYTOBAND_TYPE_TO_COLOR)

    if isinstance(cytoband_type_to_color, dict):
        type_to_color.update(cytoband_type_to_color)

    vprint("\n" + "=" * 60)
    vprint("CREATING INTEGRATED CNV+BAF PLOT WITH SIGNIFICANCE")
    vprint("=" * 60)

    if plot_ploidy_median and hide_cnv_points_when_plotting_lines:
        vprint(
            "  CNV panel: hiding jitter + amplicon median dots; "
            "showing averaged ploidy lines only"
        )

    if plot_ploidy_median and plot_ploidy_ci:
        vprint(
            f"  CNV panel: plotting grey CI bands using "
            f"{ploidy_ci_method} interval"
        )

    # =========================================================================
    # Variants and their chromosome arms
    # =========================================================================
    all_germline = [
        c for c in df_AF.columns
        if isinstance(c, str) and c.startswith("chr")
    ]

    if len(all_germline) == 0:
        vprint("❌ No variants found.")
        return

    cb_intervals, chrom_to_armranges = _build_cytoband_lookups(cytobands)

    variant_to_arm = {v: _variant_to_chr_arm(v, chrom_to_armranges) for v in all_germline}

    if chr_arms_to_plot is not None:
        all_germline = [
            v for v in all_germline
            if variant_to_arm.get(v, "") in chr_arms_to_plot
        ]

        if len(all_germline) == 0:
            vprint("❌ No variants found for selected chromosome arms.")
            return

    # =========================================================================
    # Select / align matrices
    # =========================================================================
    df_selected = df_AF.loc[:, all_germline].copy()
    df_selected["label"] = df_AF["label"]

    df_cnv_selected = df_cnv_vars.loc[
        :,
        [c for c in all_germline if c in df_cnv_vars.columns],
    ].copy()
    df_cnv_selected["label"] = df_cnv_vars["label"]

    cnv_numeric = (
        df_cnv_selected
        .drop(columns="label", errors="ignore")
        .apply(pd.to_numeric, errors="coerce")
    )
    df_cnv_selected.loc[:, cnv_numeric.columns] = cnv_numeric.clip(upper=_PLOIDY_CLIP)

    df_alt_selected = df_ALT.loc[
        :,
        [c for c in all_germline if c in df_ALT.columns],
    ].copy()

    df_ref_selected = df_REF.loc[
        :,
        [c for c in all_germline if c in df_REF.columns],
    ].copy()

    barcode_to_label = df_AF["label"].to_dict()

    df_alt_selected["label"] = df_alt_selected.index.map(
        lambda x: barcode_to_label.get(x, "Unknown")
    )
    df_ref_selected["label"] = df_ref_selected.index.map(
        lambda x: barcode_to_label.get(x, "Unknown")
    )

    variant_cols = [c for c in df_selected.columns if c != "label"]

    sorted_cols = sorted(
        variant_cols,
        key=lambda x: _extract_chr_arm_pos(x, variant_to_arm),
    )

    df_selected = df_selected[sorted_cols + ["label"]]

    df_cnv_selected = df_cnv_selected[
        [c for c in sorted_cols if c in df_cnv_selected.columns] + ["label"]
    ]

    df_alt_selected = df_alt_selected[
        [c for c in sorted_cols if c in df_alt_selected.columns] + ["label"]
    ]

    df_ref_selected = df_ref_selected[
        [c for c in sorted_cols if c in df_ref_selected.columns] + ["label"]
    ]

    # =========================================================================
    # Arm / cytoband annotation of the displayed variants
    # =========================================================================
    chr_arm_list = [variant_to_arm.get(col, "") for col in sorted_cols]

    arm_blocks = _arm_blocks(chr_arm_list)

    chr_arm_sep_x = [
        i - 0.5
        for i in range(1, len(chr_arm_list))
        if chr_arm_list[i] != chr_arm_list[i - 1]
    ]

    cytoband_list = []
    cytoband_type_list = []

    for v, arm_name in zip(sorted_cols, chr_arm_list):
        try:
            pos = int(str(v).split(":")[1])
        except Exception:
            pos = -1

        b, t = _band_and_type(cb_intervals, arm_name, pos)

        cytoband_list.append(b)
        cytoband_type_list.append(t)

    cytoband_sep_x = [
        i - 0.5
        for i in range(1, len(cytoband_list))
        if chr_arm_list[i] == chr_arm_list[i - 1]
        and cytoband_list[i] != cytoband_list[i - 1]
    ]

    # =========================================================================
    # Precompute all-amplicon CNV columns per cytoband (for ploidy lines)
    # =========================================================================
    _all_cnv_source = df_cnv_all if df_cnv_all is not None else df_cnv_vars

    _all_cnv_cols = [
        c for c in _all_cnv_source.columns
        if isinstance(c, str) and c.startswith("chr") and c != "label"
    ]

    _cnv_col_to_band = {}

    for col in _all_cnv_cols:
        _arm = _variant_to_chr_arm(col, chrom_to_armranges)

        if chr_arms_to_plot is not None and _arm not in chr_arms_to_plot:
            continue

        try:
            _pos = int(str(col).split(":")[1])
        except Exception:
            continue

        _b, _t = _band_and_type(cb_intervals, _arm, _pos)
        _cnv_col_to_band[col] = (_arm, _b)

    _band_to_cnv_cols = {}

    for col, key in _cnv_col_to_band.items():
        _band_to_cnv_cols.setdefault(key, []).append(col)

    vprint(
        f"  Ploidy line source: {len(_all_cnv_cols)} total CNV amplicons "
        f"({len(_cnv_col_to_band)} mapped to plotted cytobands)"
    )

    informative_lookup = _build_informative_lookup(wilcoxon_results_var)

    # =========================================================================
    # Plotting setup
    # =========================================================================
    labels_in_data = set(df_selected["label"].unique())

    if desired_plotting_order is None:
        unique_labels = list(df_selected["label"].unique())
    else:
        unique_labels = [
            lbl for lbl in desired_plotting_order
            if lbl in labels_in_data
        ]

    if len(unique_labels) == 0:
        vprint("❌ No labels found to plot.")
        return

    n_cols = len(sorted_cols)
    x_positions = np.arange(n_cols)
    somatic_set = set(differential_variants)
    somatic_mask = np.array([col in somatic_set for col in sorted_cols], dtype=bool)

    def _draw_separators(ax):
        for x in cytoband_sep_x:
            ax.axvline(
                x=x,
                color="#c0c0c0",
                linestyle="--",
                linewidth=cytoband_linewidth,
                alpha=cytoband_alpha,
                zorder=0.55,
            )
        for x in chr_arm_sep_x:
            ax.axvline(
                x=x,
                color="black",
                linestyle="-",
                linewidth=chr_arm_linewidth,
                alpha=chr_arm_alpha,
                zorder=0.95,
            )

    # =========================================================================
    # Core plotting per clone
    # =========================================================================
    def plot_clone_panels(label, ax_baf, ax_cnv, ax_alt, ax_ref, show_xlabel=True):
        label_clean = str(label).strip()

        # ---------------------------------------------------------------------
        # ALT / REF / DP setup
        # ---------------------------------------------------------------------
        df_alt_label = (
            df_alt_selected[df_alt_selected["label"] == label]
            .drop(columns="label")
        )
        df_ref_label = (
            df_ref_selected[df_ref_selected["label"] == label]
            .drop(columns="label")
        )

        df_alt_num = (
            df_alt_label
            .apply(pd.to_numeric, errors="coerce")
            .reindex(columns=sorted_cols)
        )
        df_ref_num = (
            df_ref_label
            .apply(pd.to_numeric, errors="coerce")
            .reindex(columns=sorted_cols)
        )

        df_dp_num = df_alt_num + df_ref_num

        dp_mean_raw_vals = (
            df_dp_num
            .mean(axis=0, numeric_only=True)
            .values
            .astype(float)
        )

        low_dp_mask = (dp_mean_raw_vals < _LOW_DP_THRESHOLD) | np.isnan(dp_mean_raw_vals)

        # ---------------------------------------------------------------------
        # BAF panel
        # ---------------------------------------------------------------------
        ax = ax_baf
        ax.set_facecolor("white")
        _style_panel_spines(ax)
        _draw_alt_arm_background(ax, chr_arm_list, _ARM_BG_COLORS)

        df_label = (
            df_selected[df_selected["label"] == label]
            .drop(columns="label")
        )

        df_label_num = (
            df_label
            .apply(pd.to_numeric, errors="coerce")
            .reindex(columns=sorted_cols)
        )

        vals = (
            df_label_num
            .mean(axis=0, numeric_only=True)
            .values
            .astype(float)
            / 100.0
        )

        informative_baf_mask = np.zeros(n_cols, dtype=bool)
        mean_colors = np.full(n_cols, "dimgray", dtype=object)

        for i, var in enumerate(sorted_cols):
            if informative_lookup.get((label_clean, str(var)), False):
                informative_baf_mask[i] = True

                v = vals[i]

                if v < 0.5:
                    mean_colors[i] = "#48cae4"
                elif v > 0.5:
                    mean_colors[i] = "#e63946"
                else:
                    mean_colors[i] = "#5e548e"

        for i, is_somatic in enumerate(somatic_mask):
            if is_somatic:
                mean_colors[i] = "#ffc733"

        mean_colors_arr = np.array(mean_colors, dtype=object)

        for j, var in enumerate(sorted_cols):
            col_vals = df_label_num[var].values / 100.0
            col_vals = col_vals[~np.isnan(col_vals)]

            if col_vals.size == 0:
                continue

            col_vals = _subsample_cells(col_vals, max_jitter_cells)

            ax.scatter(
                j + np.random.uniform(
                    -_JITTER_WIDTH,
                    _JITTER_WIDTH,
                    size=col_vals.size,
                ),
                col_vals,
                c=_baf_to_rgb(col_vals, baf_colors_rgb),
                s=8,
                alpha=0.1,
                edgecolors="none",
                zorder=1,
            )

        circle_big = (~somatic_mask) & informative_baf_mask
        circle_small = (~somatic_mask) & (~informative_baf_mask)

        # Mean markers: (variant group, size, marker, edge width, zorder); each drawn
        # first for adequate depth (opaque) then for low depth (faded).
        for group_mask, size, marker, edge_width, zorder in (
            (circle_small, 24, "o", 0.3, 2),
            (circle_big, 48, "o", 0.4, 3),
            (somatic_mask, 60, "D", 0.5, 4),
        ):
            for dp_mask, alpha in ((~low_dp_mask, 1.0), (low_dp_mask, 0.4)):
                sel = group_mask & dp_mask
                ax.scatter(
                    x_positions[sel],
                    vals[sel],
                    c=mean_colors_arr[sel],
                    s=size,
                    alpha=alpha,
                    marker=marker,
                    edgecolors="black",
                    linewidths=edge_width,
                    zorder=zorder,
                )

        for y_ref in (0.25, 0.75):
            ax.axhline(
                y=y_ref,
                color="grey",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                zorder=0.6,
            )

        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_yticklabels(["0", "0.5", "1"])

        if show_ylabel:
            ax.set_ylabel(
                "BAF",
                fontsize=font_ylabel,
                fontweight="bold",
                rotation=90,
                labelpad=10,
            )
        else:
            ax.set_ylabel("")

        ax.tick_params(axis="y", labelsize=font_yticks)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

        ax_right = ax.twinx()
        ax_right.set_ylim(0, 1)
        ax_right.set_yticks([0.0, 0.5, 1.0])
        ax_right.set_yticklabels(["AA", "AB", "BB"])
        ax_right.tick_params(axis="y", labelsize=font_genotype)

        for name, spine in ax_right.spines.items():
            spine.set_visible(name == "right")

            if name == "right":
                spine.set_linewidth(0.8)
                spine.set_edgecolor("black")
                spine.set_zorder(11)

        _draw_separators(ax)

        # ---------------------------------------------------------------------
        # CNV panel
        # ---------------------------------------------------------------------
        ax2 = ax_cnv
        ax2.set_facecolor("white")
        _style_panel_spines(ax2)

        if color_cytoband_background_in_ploidy and cytoband_list:
            for s, i in _cytoband_blocks(cytoband_list, chr_arm_list):
                face = type_to_color.get(
                    str(cytoband_type_list[s]),
                    "#ffffff",
                )

                ax2.axvspan(
                    s - 0.5,
                    (i - 1) + 0.5,
                    facecolor=face,
                    alpha=cytoband_bg_alpha,
                    zorder=0.02,
                )
        else:
            _draw_alt_arm_background(ax2, chr_arm_list, _ARM_BG_COLORS)

        df_cnv_label = (
            df_cnv_selected[df_cnv_selected["label"] == label]
            .drop(columns="label")
        )

        df_cnv_num = (
            df_cnv_label
            .apply(pd.to_numeric, errors="coerce")
            .reindex(columns=sorted_cols)
            .clip(upper=_PLOIDY_CLIP)
        )

        show_cnv_points = not (
            plot_ploidy_median and hide_cnv_points_when_plotting_lines
        )

        if show_cnv_points:
            # Single-cell CNV jitter points.
            for j, var in enumerate(sorted_cols):
                if var not in df_cnv_num.columns:
                    continue

                col_vals = df_cnv_num[var].values
                col_vals = col_vals[~np.isnan(col_vals)]

                if col_vals.size == 0:
                    continue

                col_vals = _subsample_cells(col_vals, max_jitter_cells)

                ax2.scatter(
                    j + np.random.uniform(
                        -_JITTER_WIDTH,
                        _JITTER_WIDTH,
                        size=col_vals.size,
                    ),
                    col_vals,
                    s=8,
                    alpha=0.1,
                    color="#c9ada7",
                    edgecolors="none",
                    zorder=1,
                )

            # Per-variant CNV median dots.
            cnv_variant_medians = (
                df_cnv_num
                .median(axis=0)
                .reindex(sorted_cols)
                .values
                .astype(float)
            )

            good = ~np.isnan(cnv_variant_medians)

            ax2.scatter(
                x_positions[good],
                cnv_variant_medians[good],
                s=24,
                alpha=1.0,
                c="#7f4f24",
                edgecolors="black",
                linewidths=0.3,
                zorder=2,
            )

        # Ploidy summary lines.
        if plot_ploidy_median:
            df_cnv_all_label = (
                _all_cnv_source[_all_cnv_source["label"] == label]
                .drop(columns="label", errors="ignore")
            )

            df_cnv_all_num = (
                df_cnv_all_label
                .apply(pd.to_numeric, errors="coerce")
                .clip(upper=_PLOIDY_CLIP)
            )

            label_extreme_variants = {
                str(var)
                for (clone_label, var), is_info in informative_lookup.items()
                if str(clone_label).strip() == label_clean and bool(is_info)
            }

            cytoband_segments = _ploidy_cytoband_segments(
                df_cnv_all_num,
                label_extreme_variants,
                cytoband_list,
                chr_arm_list,
                sorted_cols,
                _band_to_cnv_cols,
                ploidy_line_stat,
            )

            if merge_contiguous_ploidy_lines:
                line_segments = _merge_ploidy_segments(cytoband_segments)
            else:
                line_segments = [
                    seg for seg in cytoband_segments
                    if len(seg["amplicon_values"]) != 0
                ]

            for seg in line_segments:
                y_final = _summarise_values(
                    seg["amplicon_values"],
                    stat=ploidy_line_stat,
                )

                if plot_ploidy_ci:
                    ci_lo, ci_hi = _summarise_interval(
                        seg["amplicon_values"],
                        method=ploidy_ci_method,
                        percentiles=ploidy_ci_percentiles,
                    )

                    _draw_ci_band_and_caps(
                        ax2,
                        seg["x_start"],
                        seg["x_end"],
                        ci_lo,
                        ci_hi,
                        ci_color=ploidy_ci_color,
                        ci_alpha=ploidy_ci_alpha,
                        plot_caps=plot_ploidy_ci_caps,
                        cap_color=ploidy_ci_cap_color,
                        cap_linewidth=ploidy_ci_cap_linewidth,
                        cap_linestyle=ploidy_ci_cap_linestyle,
                        cap_alpha=ploidy_ci_cap_alpha,
                    )

                ax2.hlines(
                    y=y_final,
                    xmin=seg["x_start"],
                    xmax=seg["x_end"],
                    colors=ploidy_line_color,
                    linewidth=ploidy_line_width,
                    alpha=0.95,
                    zorder=3.4,
                )

        _draw_separators(ax2)

        for y_ref in (1.0, 3.0):
            ax2.axhline(
                y=y_ref,
                color="grey",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                zorder=0.6,
            )

        ax2.set_xlim(-0.5, n_cols - 0.5)
        ax2.set_ylim(0.0, 4.0)
        ax2.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0])
        ax2.set_yticklabels(["0", "1", "2", "3", "4+"])

        if show_ylabel:
            ax2.set_ylabel(
                "Ploidy",
                fontsize=font_ylabel,
                fontweight="bold",
                rotation=90,
            )
        else:
            ax2.set_ylabel("")

        ax2.tick_params(axis="y", labelsize=font_yticks)
        ax2.tick_params(axis="x", bottom=False, labelbottom=False)

        # ---------------------------------------------------------------------
        # ALT / REF panels
        # ---------------------------------------------------------------------
        if plot_alt_ref and ax_alt is not None:
            _plot_log_count_panel(
                ax_alt, df_alt_num, low_dp_mask, x_positions, n_cols,
                facecolor="#f8eedb", color="#8b4513", ylabel="ALT\n(log10)",
                show_ylabel=show_ylabel, font_ylabel=font_ylabel, font_yticks=font_yticks,
                draw_separators=_draw_separators,
            )

        if plot_alt_ref and ax_ref is not None:
            _plot_log_count_panel(
                ax_ref, df_ref_num, low_dp_mask, x_positions, n_cols,
                facecolor="#dbf8ee", color="#459d5b", ylabel="REF\n(log10)",
                show_ylabel=show_ylabel, font_ylabel=font_ylabel, font_yticks=font_yticks,
                draw_separators=_draw_separators,
            )

        # ---------------------------------------------------------------------
        # Bottom cytoband labels
        # ---------------------------------------------------------------------
        bottom_ax = ax_ref if (plot_alt_ref and ax_ref is not None) else ax_cnv

        for _ax in (ax_baf, ax_cnv, ax_alt, ax_ref):
            if _ax is not None:
                _ax.tick_params(axis="x", bottom=False, labelbottom=False)

        if show_xlabel:
            _add_bottom_cytoband_labels(
                bottom_ax,
                cytoband_list=cytoband_list,
                chr_arm_list=chr_arm_list,
                fontsize=cytoband_fontsize,
                rotation=cytoband_rotation,
                pad=cytoband_pad,
                min_label_spacing=cytoband_min_label_spacing,
            )

    # =========================================================================
    # Figure creation
    # =========================================================================
    if not plot_separate_clones:
        raise NotImplementedError(
            "Stacked mode was removed in this patch. "
            "Set plot_separate_clones=True, or restore stacked mode separately."
        )

    out_dir = os.path.dirname(output_path)
    out_base = os.path.basename(output_path)
    out_name, out_ext = os.path.splitext(out_base)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if plot_alt_ref:
        height_ratios = [height_baf, height_cnv, height_alt, height_ref]
        n_panels = 4
    else:
        height_ratios = [height_baf, height_cnv]
        n_panels = 2

    fig_h = height_multiplier * sum(height_ratios)

    for i, label in enumerate(unique_labels):
        vprint(f"\nPlotting clone {i + 1}: {label}")

        fig, axes = plt.subplots(
            n_panels,
            1,
            figsize=(figure_width, fig_h),
            sharex=True,
            gridspec_kw={
                "height_ratios": height_ratios,
                "hspace": hspace,
            },
        )

        fig.patch.set_facecolor("white")

        if plot_alt_ref:
            ax_baf, ax_cnv, ax_alt, ax_ref = axes
        else:
            ax_baf, ax_cnv = axes
            ax_alt, ax_ref = None, None

        plot_clone_panels(
            label,
            ax_baf,
            ax_cnv,
            ax_alt,
            ax_ref,
            show_xlabel=True,
        )

        if show_chr_arm_labels:
            _add_top_chr_arm_labels(
                ax_baf,
                arm_blocks=arm_blocks,
                rotation=rotation_chr_labels,
                fontsize=font_chr_arms,
                pad=top_chr_arm_pad,
            )

        if plot_title:
            fig.suptitle(
                str(label),
                fontsize=font_title,
                fontweight="bold",
                y=0.995,
            )
            plt.tight_layout(pad=0.5, rect=[0, 0, 1, 0.98])
        else:
            plt.tight_layout(pad=0.5)

        safe_label = _safe_label_for_filename(label)
        clone_filename = f"{out_name}_{safe_label}{out_ext}"
        clone_output_path = os.path.join(out_dir, clone_filename)

        _save_png_and_pdf(fig, clone_output_path, dpi)
        vprint(f"  ✓ Saved: {clone_output_path}")

        plt.show()
        plt.close(fig)

    vprint("\n" + "=" * 60)
    vprint("✓ PLOT COMPLETE")
    vprint("=" * 60)
