import os
import sys
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def plot_subclones(
    df_loh_cytobands: pd.DataFrame,
    df: pd.DataFrame,
    chr_arm: str = 'chr5q',
    cluster: int = 0,
    cytobands: pd.DataFrame = None,
    variant_to_arm: dict = None,
    output_path: str = None,
    figsize: tuple = (20, 12),
    verbose: bool = True
):
    """
    Plot LOH subclones for a chromosome arm of interest showing their distinct deletion patterns.
    
    Parameters
    ----------
    df_loh_cytobands : pd.DataFrame
        Output from map_loh_to_cytobands
    df : pd.DataFrame
        Original allele frequency dataframe with cells as rows
    chr_arm : str
        Chromosome arm to plot (e.g., 'chr5q', 'chr7q')
    cluster : int
        Cluster ID to analyze
    cytobands : pd.DataFrame
        Cytoband reference data
    variant_to_arm : dict
        Mapping of variants to chromosome arms
    output_path : str
        Path to save figure
    figsize : tuple
        Figure size
    verbose : bool
        Print progress
    """
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from scipy.cluster.hierarchy import linkage, leaves_list
    
    def vprint(msg):
        if verbose:
            print(msg)
            import sys
            sys.stdout.flush()
    
    vprint("\n" + "="*60)
    vprint(f"PLOTTING {chr_arm} SUBCLONES")
    vprint("="*60)
    
    # Filter to specific cluster
    cluster_data = df_loh_cytobands[
        (df_loh_cytobands['chr_arm'] == chr_arm) & 
        (df_loh_cytobands['cluster'] == cluster)
    ].copy()
    
    if len(cluster_data) == 0:
        vprint(f"⚠️  No data found for {chr_arm} cluster {cluster}")
        return
    
    # Identify subclones by cytoband range
    range_groups = cluster_data.groupby('cytoband_range').agg({
        'cell_id': lambda x: list(x),
        'length_mb': 'median',
        'cytoband_start': 'first',
        'cytoband_end': 'first'
    }).reset_index()
    
    range_groups['n_cells'] = range_groups['cell_id'].apply(len)
    range_groups = range_groups.sort_values('n_cells', ascending=False).reset_index(drop=True)
    range_groups['subclone'] = [f"Subclone {i+1}" for i in range(len(range_groups))]
    
    vprint(f"\n  Found {len(range_groups)} subclones:")
    for idx, row in range_groups.iterrows():
        vprint(f"    {row['subclone']}: {row['cytoband_range']} ({row['n_cells']} cells, {row['length_mb']:.1f} Mb)")
    
    # Get variants for target chromosome arm from df
    if variant_to_arm is None:
        vprint("⚠️  No variant_to_arm provided, cannot filter variants")
        return
    
    chr_variants = [v for v, arm in variant_to_arm.items() if arm == chr_arm]
    
    if len(chr_variants) == 0:
        vprint(f"⚠️  No variants found for {chr_arm}")
        return
    
    vprint(f"\n  Found {len(chr_variants)} variants on {chr_arm}")
    
    # Extract position from variant names for sorting
    def extract_pos(variant):
        try:
            return int(variant.split(':')[1].split('-')[0])
        except:
            return 0
    
    chr_variants_sorted = sorted(chr_variants, key=extract_pos)
    
    # Get all cells from cluster
    all_cells = cluster_data['cell_id'].unique()
    
    # Filter df to these cells and chromosome arm variants
    df_chr = df.loc[all_cells, chr_variants_sorted].copy()
    
    vprint(f"  Data shape: {df_chr.shape[0]} cells × {df_chr.shape[1]} variants")
    
    # Assign subclone labels to cells
    cell_to_subclone = {}
    for idx, row in range_groups.iterrows():
        for cell in row['cell_id']:
            cell_to_subclone[cell] = row['subclone']
    
    df_chr['subclone'] = df_chr.index.map(cell_to_subclone)
    
    # Create figure with multiple panels
    n_subclones = len(range_groups)
    fig = plt.figure(figsize=figsize)
    
    # Grid: subclone heatmaps on left, cytoband diagram on right
    gs = fig.add_gridspec(n_subclones + 1, 2, width_ratios=[4, 1], 
                         height_ratios=[1] * n_subclones + [0.3],
                         hspace=0.3, wspace=0.1)
    
    # Color palette for subclones
    colors = plt.cm.Set3(np.linspace(0, 1, n_subclones))
    
    # Plot each subclone
    for idx, (_, row) in enumerate(range_groups.iterrows()):
        subclone_name = row['subclone']
        cells = row['cell_id']
        
        # Get data for this subclone
        df_subclone = df_chr[df_chr.index.isin(cells)].drop(columns='subclone')
        
        # Cluster cells within subclone
        try:
            if len(df_subclone) > 1:
                linkage_matrix = linkage(df_subclone, method='average')
                ordered_indices = leaves_list(linkage_matrix)
                df_subclone = df_subclone.iloc[ordered_indices, :]
        except Exception as e:
            vprint(f"  Clustering failed for {subclone_name}: {e}")
        
        # Heatmap
        ax_heat = fig.add_subplot(gs[idx, 0])
        sns.heatmap(df_subclone, cmap='RdYlBu_r', center=50, 
                   vmin=0, vmax=100, cbar=True, ax=ax_heat, 
                   yticklabels=False, xticklabels=False)
        
        # Add title with subclone info
        ax_heat.set_ylabel(f'{subclone_name}\n{row["cytoband_range"]}\n({row["n_cells"]} cells)', 
                          fontsize=10, fontweight='bold', rotation=0, 
                          ha='right', va='center', labelpad=60)
        ax_heat.set_xlabel('')
        
        # Adjust colorbar
        cbar = ax_heat.collections[0].colorbar
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('AF (%)', fontsize=8)
        
        # Cytoband diagram
        ax_cyto = fig.add_subplot(gs[idx, 1])
        
        if cytobands is not None:
            # Get cytobands for target chromosome arm
            chr_cytobands = cytobands[cytobands['chromosome arm'] == chr_arm].copy()
            chr_cytobands = chr_cytobands.sort_values('start')
            
            # Determine which cytobands are deleted
            deleted_cytobands = row['cytoband_range'].split('-')
            if len(deleted_cytobands) == 2:
                start_band, end_band = deleted_cytobands
            else:
                start_band = end_band = deleted_cytobands[0]
            
            # Find indices
            start_idx = chr_cytobands[chr_cytobands['cytoband'] == start_band].index
            end_idx = chr_cytobands[chr_cytobands['cytoband'] == end_band].index
            
            if len(start_idx) > 0 and len(end_idx) > 0:
                start_idx = start_idx[0]
                end_idx = end_idx[0]
                
                # Plot chromosome arm
                for i, cb_row in chr_cytobands.iterrows():
                    y_pos = i
                    is_deleted = start_idx <= i <= end_idx
                    
                    color = colors[idx] if is_deleted else 'lightgray'
                    alpha = 0.8 if is_deleted else 0.3
                    
                    ax_cyto.barh(y_pos, 1, height=0.8, color=color, alpha=alpha, 
                               edgecolor='black', linewidth=0.5)
                    
                    # Add cytoband label
                    ax_cyto.text(0.5, y_pos, cb_row['cytoband'], 
                               ha='center', va='center', fontsize=6,
                               fontweight='bold' if is_deleted else 'normal')
                
                ax_cyto.set_ylim(-0.5, len(chr_cytobands) - 0.5)
                ax_cyto.set_xlim(0, 1)
                ax_cyto.invert_yaxis()
                ax_cyto.set_xticks([])
                ax_cyto.set_yticks([])
                ax_cyto.set_xlabel('')
                ax_cyto.set_ylabel('')
                
                if idx == 0:
                    ax_cyto.set_title('LOH Region', fontsize=10, fontweight='bold')
        else:
            # Simple text representation
            ax_cyto.text(0.5, 0.5, f'{row["cytoband_range"]}\n{row["length_mb"]:.1f} Mb', 
                        ha='center', va='center', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor=colors[idx], alpha=0.5))
            ax_cyto.set_xlim(0, 1)
            ax_cyto.set_ylim(0, 1)
            ax_cyto.axis('off')
    
    # Add x-axis labels at bottom
    ax_bottom = fig.add_subplot(gs[-1, 0])
    ax_bottom.axis('off')
    
    # Add variant position labels
    n_variants = len(chr_variants_sorted)
    positions = [extract_pos(v) for v in chr_variants_sorted]
    
    for i in range(0, n_variants, max(1, n_variants // 20)):  # Show ~20 labels
        x_pos = i / n_variants
        variant = chr_variants_sorted[i]
        pos = extract_pos(variant)
        ax_bottom.text(x_pos, 0.5, f'{pos/1e6:.1f} Mb', 
                      ha='center', va='center', fontsize=7, rotation=90)
    
    ax_bottom.set_xlim(0, 1)
    ax_bottom.set_ylim(0, 1)
    
    # Overall title
    fig.suptitle(f'{chr_arm} LOH Subclones - Cluster {cluster}', 
                fontsize=14, fontweight='bold', y=0.995)
    
    # Save figure
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        fig.savefig(output_path.replace('.png', '.pdf'), dpi=200, bbox_inches='tight')
        vprint(f"\n✓ Saved: {output_path}")
    
    plt.tight_layout()
    plt.show()
    
    return fig, range_groups


def plot_all_loh_subclones_summary(
    df_loh_cytobands: pd.DataFrame,
    cytobands: pd.DataFrame,
    output_path: str = None,
    figsize: tuple = (16, 10),
    verbose: bool = True
):
    """
    Create summary plot showing LOH patterns across all chromosome arms.
    
    Parameters
    ----------
    df_loh_cytobands : pd.DataFrame
        Output from map_loh_to_cytobands
    cytobands : pd.DataFrame
        Cytoband reference data
    output_path : str
        Path to save figure
    figsize : tuple
        Figure size
    verbose : bool
        Print progress
    """
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    
    def vprint(msg):
        if verbose:
            print(msg)
            import sys
            sys.stdout.flush()
    
    vprint("\n" + "="*60)
    vprint("PLOTTING LOH SUMMARY ACROSS ALL CHROMOSOME ARMS")
    vprint("="*60)
    
    # Group by chromosome arm
    chr_arms = sorted(df_loh_cytobands['chr_arm'].unique())
    
    vprint(f"  Chromosome arms to plot: {len(chr_arms)}")
    for chr_arm in chr_arms:
        n_cells = df_loh_cytobands[df_loh_cytobands['chr_arm'] == chr_arm]['cell_id'].nunique()
        vprint(f"    {chr_arm}: {n_cells} cells")
    
    fig, axes = plt.subplots(len(chr_arms), 1, figsize=figsize, 
                            sharex=False, sharey=False)
    
    if len(chr_arms) == 1:
        axes = [axes]
    
    for ax, chr_arm in zip(axes, chr_arms):
        chr_data = df_loh_cytobands[df_loh_cytobands['chr_arm'] == chr_arm]
        
        # Get cytoband info
        chr_cytobands = cytobands[cytobands['chromosome arm'] == chr_arm].sort_values('start')
        
        if len(chr_cytobands) == 0:
            vprint(f"  ⚠️  No cytoband data for {chr_arm}")
            continue
        
        # Get unique ranges
        ranges = chr_data['cytoband_range'].value_counts()
        
        # Plot chromosome ideogram
        for i, (_, cb_row) in enumerate(chr_cytobands.iterrows()):
            # Check if cytoband type column exists
            if 'cytoband type' in cb_row.index:
                color = 'lightgray' if cb_row['cytoband type'] == 'gneg' else 'darkgray'
            else:
                color = 'lightgray'
            
            ax.barh(0, cb_row['end'] - cb_row['start'], left=cb_row['start'], 
                   height=0.3, color=color, edgecolor='black', linewidth=0.5)
        
        # Plot LOH regions
        y_offset = 0.5
        colors = plt.cm.Set3(np.linspace(0, 1, len(ranges)))
        
        for idx, (range_name, count) in enumerate(ranges.items()):
            # Find genomic coordinates
            range_cells = chr_data[chr_data['cytoband_range'] == range_name]
            start = range_cells['genomic_start'].min()
            end = range_cells['genomic_end'].max()
            
            ax.barh(y_offset, end - start, left=start, height=0.2,
                   color=colors[idx], alpha=0.7, label=f'{range_name} (n={count})')
            y_offset += 0.25
        
        ax.set_ylim(-0.2, y_offset)
        ax.set_ylabel(chr_arm, fontsize=12, fontweight='bold')
        ax.set_xlabel('Genomic position (bp)', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='x', alpha=0.3)
    
    fig.suptitle('LOH Patterns Across Chromosome Arms', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        fig.savefig(output_path.replace('.png', '.pdf'), dpi=200, bbox_inches='tight')
        vprint(f"\n✓ Saved: {output_path}")
    
    plt.show()
    
    return fig

def plot_all_chromosomes_single_figure(
    cytobands: pd.DataFrame,
    df_cluster_summary: pd.DataFrame,
    loh_regions: pd.DataFrame = None,
    figsize: tuple = (24, 18),
    output_path: str = None,
    verbose: bool = True
):
    """
    Plot ALL chromosome ideograms in a SINGLE multi-panel figure.
    One subplot per chromosome with LOH fragments displayed ABOVE and cytoband labels BELOW.
    
    Parameters
    ----------
    cytobands : pd.DataFrame
        Cytoband reference data
    df_cluster_summary : pd.DataFrame
        Cluster-level summary from summarize_loh_cytobands_by_cluster
    loh_regions : pd.DataFrame
        LOH regions from map_loh_to_cytobands
    figsize : tuple
        Overall figure size
    output_path : str
        Path to save combined figure
    verbose : bool
        Print progress
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Combined figure with all chromosomes
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Wedge
    import numpy as np
    import re
    
    def vprint(msg):
        if verbose:
            print(msg)
            import sys
            sys.stdout.flush()
    
    def sort_chr_arms(chr_list):
        """Sort chromosome arms in biological order"""
        def chr_key(chr_arm):
            match = re.match(r'chr(\d+|X|Y)([pq])', chr_arm)
            if not match:
                return (999, chr_arm, '')
            
            chr_num, arm = match.groups()
            
            if chr_num == 'X':
                chr_order = 23
            elif chr_num == 'Y':
                chr_order = 24
            else:
                chr_order = int(chr_num)
            
            arm_order = 0 if arm == 'p' else 1
            
            return (chr_order, arm_order, chr_arm)
        
        return sorted(chr_list, key=chr_key)
    
    vprint("\n" + "="*60)
    vprint("PLOTTING ALL CHROMOSOMES - SINGLE FIGURE")
    vprint("="*60)
    
    # Get unique chromosomes (not arms) that have LOH
    chr_arms_with_loh = sorted(df_cluster_summary['chr_arm'].unique())
    
    # Extract unique chromosome names
    chromosomes = sorted(set([
        re.match(r'(chr\d+|chrX|chrY)', arm).group(1) 
        for arm in chr_arms_with_loh 
        if re.match(r'(chr\d+|chrX|chrY)', arm)
    ]), key=lambda x: (int(x[3:]) if x[3:].isdigit() else (23 if x[3:] == 'X' else 24)))
    
    n_chr = len(chromosomes)
    
    vprint(f"\n  Chromosomes to plot: {n_chr}")
    vprint(f"  {', '.join(chromosomes)}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_chr, 1, figsize=figsize)
    
    if n_chr == 1:
        axes = [axes]
    
    # Giemsa colors
    colors = {
        'gneg': 'white',
        'gpos25': '#DDDDDD',
        'gpos50': '#AAAAAA',
        'gpos75': '#777777',
        'gpos100': '#444444',
        'acen': 'white'
    }
    
    # Plot each chromosome
    for ax, chr_name in zip(axes, chromosomes):
        vprint(f"\n  Plotting {chr_name}...")
        
        # Extract chromosome number
        chr_num = chr_name.replace('chr', '')
        
        # Filter cytobands
        chr_data = cytobands[cytobands['chromosome'].astype(str) == chr_num].copy()
        
        if len(chr_data) == 0:
            ax.text(0.5, 0.5, f'No data for {chr_name}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
        
        # Sort by arm and position
        chr_data = chr_data.sort_values(['arm', 'start']).reset_index(drop=True)
        
        # Get p and q arms
        p_arm = chr_data[chr_data['arm'] == 'p']
        q_arm = chr_data[chr_data['arm'] == 'q']
        
        # Get chromosome boundaries
        chr_start = chr_data['start'].min()
        chr_end = chr_data['end'].max()
        chr_length = chr_end - chr_start
        
        # Find centromere position
        if len(p_arm) > 0 and len(q_arm) > 0:
            centro_pos = (p_arm['end'].max() + q_arm['start'].min()) / 2
        else:
            centro_pos = (chr_start + chr_end) / 2
        
        # Set axis limits with space for LOH above AND labels below
        ax.set_xlim(chr_start - chr_length*0.05, chr_end + chr_length*0.05)
        ax.set_ylim(-0.5, 1.5)  # Space above and below
        
        # Chromosome dimensions - TALLER
        chr_y_bottom = 0.3
        chr_y_top = 0.4
        chr_height = chr_y_top - chr_y_bottom
        chr_y_center = (chr_y_bottom + chr_y_top) / 2
        
        radius = chr_height / 2
        
        # Draw cytobands
        for idx, row in chr_data.iterrows():
            x_start = row['start']
            x_end = row['end']
            width = x_end - x_start
            
            band_type = row['cytoband type']
            
            if band_type == 'gneg':
                color = 'white' if idx % 2 == 0 else '#F5F5F5'
            else:
                color = colors.get(band_type, 'lightgray')
            
            is_first = (idx == 0)
            is_last = (idx == len(chr_data) - 1)
            
            if is_first:
                # Rounded left end
                wedge = Wedge((x_start + radius, chr_y_center), radius, 90, 270,
                             facecolor=color, edgecolor='black', linewidth=0.5,
                             transform=ax.transData)
                ax.add_patch(wedge)
                
                if width > radius:
                    rect = patches.Rectangle((x_start + radius, chr_y_bottom),
                                            width - radius, chr_height,
                                            facecolor=color, edgecolor='black',
                                            linewidth=0.5, transform=ax.transData)
                    ax.add_patch(rect)
            
            elif is_last:
                # Rounded right end
                if width > radius:
                    rect = patches.Rectangle((x_start, chr_y_bottom),
                                            width - radius, chr_height,
                                            facecolor=color, edgecolor='black',
                                            linewidth=0.5, transform=ax.transData)
                    ax.add_patch(rect)
                
                wedge = Wedge((x_end - radius, chr_y_center), radius, -90, 90,
                             facecolor=color, edgecolor='black', linewidth=0.5,
                             transform=ax.transData)
                ax.add_patch(wedge)
            
            else:
                # Regular rectangle
                rect = patches.Rectangle((x_start, chr_y_bottom), width, chr_height,
                                        facecolor=color, edgecolor='black',
                                        linewidth=0.5, transform=ax.transData)
                ax.add_patch(rect)
            
            # Add cytoband labels BELOW chromosome (rotated 90°)
            if width > chr_length * 0.008:  # Only label larger bands
                ax.text(x_start + width/2, chr_y_bottom - 0.04,
                       row['cytoband'],
                       ha='center', va='top',
                       fontsize=16, rotation=90,
                       transform=ax.transData)
        
        # Draw centromere line - TALLER
        ax.plot([centro_pos, centro_pos], [chr_y_bottom, chr_y_top],
               'r-', linewidth=5, zorder=10, solid_capstyle='round')
        
        # Add chromosome label
        ax.text(chr_start - chr_length * 0.02, chr_y_center, chr_name,
               ha='right', va='center', fontsize=12, fontweight='bold')
        
        # Plot LOH regions ABOVE chromosome
        chr_summary = df_cluster_summary[
            df_cluster_summary['chr_arm'].str.startswith(chr_name)
        ].copy()
        
        if len(chr_summary) > 0:
            loh_colors = plt.cm.Set3(np.linspace(0, 1, len(chr_summary)))
            
            y_offset = chr_y_top + 0.05  # Start ABOVE chromosome
            
            for idx, (_, row) in enumerate(chr_summary.iterrows()):
                chr_arm = row['chr_arm']
                consensus = row['cytoband_consensus']
                n_cells = row['n_cells']
                
                # Get genomic coordinates
                if loh_regions is not None:
                    arm_loh = loh_regions[
                        (loh_regions['chr_arm'] == chr_arm) &
                        (loh_regions['cytoband_range'] == consensus)
                    ]
                    
                    if len(arm_loh) > 0:
                        loh_start = arm_loh['genomic_start'].min()
                        loh_end = arm_loh['genomic_end'].max()
                        
                        y_loh = y_offset + (idx * 0.20)  # Stack UPWARD
                        loh_height = 0.04  # Taller LOH bars
                        loh_radius = loh_height / 2
                        
                        # Draw rounded LOH bar
                        wedge_left = Wedge((loh_start + loh_radius, y_loh),
                                          loh_radius, 90, 270,
                                          facecolor=loh_colors[idx], alpha=0.7,
                                          edgecolor='black', linewidth=1.5,
                                          transform=ax.transData)
                        ax.add_patch(wedge_left)
                        
                        rect_loh = patches.Rectangle(
                            (loh_start + loh_radius, y_loh - loh_radius),
                            loh_end - loh_start - 2*loh_radius, loh_height,
                            facecolor=loh_colors[idx], alpha=0.7,
                            edgecolor='black', linewidth=1.5,
                            transform=ax.transData)
                        ax.add_patch(rect_loh)
                        
                        wedge_right = Wedge((loh_end - loh_radius, y_loh),
                                           loh_radius, -90, 90,
                                           facecolor=loh_colors[idx], alpha=0.7,
                                           edgecolor='black', linewidth=1.5,
                                           transform=ax.transData)
                        ax.add_patch(wedge_right)
                        
                        # Add label ABOVE the bar
                        ax.text((loh_start + loh_end) / 2, y_loh + loh_radius + 0.05,
                               f'{chr_arm}\n{consensus}\n(n={n_cells})',
                               ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.set_aspect('auto')
        ax.axis('off')
    
    plt.suptitle('Chromosome Ideograms with LOH Regions',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
        vprint(f"\n✓ Saved: {output_path}")
    
    plt.show()
    
    return fig


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
    plot_separate_clones: bool = False,  # NEW PARAMETER
    font_title: int = 14,  # NEW: Font size for clone name title
    verbose: bool = True
) -> None:
    """
    Visualize informative variants by clone with chromosome arm annotations.
    
    Creates heatmaps showing allele frequencies for informative variants,
    organized by chromosome arm, with one subplot per clone label.
    Automatically splits into multiple figures if needed.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cells as rows, variants as columns, and a label column
    variants_to_plot : list
        List of variant IDs to include in visualization
    cytobands : pd.DataFrame
        Cytoband reference with columns: chromosome, arm, start, end, cytoband
    output_dir : str
        Directory to save output figures
    output_prefix : str
        Prefix for output filenames
    label_col : str
        Name of column containing clone labels
    max_height_per_subplot : float
        Maximum height (inches) per subplot
    max_total_height : float
        Maximum total figure height (inches)
    max_subplots_per_figure : int
        Maximum subplots per figure (splits if exceeded)
    dpi : int
        Resolution for saved figures
    plot_categorical : bool
        If True, uses categorical coloring: <10 = dark blue, >90 = dark red, else white
        If False, uses continuous RdYlBu_r colormap
    split_by_cytobands : bool
        If True, adds vertical lines separating cytobands (more granular than chromosome arms)
        If False, only shows chromosome arm boundaries
    desired_clone_plotting_order : list, optional
        Custom order for clone labels. Clones not in the list will be appended at the end.
        Example: ['WT', 'Clone1', 'Clone2']
    plot_separate_clones : bool
        If True, creates separate PNG for each clone with clone name as title.
        If False, creates single figure with all clones stacked (default behavior)
    font_title : int
        Font size for clone name title (only used when plot_separate_clones=True)
    verbose : bool
        Print progress messages
        
    Returns
    -------
    None
        Saves figure(s) to output_dir
    """
    
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage, leaves_list
    
    def vprint(msg):
        if verbose:
            print(msg)
            import sys
            sys.stdout.flush()
    
    def extract_chr_arm_pos(col):
        """Extract chromosome number, arm, and position for sorting"""
        chr_arm = variant_to_arm.get(col, 'chr0p')
        try:
            chr_num = 23 if 'chrX' in chr_arm else int(chr_arm[3:-1])
        except Exception:
            chr_num = 0
        arm = chr_arm[-1]
        pos = int(col.split(':')[1])
        return (chr_num, arm, pos)
    
    def get_cytoband_for_variant(col):
        """Get cytoband name for a variant"""
        return variant_to_cytoband.get(col, '')
    
    def categorize_data(data):
        """Convert continuous AF data to categorical (0=low, 1=mid, 2=high)"""
        categorized = np.zeros_like(data)
        categorized[data <= 10] = 0  # Low AF
        categorized[(data > 10) & (data < 90)] = 1  # Mid AF
        categorized[data >= 90] = 2  # High AF
        return categorized
    
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
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Select informative variants and label column
    df_selected = df.loc[:, variants_to_plot].copy()
    df_selected[label_col] = df[label_col]
    
    vprint(f"\n  Data shape: {df_selected.shape[0]} cells × {len(variants_to_plot)} variants")
    
    # Build chromosome arm lookup
    vprint("\n  Building chromosome arm lookup...")
    
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
    
    vprint(f"    ✓ Lookup table built for {len(arm_lookup)} chromosomes")
    
    # Assign variants to chromosome arms AND cytobands (if needed)
    vprint(f"\n  Assigning variants to chromosome arms{' and cytobands' if split_by_cytobands else ''}...")
    
    variant_cols = [col for col in df_selected.columns if col != label_col]
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
        
        # If splitting by cytobands, also assign to specific cytoband
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
    
    vprint(f"    ✓ Assigned {len(variant_to_arm)} variants to chromosome arms")
    if split_by_cytobands:
        n_cytobands = len(set(variant_to_cytoband.values()))
        vprint(f"    ✓ Assigned variants to {n_cytobands} unique cytobands")
    
    # Sort columns by chromosome and position
    sorted_cols = sorted(variant_cols, key=extract_chr_arm_pos)
    df_selected = df_selected[sorted_cols + [label_col]]
    
    # Create boundary list based on mode
    if split_by_cytobands:
        boundary_list = pd.Index(sorted_cols).map(get_cytoband_for_variant)
    else:
        boundary_list = pd.Index(sorted_cols).map(lambda col: variant_to_arm.get(col, ''))
    
    # Get unique labels and apply custom ordering
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
    
    # Setup colormap
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
    
    # =========================================================================
    # HELPER FUNCTION: PLOT SINGLE CLONE
    # =========================================================================
    
    def plot_single_clone(ax, label, show_ylabel=True):
        """Plot heatmap for a single clone"""
        df_label = df_selected[df_selected[label_col] == label].drop(columns=label_col)
        
        if len(df_label) == 0:
            vprint(f"    Skipping {label}: no cells")
            ax.text(0.5, 0.5, f'{label}\n(No cells)',
                   ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        # Cluster cells
        try:
            linkage_matrix = linkage(df_label, method='average')
            ordered_indices = leaves_list(linkage_matrix)
            df_label = df_label.iloc[ordered_indices, :]
        except Exception as e:
            vprint(f"    Clustering failed for {label}: {e}")
        
        # Prepare data for plotting
        if plot_categorical:
            plot_data = categorize_data(df_label.values)
            plot_data = pd.DataFrame(
                plot_data, 
                index=df_label.index, 
                columns=df_label.columns
            )
        else:
            plot_data = df_label
        
        # Plot heatmap
        sns.heatmap(plot_data, cbar=True, ax=ax, yticklabels=False,
                   cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
        
        # Add boundaries
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
        
        # Style
        ax.set_yticklabels([])
        ax.tick_params(axis='x', labelsize=8, labelrotation=90)
        
        if show_ylabel:
            ax.set_ylabel(f'{label} (n={len(df_label)})',
                         fontsize=12, fontweight='bold', rotation=90, labelpad=15)
        else:
            ax.set_ylabel('')
        
        # Adjust colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=9)
        
        if plot_categorical:
            cbar.set_ticks([0.5, 1.5, 2.5])
            cbar.set_ticklabels(['≤10%', '10-90%', '≥90%'])
            cbar.set_label('AF Category', fontsize=9)
        else:
            cbar.set_label('AF (%)', fontsize=9)
    
    # =========================================================================
    # BRANCH: SEPARATE vs COMBINED FIGURES
    # =========================================================================
    
    if plot_separate_clones:
        # =====================================================================
        # MODE 1: SEPARATE FIGURES PER CLONE
        # =====================================================================
        vprint("\n  Creating separate figures for each clone...")
        
        suffix = "_by_cytoband" if split_by_cytobands else "_by_chr_arm"
        
        for i, label in enumerate(unique_labels, 1):
            vprint(f"\n  Plotting clone {i}/{n_labels}: {label}")
            
            # Create single-panel figure
            fig, ax = plt.subplots(1, 1, figsize=(18, max_height_per_subplot))
            
            # Plot without ylabel (since we'll use title)
            plot_single_clone(ax, label, show_ylabel=False)
            
            # Add title with clone name
            n_cells = (df_selected[label_col] == label).sum()
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save with clone name in filename
            clone_filename = f"{output_prefix}{suffix}_{label.replace(' ', '_').replace('/', '_')}.png"
            output_path = os.path.join(output_dir, clone_filename)
            if save_figures:
                fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
                fig.savefig(output_path.replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight')
            vprint(f"    ✓ Saved: {clone_filename}")
            plt.show()
            plt.close(fig)
    
    else:
        # =====================================================================
        # MODE 2: COMBINED FIGURE (ORIGINAL BEHAVIOR)
        # =====================================================================
        
        height_per_subplot = min(max_height_per_subplot, max_total_height / n_labels)
        total_height = height_per_subplot * n_labels
        
        # Split into multiple figures if needed
        if n_labels > max_subplots_per_figure:
            vprint(f"\n  ⚠️  Too many labels ({n_labels}). Splitting into multiple figures...")
            n_figures = int(np.ceil(n_labels / max_subplots_per_figure))
            
            for fig_idx in range(n_figures):
                start_idx = fig_idx * max_subplots_per_figure
                end_idx = min((fig_idx + 1) * max_subplots_per_figure, n_labels)
                labels_subset = unique_labels[start_idx:end_idx]
                n_subplots = len(labels_subset)
                
                vprint(f"\n  Creating figure {fig_idx + 1}/{n_figures} with {n_subplots} subplots...")
                
                fig, axes = plt.subplots(
                    n_subplots, 1,
                    figsize=(18, min(8 * n_subplots, max_total_height)),
                    sharex=True
                )
                
                if n_subplots == 1:
                    axes = [axes]
                
                for ax, label in zip(axes, labels_subset):
                    plot_single_clone(ax, label, show_ylabel=True)
                
                plt.tight_layout()
                
                suffix = "_by_cytoband" if split_by_cytobands else "_by_chr_arm"
                output_path = os.path.join(output_dir, f"{output_prefix}{suffix}_part{fig_idx + 1}.png")
                fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
                fig.savefig(output_path.replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight')
                vprint(f"    ✓ Saved: {os.path.basename(output_path)}")
                plt.show()
                plt.close(fig)
        
        else:
            # Single figure
            vprint(f"\n  Creating single figure with {n_labels} subplots...")
            
            fig, axes = plt.subplots(
                n_labels, 1,
                figsize=(18, min(total_height, max_total_height)),
                sharex=True
            )
            
            if n_labels == 1:
                axes = [axes]
            
            for ax, label in zip(axes, unique_labels):
                plot_single_clone(ax, label, show_ylabel=True)
            
            plt.tight_layout()
            
            suffix = "_by_cytoband" if split_by_cytobands else "_by_chr_arm"
            output_path = os.path.join(output_dir, f"{output_prefix}{suffix}.png")
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            fig.savefig(output_path.replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight')
            vprint(f"    ✓ Saved: {os.path.basename(output_path)}")
            plt.show()
            plt.close(fig)
    
    vprint("\n" + "="*60)
    vprint("VISUALIZATION COMPLETE")
    vprint("="*60)

def visualise_binary_variants_by_clone(
    df: pd.DataFrame,
    wilcoxon_results_var: dict,
    cytobands: pd.DataFrame,
    output_dir: str,
    output_prefix: str = "Single_cell_clones_AF_Informative_by_chr_arm",
    label_col: str = 'label',
    max_height_per_subplot: float = 8.0,
    max_total_height: float = 200.0,
    max_subplots_per_figure: int = 25,
    dpi: int = 150,
    split_by_cytobands: bool = False,
    desired_clone_plotting_order: list = None,
    plot_separate_clones: bool = False,  # NEW PARAMETER
    font_title: int = 14,  # NEW: Font size for clone name title
    verbose: bool = True
) -> None:
    """
    Visualize AF variants by informativeness (binary: informative vs non-informative).
    
    Uses Wilcoxon test results to determine which variants are informative for each clone.
    Informative variants are shown in BLACK, non-informative in WHITE.
    
    Parameters
    ----------
    df : pd.DataFrame
        Allele frequency data with cells as rows, AF variants as columns, and 'label' column
    wilcoxon_results_var : dict
        Dictionary from test_variant_informativeness with chromosome arms as keys,
        DataFrames with 'variant', 'label', 'informative' columns as values
    cytobands : pd.DataFrame
        Cytoband reference with columns: chromosome, arm, start, end, cytoband
    output_dir : str
        Directory to save output figures
    output_prefix : str
        Prefix for output filenames
    label_col : str
        Name of column containing clone labels
    max_height_per_subplot : float
        Maximum height (inches) per subplot
    max_total_height : float
        Maximum total figure height (inches)
    max_subplots_per_figure : int
        Maximum subplots per figure (splits if exceeded)
    dpi : int
        Resolution for saved figures
    split_by_cytobands : bool
        If True, adds vertical lines separating cytobands
        If False, only shows chromosome arm boundaries
    desired_clone_plotting_order : list, optional
        Custom order for clone labels
    plot_separate_clones : bool
        If True, creates separate PNG for each clone with clone name as title.
        If False, creates single figure with all clones stacked (default behavior)
    font_title : int
        Font size for clone name title (only used when plot_separate_clones=True)
    verbose : bool
        Print progress messages
        
    Returns
    -------
    None
        Saves figure(s) to output_dir
    """
    
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage, leaves_list
    
    def vprint(msg):
        if verbose:
            print(msg)
            import sys
            sys.stdout.flush()
    
    def extract_chr_arm_pos(col):
        """Extract chromosome number, arm, and position for sorting"""
        chr_arm = variant_to_arm.get(col, 'chr0p')
        try:
            chr_num = 23 if 'chrX' in chr_arm else int(chr_arm[3:-1])
        except Exception:
            chr_num = 0
        arm = chr_arm[-1]
        pos = int(col.split(':')[1])
        return (chr_num, arm, pos)
    
    def get_cytoband_for_variant(col):
        """Get cytoband name for a variant"""
        return variant_to_cytoband.get(col, '')
    
    vprint("\n" + "="*60)
    vprint("VISUALIZING AF VARIANT INFORMATIVENESS BY CLONE")
    vprint("="*60)
    
    if plot_separate_clones:
        vprint("MODE: Separate figures per clone (clone name as title)")
    else:
        vprint("MODE: Single figure with all clones stacked")
    
    vprint("  Color scheme: BINARY (non-informative=WHITE, informative=BLACK)")
    
    if split_by_cytobands:
        vprint("  Boundaries: CYTOBANDS (individual cytobands separated)")
    else:
        vprint("  Boundaries: CHROMOSOME ARMS (p/q arms separated)")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Build informative variant lookup: (clone_label, variant) -> is_informative
    vprint("\n  Building informative variant lookup from Wilcoxon results...")
    
    informative_lookup = {}
    total_comparisons = 0
    total_informative = 0
    
    for chr_arm, results_df in wilcoxon_results_var.items():
        for _, row in results_df.iterrows():
            variant = row['variant']
            label = row['label']
            is_informative = row['informative']
            
            informative_lookup[(label, variant)] = is_informative
            total_comparisons += 1
            if is_informative:
                total_informative += 1
    
    vprint(f"    ✓ Loaded {total_comparisons} variant-clone comparisons")
    vprint(f"    ✓ Informative: {total_informative} ({100*total_informative/total_comparisons:.1f}%)")
    
    # Get AF variants (exclude metadata columns)
    af_variants = [col for col in df.columns 
                   if col not in ['label', 'barcode'] and ':' in str(col)]
    
    vprint(f"\n  AF variants in data: {len(af_variants)}")
    
    # Select AF variants and label column
    df_selected = df[af_variants + [label_col]].copy()
    
    vprint(f"  Data shape: {df_selected.shape[0]} cells × {len(af_variants)} variants")
    
    # Build chromosome arm lookup
    vprint("\n  Building chromosome arm lookup...")
    
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
    
    vprint(f"    ✓ Lookup table built for {len(arm_lookup)} chromosomes")
    
    # Assign variants to chromosome arms AND cytobands
    vprint(f"\n  Assigning variants to chromosome arms{' and cytobands' if split_by_cytobands else ''}...")
    
    variant_to_arm = {}
    variant_to_cytoband = {}
    
    for col in af_variants:
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
    
    vprint(f"    ✓ Assigned {len(variant_to_arm)} variants to chromosome arms")
    if split_by_cytobands:
        n_cytobands = len(set(variant_to_cytoband.values()))
        vprint(f"    ✓ Assigned variants to {n_cytobands} unique cytobands")
    
    # Sort columns by chromosome and position
    sorted_cols = sorted(af_variants, key=extract_chr_arm_pos)
    df_selected = df_selected[sorted_cols + [label_col]]
    
    # Create boundary list
    if split_by_cytobands:
        boundary_list = pd.Index(sorted_cols).map(get_cytoband_for_variant)
    else:
        boundary_list = pd.Index(sorted_cols).map(lambda col: variant_to_arm.get(col, ''))
    
    # Get unique labels and apply custom ordering
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
    
    vprint(f"\n  Clone labels to plot: {n_labels} (including WT)")
    vprint(f"  Final plotting order:")
    for idx, label in enumerate(unique_labels, 1):
        n_cells = (df_selected[label_col] == label).sum()
        
        # Count informative variants for this clone
        n_informative = sum(1 for v in sorted_cols if informative_lookup.get((label, v), False))
        
        # Special message for WT
        if label == 'WT':
            vprint(f"    {idx}. {label} (n={n_cells} cells, BASELINE)")
        else:
            vprint(f"    {idx}. {label} (n={n_cells} cells, {n_informative} informative variants)")
    
    # Setup colormap: white (0) vs black (1)
    colors_list = ['white', 'black']
    cmap = mcolors.ListedColormap(colors_list)
    bounds = [0, 0.5, 1]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    vmin, vmax = 0, 1
    
    # =========================================================================
    # HELPER FUNCTION: PLOT SINGLE CLONE
    # =========================================================================
    
    def plot_single_clone(ax, label, show_ylabel=True):
        """Plot heatmap for a single clone"""
        df_label = df_selected[df_selected[label_col] == label].drop(columns=label_col)
        
        if len(df_label) == 0:
            vprint(f"    Skipping {label}: no cells")
            ax.text(0.5, 0.5, f'{label}\n(No cells)',
                   ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        # Cluster cells
        try:
            linkage_matrix = linkage(df_label, method='average')
            ordered_indices = leaves_list(linkage_matrix)
            df_label = df_label.iloc[ordered_indices, :]
        except Exception as e:
            vprint(f"    Clustering failed for {label}: {e}")
        
        # Create binary informativeness matrix
        plot_data = np.zeros_like(df_label.values)
        
        for col_idx, variant in enumerate(df_label.columns):
            is_informative = informative_lookup.get((label, variant), False)
            if is_informative:
                plot_data[:, col_idx] = 1  # Black
            # else: stays 0 (white)
        
        plot_data = pd.DataFrame(
            plot_data,
            index=df_label.index,
            columns=df_label.columns
        )
        
        # Plot heatmap
        sns.heatmap(plot_data, cbar=True, ax=ax, yticklabels=False,
                   cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
        
        # Add boundaries
        prev_boundary = None
        for j, current_boundary in enumerate(boundary_list):
            if current_boundary != prev_boundary:
                line_width = 4 if split_by_cytobands else 2
                line_color = 'red'  # Use red for visibility
                ax.axvline(x=j, color=line_color, linestyle='--', linewidth=line_width)
                
                if j > 0:
                    fontsize = 6 if split_by_cytobands else 8
                    ax.text(j, ax.get_ylim()[1] - 2, current_boundary, 
                           ha='center', va='bottom',
                           color='black', fontsize=fontsize, fontweight='bold',
                           rotation=90)
                prev_boundary = current_boundary
        
        # Style
        ax.set_yticklabels([])
        ax.tick_params(axis='x', labelsize=8, labelrotation=90)
        
        # Count informative variants
        n_informative = sum(1 for v in df_label.columns if informative_lookup.get((label, v), False))
        
        if show_ylabel:
            # Style with different label for WT
            if label == 'WT':
                ylabel = f'{label}\n(n={len(df_label)})\nBASELINE'
            else:
                ylabel = f'{label}\n(n={len(df_label)})\n{n_informative} informative'
            
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', rotation=90, labelpad=15)
        else:
            ax.set_ylabel('')
        
        # Adjust colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=9)
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(['Not Informative', 'Informative'])
        cbar.set_label('AF Informativeness', fontsize=9)
    
    # =========================================================================
    # BRANCH: SEPARATE vs COMBINED FIGURES
    # =========================================================================
    
    if plot_separate_clones:
        # =====================================================================
        # MODE 1: SEPARATE FIGURES PER CLONE
        # =====================================================================
        vprint("\n  Creating separate figures for each clone...")
        
        suffix = "_informative_by_cytoband" if split_by_cytobands else "_informative_by_chr_arm"
        
        for i, label in enumerate(unique_labels, 1):
            vprint(f"\n  Plotting clone {i}/{n_labels}: {label}")
            
            # Create single-panel figure
            fig, ax = plt.subplots(1, 1, figsize=(18, max_height_per_subplot))
            
            # Plot without ylabel (since we'll use title)
            plot_single_clone(ax, label, show_ylabel=False)
            
            # Add title with clone name
            n_cells = (df_selected[label_col] == label).sum()
            n_informative = sum(1 for v in sorted_cols if informative_lookup.get((label, v), False))
            
            if label == 'WT':
                title_text = f'{label} (n={n_cells}) - BASELINE'
            else:
                title_text = f'{label} (n={n_cells}) - {n_informative} informative variants'
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save with clone name in filename
            clone_filename = f"{output_prefix}{suffix}_{label.replace(' ', '_').replace('/', '_')}.png"
            output_path = os.path.join(output_dir, clone_filename)
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            fig.savefig(output_path.replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight')
            vprint(f"    ✓ Saved: {clone_filename}")
            plt.show()
            plt.close(fig)
    
    else:
        # =====================================================================
        # MODE 2: COMBINED FIGURE (ORIGINAL BEHAVIOR)
        # =====================================================================
        
        # Calculate figure dimensions
        height_per_subplot = min(max_height_per_subplot, max_total_height / n_labels)
        total_height = height_per_subplot * n_labels
        
        # Split into multiple figures if needed
        if n_labels > max_subplots_per_figure:
            vprint(f"\n  ⚠️  Too many labels ({n_labels}). Splitting into multiple figures...")
            n_figures = int(np.ceil(n_labels / max_subplots_per_figure))
            
            for fig_idx in range(n_figures):
                start_idx = fig_idx * max_subplots_per_figure
                end_idx = min((fig_idx + 1) * max_subplots_per_figure, n_labels)
                labels_subset = unique_labels[start_idx:end_idx]
                n_subplots = len(labels_subset)
                
                vprint(f"\n  Creating figure {fig_idx + 1}/{n_figures} with {n_subplots} subplots...")
                
                fig, axes = plt.subplots(
                    n_subplots, 1,
                    figsize=(18, min(8 * n_subplots, max_total_height)),
                    sharex=True
                )
                
                if n_subplots == 1:
                    axes = [axes]
                
                for ax, label in zip(axes, labels_subset):
                    plot_single_clone(ax, label, show_ylabel=True)
                
                plt.tight_layout()
                
                suffix = "_informative_by_cytoband" if split_by_cytobands else "_informative_by_chr_arm"
                output_path = os.path.join(output_dir, f"{output_prefix}{suffix}_part{fig_idx + 1}.png")
                fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
                fig.savefig(output_path.replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight')
                vprint(f"    ✓ Saved: {os.path.basename(output_path)}")
                plt.show()
                plt.close(fig)
        
        else:
            # Single figure
            vprint(f"\n  Creating single figure with {n_labels} subplots...")
            
            fig, axes = plt.subplots(
                n_labels, 1,
                figsize=(18, min(total_height, max_total_height)),
                sharex=True
            )
            
            if n_labels == 1:
                axes = [axes]
            
            for ax, label in zip(axes, unique_labels):
                plot_single_clone(ax, label, show_ylabel=True)
            
            plt.tight_layout()
            
            suffix = "_informative_by_cytoband" if split_by_cytobands else "_informative_by_chr_arm"
            output_path = os.path.join(output_dir, f"{output_prefix}{suffix}.png")
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            fig.savefig(output_path.replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight')
            vprint(f"    ✓ Saved: {os.path.basename(output_path)}")
            plt.show()
            plt.close(fig)
    
    vprint("\n" + "="*60)
    vprint("VISUALIZATION COMPLETE")
    vprint("="*60)

def visualise_variants_by_clone_grouped(
    df_AF_grouped: pd.DataFrame,
    variant_info: pd.DataFrame,
    output_path: str,
    cytobands: pd.DataFrame = None,
    desired_clone_plotting_order: List[str] = None,
    plot_categorical: bool = False,
    split_by_cytobands: bool = False,
    figsize: tuple = (20, 6),
    dpi: int = 300,
    verbose: bool = True
) -> None:
    """
    Plot grouped variant allele frequencies (variant × clone) as a single heatmap.
    - Rows: clones (ordered by desired_clone_plotting_order if provided)
    - Columns: variants (genomic order using variant_info)
    - Palette: 
        * Continuous: RdYlBu_r (vmin=0, vmax=100, center=50)
        * Categorical: <10%=dark blue, 10-90%=white, >90%=dark red
    - Adds vertical chromosome separators and headers.

    Parameters
    ----------
    df_AF_grouped : pd.DataFrame
        Variant (index) × clone (columns) AF medians/means.
    variant_info : pd.DataFrame
        Variant metadata with at least ['CHROM', 'POS', 'id'].
        Can be extracted from Sample.dna.col_attrs.
    output_path : str
        Path to save the figure.
    cytobands : pd.DataFrame, optional
        Cytoband reference with columns: chromosome, arm, start, end, cytoband.
        Required if split_by_cytobands=True.
    desired_clone_plotting_order : List[str], optional
        Preferred clone ordering; others appended after.
    plot_categorical : bool
        If True, uses categorical coloring; else continuous AF.
    split_by_cytobands : bool
        If True, adds vertical lines separating cytobands (more granular).
        If False, only shows chromosome arm boundaries.
        Requires cytobands parameter.
    figsize : tuple
        Figure size.
    dpi : int
        Output resolution.
    verbose : bool
        Verbose logging.
    """
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    def vprint(msg):
        if verbose:
            print(msg)

    def chrom_key(series: pd.Series):
        """Sort key for chromosome names like '1','2',...,'X','Y'"""
        def chrom_to_int(chrom):
            c = str(chrom).upper().replace('CHR', '')
            if c == 'X': return 23
            if c == 'Y': return 24
            if c in ('M', 'MT'): return 25
            try:
                return int(c)
            except ValueError:
                return 999
        return series.map(chrom_to_int)

    vprint("\n" + "="*60)
    vprint("VISUALIZING GROUPED VARIANT ALLELE FREQUENCIES")
    vprint("="*60)
    
    if plot_categorical:
        vprint("  Color scheme: CATEGORICAL (<10%=dark blue, 10-90%=white, >90%=dark red)")
    else:
        vprint("  Color scheme: CONTINUOUS (RdYlBu_r, 0-100%, center at 50%)")
    
    if split_by_cytobands:
        vprint("  Boundaries: CYTOBANDS (individual cytobands separated)")
        if cytobands is None:
            vprint("  ❌ ERROR: split_by_cytobands=True requires cytobands parameter")
            return
    else:
        vprint("  Boundaries: CHROMOSOME ARMS (chr separators only)")

    # Validate variant_info
    required = {'CHROM', 'POS', 'id'}
    missing = required - set(variant_info.columns)
    if missing:
        raise KeyError(f"variant_info missing required columns: {missing}")

    # Sort variants by genomic position
    variant_info_sorted = variant_info.copy()
    variant_info_sorted['POS'] = pd.to_numeric(variant_info_sorted['POS'], errors='coerce')
    variant_info_sorted = variant_info_sorted.sort_values(by=['CHROM', 'POS'], key=chrom_key)

    # Align df_AF_grouped rows to genomic order
    ordered_variants = [vid for vid in variant_info_sorted['id'] if vid in df_AF_grouped.index]
    
    if len(ordered_variants) == 0:
        vprint("  ⚠️  No variants to plot")
        return
    
    df_AF_grouped = df_AF_grouped.reindex(ordered_variants)
    vprint(f"\n  Variants to plot: {len(ordered_variants)}")

    # Build clone ordering
    if desired_clone_plotting_order:
        ordered_clones = [c for c in desired_clone_plotting_order if c in df_AF_grouped.columns]
        other_clones = [c for c in df_AF_grouped.columns if c not in ordered_clones]
        df_plot = df_AF_grouped[ordered_clones + sorted(other_clones)]
        vprint(f"  Clones (ordered): {len(ordered_clones)}")
        if other_clones:
            vprint(f"  Clones (additional): {len(other_clones)}")
    else:
        df_plot = df_AF_grouped
        vprint(f"  Clones: {len(df_plot.columns)}")

    # Prepare data for heatmap
    if plot_categorical:
        # Categorical AF: <10%=low(0), 10-90%=mid(1), >90%=high(2)
        vals = df_plot.apply(pd.to_numeric, errors='coerce')
        cats = np.zeros_like(vals.values, dtype=int)
        cats[vals.values < 10] = 0
        cats[(vals.values >= 10) & (vals.values <= 90)] = 1
        cats[vals.values > 90] = 2
        cats_df = pd.DataFrame(cats, index=vals.index, columns=vals.columns)

        cmap = mcolors.ListedColormap(['#00008B', 'white', '#8B0000'])
        norm = mcolors.BoundaryNorm([0, 1, 2, 3], cmap.N)
        data_to_plot = cats_df.T
        heatmap_kwargs = dict(cmap=cmap, norm=norm, vmin=0, vmax=2)
        cbar_label = 'AF (categorical)'
    else:
        # Continuous AF (0-100%)
        vals = df_plot.apply(pd.to_numeric, errors='coerce').clip(lower=0, upper=100)
        data_to_plot = vals.T
        heatmap_kwargs = dict(cmap='RdYlBu_r', vmin=0, vmax=100, center=50)
        cbar_label = 'Allele Frequency (%)'

    vprint(f"\n  Plotting {data_to_plot.shape[0]} clones × {data_to_plot.shape[1]} variants")

    # Plot
    plt.figure(figsize=figsize)
    sns.set(font_scale=0.9)

    ax = sns.heatmap(
        data_to_plot,
        cbar=True,
        xticklabels=False,
        **heatmap_kwargs
    )

    # Build boundary list based on mode
    if split_by_cytobands and cytobands is not None:
        # Map variants to cytobands for more granular separators
        vprint("\n  Mapping variants to cytobands...")
        
        variant_to_cytoband = {}
        for variant_id in ordered_variants:
            # Get chromosome and position for this variant
            variant_data = variant_info_sorted[variant_info_sorted['id'] == variant_id]
            if len(variant_data) == 0:
                continue
            
            chrom = str(variant_data.iloc[0]['CHROM'])
            pos = variant_data.iloc[0]['POS']
            
            # Find matching cytoband
            matching_bands = cytobands[
                (cytobands['chromosome'].astype(str) == chrom) &
                (cytobands['start'] <= pos) &
                (cytobands['end'] >= pos)
            ]
            
            if len(matching_bands) > 0:
                cytoband_name = matching_bands.iloc[0]['cytoband']
                variant_to_cytoband[variant_id] = f"chr{chrom}{cytoband_name}"
            else:
                # Fallback to chromosome arm
                arm_info = cytobands[cytobands['chromosome'].astype(str) == chrom]
                if len(arm_info) > 0:
                    if pos < arm_info['start'].median():
                        variant_to_cytoband[variant_id] = f"chr{chrom}p"
                    else:
                        variant_to_cytoband[variant_id] = f"chr{chrom}q"
                else:
                    variant_to_cytoband[variant_id] = f"chr{chrom}"
        
        boundary_list = [variant_to_cytoband.get(vid, '') for vid in ordered_variants]
        vprint(f"    ✓ Mapped to {len(set(boundary_list))} unique cytobands")
    else:
        # Build chromosome labels for separators (columns = variants)
        boundary_list = ['chr' + str(variant_info_sorted[variant_info_sorted['id'] == vid]['CHROM'].iloc[0]) 
                        if vid in variant_info_sorted['id'].values else '' 
                        for vid in ordered_variants]

    # Add separators and labels
    prev_boundary = None
    for j, current_boundary in enumerate(boundary_list):
        if current_boundary != prev_boundary:
            line_width = 4 if split_by_cytobands else 2
            line_color = 'black' if plot_categorical else 'grey'
            
            ax.axvline(x=j, color=line_color, linestyle='--', linewidth=line_width)
            
            # Add label (skip empty labels)
            if current_boundary and j > 0:
                fontsize = 6 if split_by_cytobands else 8
                ax.text(
                    j, ax.get_ylim()[1] - 0.2, current_boundary,
                    ha='center', va='bottom',
                    color='black', fontsize=fontsize, fontweight='bold', rotation=90
                )
            prev_boundary = current_boundary

    # Y-axis styling (clone labels)
    ax.tick_params(axis='y', labelsize=8, labelrotation=0)
    for label in ax.get_yticklabels():
        label.set_weight('bold')

    # Colorbar styling
    if ax.collections:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(cbar_label, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=dpi)
    plt.show()
    plt.close()

    vprint(f"\n✓ Saved: {output_path}")
    vprint("="*60 + "\n")

import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import ranksums

import pandas as pd
import numpy as np

def visualise_variants_ploidy_by_clone(
    df: pd.DataFrame,
    variants_to_plot: list,
    cytobands: pd.DataFrame,
    output_dir: str,
    output_prefix: str = "Single_cell_clones_CNV_Ploidy_by_chr_arm",
    label_col: str = 'label',
    max_height_per_subplot: float = 8.0,
    max_total_height: float = 200.0,
    max_subplots_per_figure: int = 25,
    dpi: int = 150,
    plot_categorical: bool = False,
    split_by_cytobands: bool = False,
    desired_clone_plotting_order: list = None,
    plot_separate_clones: bool = False,  # NEW PARAMETER
    font_title: int = 14,  # NEW: Font size for clone name title
    verbose: bool = True
) -> None:
    """
    Visualize CNV ploidy by clone with chromosome arm annotations.
    
    Creates heatmaps showing copy number (ploidy) for CNV variants,
    organized by chromosome arm, with one subplot per clone label.
    Automatically splits into multiple figures if needed.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cells as rows, variants as columns, and a label column
    variants_to_plot : list
        List of variant IDs to include in visualization
    cytobands : pd.DataFrame
        Cytoband reference with columns: chromosome, arm, start, end, cytoband
    output_dir : str
        Directory to save output figures
    output_prefix : str
        Prefix for output filenames
    label_col : str
        Name of column containing clone labels
    max_height_per_subplot : float
        Maximum height (inches) per subplot
    max_total_height : float
        Maximum total figure height (inches)
    max_subplots_per_figure : int
        Maximum subplots per figure (splits if exceeded)
    dpi : int
        Resolution for saved figures
    plot_categorical : bool
        If True, uses categorical coloring: <1.5 = dark blue (#00008B), 
        1.5-2.5 = white, >2.5 = dark red (#8B0000)
        If False, uses continuous RdYlBu_r colormap (vmin=0, vmax=4, center=2)
    split_by_cytobands : bool
        If True, adds vertical lines separating cytobands
        If False, only shows chromosome arm boundaries
    desired_clone_plotting_order : list, optional
        Custom order for clone labels
    plot_separate_clones : bool
        If True, creates separate PNG for each clone with clone name as title.
        If False, creates single figure with all clones stacked (default behavior)
    font_title : int
        Font size for clone name title (only used when plot_separate_clones=True)
    verbose : bool
        Print progress messages
        
    Returns
    -------
    None
        Saves figure(s) to output_dir
    """
    
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage, leaves_list
    
    def vprint(msg):
        if verbose:
            print(msg)
            import sys
            sys.stdout.flush()
    
    def extract_chr_arm_pos(col):
        """Extract chromosome number, arm, and position for sorting"""
        chr_arm = variant_to_arm.get(col, 'chr0p')
        try:
            chr_num = 23 if 'chrX' in chr_arm else int(chr_arm[3:-1])
        except Exception:
            chr_num = 0
        arm = chr_arm[-1]
        pos = int(col.split(':')[1])
        return (chr_num, arm, pos)
    
    def get_cytoband_for_variant(col):
        """Get cytoband name for a variant"""
        return variant_to_cytoband.get(col, '')
    
    def categorize_ploidy(data):
        """Convert continuous ploidy data to categorical (0=loss, 1=neutral, 2=gain)"""
        categorized = np.zeros_like(data)
        categorized[data < 1.5] = 0  # Loss (< 1.5)
        categorized[(data >= 1.5) & (data <= 2.5)] = 1  # Neutral (1.5-2.5)
        categorized[data > 2.5] = 2  # Gain (> 2.5)
        return categorized
    
    vprint("\n" + "="*60)
    vprint("VISUALIZING CNV PLOIDY BY CLONE")
    vprint("="*60)
    
    if plot_separate_clones:
        vprint("MODE: Separate figures per clone (clone name as title)")
    else:
        vprint("MODE: Single figure with all clones stacked")
    
    if plot_categorical:
        vprint("  Color scheme: CATEGORICAL (<1.5=dark blue, 1.5-2.5=white, >2.5=dark red)")
    else:
        vprint("  Color scheme: CONTINUOUS (RdYlBu_r, 0-4, center at 2)")
    
    if split_by_cytobands:
        vprint("  Boundaries: CYTOBANDS (individual cytobands separated)")
    else:
        vprint("  Boundaries: CHROMOSOME ARMS (p/q arms separated)")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Select informative variants and label column
    df_selected = df.loc[:, variants_to_plot].copy()
    df_selected[label_col] = df[label_col]
    
    vprint(f"\n  Data shape: {df_selected.shape[0]} cells × {len(variants_to_plot)} variants")
    
    # Build chromosome arm lookup
    vprint("\n  Building chromosome arm lookup...")
    
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
    
    vprint(f"    ✓ Lookup table built for {len(arm_lookup)} chromosomes")
    
    # Assign variants to chromosome arms AND cytobands (if needed)
    vprint(f"\n  Assigning variants to chromosome arms{' and cytobands' if split_by_cytobands else ''}...")
    
    variant_cols = [col for col in df_selected.columns if col != label_col]
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
        
        # If splitting by cytobands, also assign to specific cytoband
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
    
    vprint(f"    ✓ Assigned {len(variant_to_arm)} variants to chromosome arms")
    if split_by_cytobands:
        n_cytobands = len(set(variant_to_cytoband.values()))
        vprint(f"    ✓ Assigned variants to {n_cytobands} unique cytobands")
    
    # Sort columns by chromosome and position
    sorted_cols = sorted(variant_cols, key=extract_chr_arm_pos)
    df_selected = df_selected[sorted_cols + [label_col]]
    
    # Create boundary list based on mode
    if split_by_cytobands:
        boundary_list = pd.Index(sorted_cols).map(get_cytoband_for_variant)
    else:
        boundary_list = pd.Index(sorted_cols).map(lambda col: variant_to_arm.get(col, ''))
    
    # Get unique labels and apply custom ordering
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
    
    # Setup colormap
    if plot_categorical:
        # Categorical colormap: dark blue (#00008B), white, dark red (#8B0000)
        colors_list = ['#00008B', 'white', '#8B0000']
        cmap = mcolors.ListedColormap(colors_list)
        bounds = [0, 1, 2, 3]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        vmin, vmax = 0, 2
    else:
        # Continuous colormap: RdYlBu_r with center at 2
        cmap = 'RdYlBu_r'
        norm = None
        vmin, vmax = 0, 4
    
    # =========================================================================
    # HELPER FUNCTION: PLOT SINGLE CLONE
    # =========================================================================
    
    def plot_single_clone(ax, label, show_ylabel=True):
        """Plot heatmap for a single clone"""
        df_label = df_selected[df_selected[label_col] == label].drop(columns=label_col)
        
        if len(df_label) == 0:
            vprint(f"    Skipping {label}: no cells")
            ax.text(0.5, 0.5, f'{label}\n(No cells)',
                   ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        # Cluster cells
        try:
            linkage_matrix = linkage(df_label, method='average')
            ordered_indices = leaves_list(linkage_matrix)
            df_label = df_label.iloc[ordered_indices, :]
        except Exception as e:
            vprint(f"    Clustering failed for {label}: {e}")
        
        # Prepare data for plotting
        if plot_categorical:
            plot_data = categorize_ploidy(df_label.values)
            plot_data = pd.DataFrame(
                plot_data, 
                index=df_label.index, 
                columns=df_label.columns
            )
        else:
            plot_data = df_label
        
        # Plot heatmap
        sns.heatmap(plot_data, cbar=True, ax=ax, yticklabels=False,
                   cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, center=2 if not plot_categorical else None)
        
        # Add boundaries
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
        
        # Style
        ax.set_yticklabels([])
        ax.tick_params(axis='x', labelsize=8, labelrotation=90)
        
        if show_ylabel:
            ax.set_ylabel(f'{label} (n={len(df_label)})',
                         fontsize=12, fontweight='bold', rotation=90, labelpad=15)
        else:
            ax.set_ylabel('')
        
        # Adjust colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=9)
        
        if plot_categorical:
            cbar.set_ticks([0.5, 1.5, 2.5])
            cbar.set_ticklabels(['<1.5', '1.5-2.5', '>2.5'])
            cbar.set_label('Ploidy', fontsize=9)
        else:
            cbar.set_label('Ploidy', fontsize=9)
    
    # =========================================================================
    # BRANCH: SEPARATE vs COMBINED FIGURES
    # =========================================================================
    
    if plot_separate_clones:
        # =====================================================================
        # MODE 1: SEPARATE FIGURES PER CLONE
        # =====================================================================
        vprint("\n  Creating separate figures for each clone...")
        
        suffix = "_by_cytoband" if split_by_cytobands else "_by_chr_arm"
        
        for i, label in enumerate(unique_labels, 1):
            vprint(f"\n  Plotting clone {i}/{n_labels}: {label}")
            
            # Create single-panel figure
            fig, ax = plt.subplots(1, 1, figsize=(18, max_height_per_subplot))
            
            # Plot without ylabel (since we'll use title)
            plot_single_clone(ax, label, show_ylabel=False)
            
            # Add title with clone name
            n_cells = (df_selected[label_col] == label).sum()
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save with clone name in filename
            clone_filename = f"{output_prefix}{suffix}_{label.replace(' ', '_').replace('/', '_')}.png"
            output_path = os.path.join(output_dir, clone_filename)
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            fig.savefig(output_path.replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight')
            vprint(f"    ✓ Saved: {clone_filename}")
            plt.show()
            plt.close(fig)
    
    else:
        # =====================================================================
        # MODE 2: COMBINED FIGURE (ORIGINAL BEHAVIOR)
        # =====================================================================
        
        # Calculate figure dimensions
        height_per_subplot = min(max_height_per_subplot, max_total_height / n_labels)
        total_height = height_per_subplot * n_labels
        
        # Split into multiple figures if needed
        if n_labels > max_subplots_per_figure:
            vprint(f"\n  ⚠️  Too many labels ({n_labels}). Splitting into multiple figures...")
            n_figures = int(np.ceil(n_labels / max_subplots_per_figure))
            
            for fig_idx in range(n_figures):
                start_idx = fig_idx * max_subplots_per_figure
                end_idx = min((fig_idx + 1) * max_subplots_per_figure, n_labels)
                labels_subset = unique_labels[start_idx:end_idx]
                n_subplots = len(labels_subset)
                
                vprint(f"\n  Creating figure {fig_idx + 1}/{n_figures} with {n_subplots} subplots...")
                
                fig, axes = plt.subplots(
                    n_subplots, 1,
                    figsize=(18, min(8 * n_subplots, max_total_height)),
                    sharex=True
                )
                
                if n_subplots == 1:
                    axes = [axes]
                
                for ax, label in zip(axes, labels_subset):
                    plot_single_clone(ax, label, show_ylabel=True)
                
                plt.tight_layout()
                
                suffix = "_by_cytoband" if split_by_cytobands else "_by_chr_arm"
                output_path = os.path.join(output_dir, f"{output_prefix}{suffix}_part{fig_idx + 1}.png")
                fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
                fig.savefig(output_path.replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight')
                vprint(f"    ✓ Saved: {os.path.basename(output_path)}")
                plt.show()
                plt.close(fig)
        
        else:
            # Single figure
            vprint(f"\n  Creating single figure with {n_labels} subplots...")
            
            fig, axes = plt.subplots(
                n_labels, 1,
                figsize=(18, min(total_height, max_total_height)),
                sharex=True
            )
            
            if n_labels == 1:
                axes = [axes]
            
            for ax, label in zip(axes, unique_labels):
                plot_single_clone(ax, label, show_ylabel=True)
            
            plt.tight_layout()
            
            suffix = "_by_cytoband" if split_by_cytobands else "_by_chr_arm"
            output_path = os.path.join(output_dir, f"{output_prefix}{suffix}.png")
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            fig.savefig(output_path.replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight')
            vprint(f"    ✓ Saved: {os.path.basename(output_path)}")
            plt.show()
            plt.close(fig)
    
    vprint("\n" + "="*60)
    vprint("VISUALIZATION COMPLETE")
    vprint("="*60)

def visualise_binary_variants_ploidy_by_clone(
    df: pd.DataFrame,
    wilcoxon_results_cnv: dict,
    cytobands: pd.DataFrame,
    output_dir: str,
    output_prefix: str = "Single_cell_clones_CNV_Informative_by_chr_arm",
    label_col: str = 'label',
    max_height_per_subplot: float = 8.0,
    max_total_height: float = 200.0,
    max_subplots_per_figure: int = 25,
    dpi: int = 150,
    split_by_cytobands: bool = False,
    desired_clone_plotting_order: list = None,
    plot_separate_clones: bool = False,  # NEW PARAMETER
    font_title: int = 14,  # NEW: Font size for clone name title
    verbose: bool = True
) -> None:
    """
    Visualize CNV variants by informativeness (binary: informative vs non-informative).
    
    Uses Wilcoxon test results to determine which variants are informative for each clone.
    Informative variants are shown in BLACK, non-informative in WHITE.
    
    Parameters
    ----------
    df : pd.DataFrame
        CNV data with cells as rows, CNV variants as columns, and 'label' column
    wilcoxon_results_cnv : dict
        Dictionary from test_cnv_informative_variants with chromosome arms as keys,
        DataFrames with 'variant', 'label', 'informative' columns as values
    cytobands : pd.DataFrame
        Cytoband reference with columns: chromosome, arm, start, end, cytoband
    output_dir : str
        Directory to save output figures
    output_prefix : str
        Prefix for output filenames
    label_col : str
        Name of column containing clone labels
    max_height_per_subplot : float
        Maximum height (inches) per subplot
    max_total_height : float
        Maximum total figure height (inches)
    max_subplots_per_figure : int
        Maximum subplots per figure (splits if exceeded)
    dpi : int
        Resolution for saved figures
    split_by_cytobands : bool
        If True, adds vertical lines separating cytobands
        If False, only shows chromosome arm boundaries
    desired_clone_plotting_order : list, optional
        Custom order for clone labels
    plot_separate_clones : bool
        If True, creates separate PNG for each clone with clone name as title.
        If False, creates single figure with all clones stacked (default behavior)
    font_title : int
        Font size for clone name title (only used when plot_separate_clones=True)
    verbose : bool
        Print progress messages
        
    Returns
    -------
    None
        Saves figure(s) to output_dir
    """
    
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage, leaves_list
    
    def vprint(msg):
        if verbose:
            print(msg)
            import sys
            sys.stdout.flush()
    
    def extract_chr_arm_pos(col):
        """Extract chromosome number, arm, and position for sorting"""
        chr_arm = variant_to_arm.get(col, 'chr0p')
        try:
            chr_num = 23 if 'chrX' in chr_arm else int(chr_arm[3:-1])
        except Exception:
            chr_num = 0
        arm = chr_arm[-1]
        pos = int(col.split(':')[1])
        return (chr_num, arm, pos)
    
    def get_cytoband_for_variant(col):
        """Get cytoband name for a variant"""
        return variant_to_cytoband.get(col, '')
    
    vprint("\n" + "="*60)
    vprint("VISUALIZING CNV INFORMATIVENESS BY CLONE")
    vprint("="*60)
    
    if plot_separate_clones:
        vprint("MODE: Separate figures per clone (clone name as title)")
    else:
        vprint("MODE: Single figure with all clones stacked")
    
    vprint("  Color scheme: BINARY (non-informative=WHITE, informative=BLACK)")
    
    if split_by_cytobands:
        vprint("  Boundaries: CYTOBANDS (individual cytobands separated)")
    else:
        vprint("  Boundaries: CHROMOSOME ARMS (p/q arms separated)")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Build informative variant lookup: (clone_label, variant) -> is_informative
    vprint("\n  Building informative variant lookup from Wilcoxon results...")
    
    informative_lookup = {}
    total_comparisons = 0
    total_informative = 0
    
    for chr_arm, results_df in wilcoxon_results_cnv.items():
        for _, row in results_df.iterrows():
            variant = row['variant']
            label = row['label']
            is_informative = row['informative']
            
            informative_lookup[(label, variant)] = is_informative
            total_comparisons += 1
            if is_informative:
                total_informative += 1
    
    vprint(f"    ✓ Loaded {total_comparisons} variant-clone comparisons")
    vprint(f"    ✓ Informative: {total_informative} ({100*total_informative/total_comparisons:.1f}%)")
    
    # Get CNV variants
    cnv_variants = [col for col in df.columns if col.startswith('chr')]
    
    vprint(f"\n  CNV variants in data: {len(cnv_variants)}")
    
    # Select CNV variants and label column
    df_selected = df[cnv_variants + [label_col]].copy()
    
    vprint(f"  Data shape: {df_selected.shape[0]} cells × {len(cnv_variants)} variants")
    
    # Build chromosome arm lookup
    vprint("\n  Building chromosome arm lookup...")
    
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
    
    vprint(f"    ✓ Lookup table built for {len(arm_lookup)} chromosomes")
    
    # Assign variants to chromosome arms AND cytobands
    vprint(f"\n  Assigning variants to chromosome arms{' and cytobands' if split_by_cytobands else ''}...")
    
    variant_to_arm = {}
    variant_to_cytoband = {}
    
    for col in cnv_variants:
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
    
    vprint(f"    ✓ Assigned {len(variant_to_arm)} variants to chromosome arms")
    if split_by_cytobands:
        n_cytobands = len(set(variant_to_cytoband.values()))
        vprint(f"    ✓ Assigned variants to {n_cytobands} unique cytobands")
    
    # Sort columns by chromosome and position
    sorted_cols = sorted(cnv_variants, key=extract_chr_arm_pos)
    df_selected = df_selected[sorted_cols + [label_col]]
    
    # Create boundary list
    if split_by_cytobands:
        boundary_list = pd.Index(sorted_cols).map(get_cytoband_for_variant)
    else:
        boundary_list = pd.Index(sorted_cols).map(lambda col: variant_to_arm.get(col, ''))
    
    # Get unique labels and apply custom ordering
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
    
    vprint(f"\n  Clone labels to plot: {n_labels} (including WT)")
    vprint(f"  Final plotting order:")
    for idx, label in enumerate(unique_labels, 1):
        n_cells = (df_selected[label_col] == label).sum()
        
        # Count informative variants for this clone
        n_informative = sum(1 for v in sorted_cols if informative_lookup.get((label, v), False))
        
        # Special message for WT
        if label == 'WT':
            vprint(f"    {idx}. {label} (n={n_cells} cells, BASELINE)")
        else:
            vprint(f"    {idx}. {label} (n={n_cells} cells, {n_informative} informative variants)")
    
    # Setup colormap: white (0) vs black (1)
    colors_list = ['white', 'black']
    cmap = mcolors.ListedColormap(colors_list)
    bounds = [0, 0.5, 1]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    vmin, vmax = 0, 1
    
    # =========================================================================
    # HELPER FUNCTION: PLOT SINGLE CLONE
    # =========================================================================
    
    def plot_single_clone(ax, label, show_ylabel=True):
        """Plot heatmap for a single clone"""
        df_label = df_selected[df_selected[label_col] == label].drop(columns=label_col)
        
        if len(df_label) == 0:
            vprint(f"    Skipping {label}: no cells")
            ax.text(0.5, 0.5, f'{label}\n(No cells)',
                   ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        # Cluster cells
        try:
            linkage_matrix = linkage(df_label, method='average')
            ordered_indices = leaves_list(linkage_matrix)
            df_label = df_label.iloc[ordered_indices, :]
        except Exception as e:
            vprint(f"    Clustering failed for {label}: {e}")
        
        # Create binary informativeness matrix
        plot_data = np.zeros_like(df_label.values)
        
        for col_idx, variant in enumerate(df_label.columns):
            is_informative = informative_lookup.get((label, variant), False)
            if is_informative:
                plot_data[:, col_idx] = 1  # Black
            # else: stays 0 (white)
        
        plot_data = pd.DataFrame(
            plot_data,
            index=df_label.index,
            columns=df_label.columns
        )
        
        # Plot heatmap
        sns.heatmap(plot_data, cbar=True, ax=ax, yticklabels=False,
                   cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
        
        # Add boundaries
        prev_boundary = None
        for j, current_boundary in enumerate(boundary_list):
            if current_boundary != prev_boundary:
                line_width = 4 if split_by_cytobands else 2
                line_color = 'red'  # Use red for visibility
                ax.axvline(x=j, color=line_color, linestyle='--', linewidth=line_width)
                
                if j > 0:
                    fontsize = 6 if split_by_cytobands else 8
                    ax.text(j, ax.get_ylim()[1] - 2, current_boundary, 
                           ha='center', va='bottom',
                           color='black', fontsize=fontsize, fontweight='bold',
                           rotation=90)
                prev_boundary = current_boundary
        
        # Count informative variants
        n_informative = sum(1 for v in df_label.columns if informative_lookup.get((label, v), False))
        
        # Style
        ax.set_yticklabels([])
        ax.tick_params(axis='x', labelsize=8, labelrotation=90)
        
        if show_ylabel:
            # Style with different label for WT
            if label == 'WT':
                ylabel = f'{label}\n(n={len(df_label)})\nBASELINE'
            else:
                ylabel = f'{label}\n(n={len(df_label)})\n{n_informative} informative'
            
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', rotation=90, labelpad=15)
        else:
            ax.set_ylabel('')
        
        # Adjust colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=9)
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(['Not Informative', 'Informative'])
        cbar.set_label('CNV Informativeness', fontsize=9)
    
    # =========================================================================
    # BRANCH: SEPARATE vs COMBINED FIGURES
    # =========================================================================
    
    if plot_separate_clones:
        # =====================================================================
        # MODE 1: SEPARATE FIGURES PER CLONE
        # =====================================================================
        vprint("\n  Creating separate figures for each clone...")
        
        suffix = "_informative_by_cytoband" if split_by_cytobands else "_informative_by_chr_arm"
        
        for i, label in enumerate(unique_labels, 1):
            vprint(f"\n  Plotting clone {i}/{n_labels}: {label}")
            
            # Create single-panel figure
            fig, ax = plt.subplots(1, 1, figsize=(18, max_height_per_subplot))
            
            # Plot without ylabel (since we'll use title)
            plot_single_clone(ax, label, show_ylabel=False)
            
            # Add title with clone name
            n_cells = (df_selected[label_col] == label).sum()
            n_informative = sum(1 for v in sorted_cols if informative_lookup.get((label, v), False))
            
            if label == 'WT':
                title_text = f'{label} (n={n_cells}) - BASELINE'
            else:
                title_text = f'{label} (n={n_cells}) - {n_informative} informative variants'
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save with clone name in filename
            clone_filename = f"{output_prefix}{suffix}_{label.replace(' ', '_').replace('/', '_')}.png"
            output_path = os.path.join(output_dir, clone_filename)
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            fig.savefig(output_path.replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight')
            vprint(f"    ✓ Saved: {clone_filename}")
            plt.show()
            plt.close(fig)
    
    else:
        # =====================================================================
        # MODE 2: COMBINED FIGURE (ORIGINAL BEHAVIOR)
        # =====================================================================
        
        # Calculate figure dimensions
        height_per_subplot = min(max_height_per_subplot, max_total_height / n_labels)
        total_height = height_per_subplot * n_labels
        
        # Split into multiple figures if needed
        if n_labels > max_subplots_per_figure:
            vprint(f"\n  ⚠️  Too many labels ({n_labels}). Splitting into multiple figures...")
            n_figures = int(np.ceil(n_labels / max_subplots_per_figure))
            
            for fig_idx in range(n_figures):
                start_idx = fig_idx * max_subplots_per_figure
                end_idx = min((fig_idx + 1) * max_subplots_per_figure, n_labels)
                labels_subset = unique_labels[start_idx:end_idx]
                n_subplots = len(labels_subset)
                
                vprint(f"\n  Creating figure {fig_idx + 1}/{n_figures} with {n_subplots} subplots...")
                
                fig, axes = plt.subplots(
                    n_subplots, 1,
                    figsize=(18, min(8 * n_subplots, max_total_height)),
                    sharex=True
                )
                
                if n_subplots == 1:
                    axes = [axes]
                
                for ax, label in zip(axes, labels_subset):
                    plot_single_clone(ax, label, show_ylabel=True)
                
                plt.tight_layout()
                
                suffix = "_informative_by_cytoband" if split_by_cytobands else "_informative_by_chr_arm"
                output_path = os.path.join(output_dir, f"{output_prefix}{suffix}_part{fig_idx + 1}.png")
                fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
                fig.savefig(output_path.replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight')
                vprint(f"    ✓ Saved: {os.path.basename(output_path)}")
                plt.show()
                plt.close(fig)
        
        else:
            # Single figure
            vprint(f"\n  Creating single figure with {n_labels} subplots...")
            
            fig, axes = plt.subplots(
                n_labels, 1,
                figsize=(18, min(total_height, max_total_height)),
                sharex=True
            )
            
            if n_labels == 1:
                axes = [axes]
            
            for ax, label in zip(axes, unique_labels):
                plot_single_clone(ax, label, show_ylabel=True)
            
            plt.tight_layout()
            
            suffix = "_informative_by_cytoband" if split_by_cytobands else "_informative_by_chr_arm"
            output_path = os.path.join(output_dir, f"{output_prefix}{suffix}.png")
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            fig.savefig(output_path.replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight')
            vprint(f"    ✓ Saved: {os.path.basename(output_path)}")
            plt.show()
            plt.close(fig)
    
    vprint("\n" + "="*60)
    vprint("VISUALIZATION COMPLETE")
    vprint("="*60)

def visualise_variants_ploidy_by_clone_grouped(
    df_cnv_grouped: pd.DataFrame,
    cnv_info: pd.DataFrame,
    output_path: str,
    cytobands: pd.DataFrame = None,
    desired_clone_plotting_order: List[str] = None,
    plot_categorical: bool = False,
    split_by_cytobands: bool = False,
    figsize: tuple = (20, 6),
    dpi: int = 300,
    verbose: bool = True
) -> None:
    """
    Plot grouped CNV ploidy (amplicon × clone) as a single heatmap.
    - Rows: clones (ordered by desired_clone_plotting_order if provided)
    - Columns: amplicons (genomic order using cnv_info)
    - Palette: same strategy as visualise_variants_ploidy_by_clone
        * Continuous: RdYlBu_r (vmin=0, vmax=4, center=2)
        * Categorical: <1.5=dark blue, 1.5-2.5=white, >2.5=dark red
    - Adds vertical chromosome separators and headers.

    Parameters
    ----------
    df_cnv_grouped : pd.DataFrame
        Amplicon (index) × clone (columns) ploidy medians.
    cnv_info : pd.DataFrame
        Amplicon metadata with at least ['CHROM', 'start_pos', 'id'].
    output_path : str
        Path to save the figure.
    cytobands : pd.DataFrame, optional
        Cytoband reference with columns: chromosome, arm, start, end, cytoband.
        Required if split_by_cytobands=True.
    desired_clone_plotting_order : List[str], optional
        Preferred clone ordering; others appended after.
    plot_categorical : bool
        If True, uses categorical coloring; else continuous ploidy.
    split_by_cytobands : bool
        If True, adds vertical lines separating cytobands (more granular).
        If False, only shows chromosome arm boundaries.
        Requires cytobands parameter.
    figsize : tuple
        Figure size.
    dpi : int
        Output resolution.
    verbose : bool
        Verbose logging.
    """
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    def vprint(msg):
        if verbose:
            print(msg)

    def chrom_key(series: pd.Series):
        """Sort key for chromosome names like '1','2',...,'X','Y'"""
        def chrom_to_int(chrom):
            c = str(chrom).upper().replace('CHR', '')
            if c == 'X': return 23
            if c == 'Y': return 24
            if c in ('M', 'MT'): return 25
            try:
                return int(c)
            except ValueError:
                return 999
        return series.map(chrom_to_int)

    vprint("\n" + "="*60)
    vprint("VISUALIZING GROUPED CNV PLOIDY")
    vprint("="*60)
    
    if plot_categorical:
        vprint("  Color scheme: CATEGORICAL (<1.5=dark blue, 1.5-2.5=white, >2.5=dark red)")
    else:
        vprint("  Color scheme: CONTINUOUS (RdYlBu_r, 0-4, center at 2)")
    
    if split_by_cytobands:
        vprint("  Boundaries: CYTOBANDS (individual cytobands separated)")
        if cytobands is None:
            vprint("  ❌ ERROR: split_by_cytobands=True requires cytobands parameter")
            return
    else:
        vprint("  Boundaries: CHROMOSOME ARMS (chr separators only)")

    # Validate cnv_info
    required = {'CHROM', 'start_pos', 'id'}
    missing = required - set(cnv_info.columns)
    if missing:
        raise KeyError(f"cnv_info missing required columns: {missing}")

    # Sort amplicons by genomic position
    cnv_info_sorted = cnv_info.copy()
    cnv_info_sorted['start_pos'] = pd.to_numeric(cnv_info_sorted['start_pos'], errors='coerce')
    cnv_info_sorted = cnv_info_sorted.sort_values(by=['CHROM', 'start_pos'], key=chrom_key)

    # Align df_cnv_grouped rows to genomic order
    ordered_amplicons = [aid for aid in cnv_info_sorted['id'] if aid in df_cnv_grouped.index]
    
    if len(ordered_amplicons) == 0:
        vprint("  ⚠️  No amplicons to plot")
        return
    
    df_cnv_grouped = df_cnv_grouped.reindex(ordered_amplicons)
    vprint(f"\n  Amplicons to plot: {len(ordered_amplicons)}")

    # Build clone ordering
    if desired_clone_plotting_order:
        ordered_clones = [c for c in desired_clone_plotting_order if c in df_cnv_grouped.columns]
        other_clones = [c for c in df_cnv_grouped.columns if c not in ordered_clones]
        df_plot = df_cnv_grouped[ordered_clones + sorted(other_clones)]
        vprint(f"  Clones (ordered): {len(ordered_clones)}")
        if other_clones:
            vprint(f"  Clones (additional): {len(other_clones)}")
    else:
        df_plot = df_cnv_grouped
        vprint(f"  Clones: {len(df_plot.columns)}")

    # Prepare data for heatmap
    if plot_categorical:
        # Categorical ploidy: <1.5=loss(0), 1.5-2.5=neutral(1), >2.5=gain(2)
        vals = df_plot.apply(pd.to_numeric, errors='coerce')
        cats = np.zeros_like(vals.values, dtype=int)
        cats[vals.values < 1.5] = 0
        cats[(vals.values >= 1.5) & (vals.values <= 2.5)] = 1
        cats[vals.values > 2.5] = 2
        cats_df = pd.DataFrame(cats, index=vals.index, columns=vals.columns)

        cmap = mcolors.ListedColormap(['#00008B', 'white', '#8B0000'])
        norm = mcolors.BoundaryNorm([0, 1, 2, 3], cmap.N)
        data_to_plot = cats_df.T
        heatmap_kwargs = dict(cmap=cmap, norm=norm, vmin=0, vmax=2)
        cbar_label = 'Ploidy (categorical)'
    else:
        # Continuous ploidy (clip upper to 4)
        vals = df_plot.apply(pd.to_numeric, errors='coerce').clip(upper=4.0)
        data_to_plot = vals.T
        heatmap_kwargs = dict(cmap='RdYlBu_r', vmin=0, vmax=4, center=2)
        cbar_label = 'Ploidy'

    vprint(f"\n  Plotting {data_to_plot.shape[0]} clones × {data_to_plot.shape[1]} amplicons")

    # Plot
    plt.figure(figsize=figsize)
    sns.set(font_scale=0.9)

    ax = sns.heatmap(
        data_to_plot,
        cbar=True,
        xticklabels=False,
        **heatmap_kwargs
    )

    # Build boundary list based on mode
    if split_by_cytobands and cytobands is not None:
        # Map amplicons to cytobands for more granular separators
        vprint("\n  Mapping amplicons to cytobands...")
        
        amplicon_to_cytoband = {}
        for amplicon_id in ordered_amplicons:
            # Get chromosome and position for this amplicon
            amplicon_info = cnv_info_sorted[cnv_info_sorted['id'] == amplicon_id]
            if len(amplicon_info) == 0:
                continue
            
            chrom = str(amplicon_info.iloc[0]['CHROM'])
            pos = amplicon_info.iloc[0]['start_pos']
            
            # Find matching cytoband
            matching_bands = cytobands[
                (cytobands['chromosome'].astype(str) == chrom) &
                (cytobands['start'] <= pos) &
                (cytobands['end'] >= pos)
            ]
            
            if len(matching_bands) > 0:
                cytoband_name = matching_bands.iloc[0]['cytoband']
                amplicon_to_cytoband[amplicon_id] = f"chr{chrom}{cytoband_name}"
            else:
                # Fallback to chromosome arm
                arm_info = cytobands[cytobands['chromosome'].astype(str) == chrom]
                if len(arm_info) > 0:
                    if pos < arm_info['start'].median():
                        amplicon_to_cytoband[amplicon_id] = f"chr{chrom}p"
                    else:
                        amplicon_to_cytoband[amplicon_id] = f"chr{chrom}q"
                else:
                    amplicon_to_cytoband[amplicon_id] = f"chr{chrom}"
        
        boundary_list = [amplicon_to_cytoband.get(aid, '') for aid in ordered_amplicons]
        vprint(f"    ✓ Mapped to {len(set(boundary_list))} unique cytobands")
    else:
        # Build chromosome labels for separators (columns = amplicons)
        boundary_list = ['chr' + str(cnv_info_sorted[cnv_info_sorted['id'] == aid]['CHROM'].iloc[0]) 
                        if aid in cnv_info_sorted['id'].values else '' 
                        for aid in ordered_amplicons]

    # Add separators and labels
    prev_boundary = None
    for j, current_boundary in enumerate(boundary_list):
        if current_boundary != prev_boundary:
            line_width = 4 if split_by_cytobands else 2
            line_color = 'black' if plot_categorical else 'grey'
            
            ax.axvline(x=j, color=line_color, linestyle='--', linewidth=line_width)
            
            # Add label (skip empty labels)
            if current_boundary and j > 0:
                fontsize = 6 if split_by_cytobands else 8
                ax.text(
                    j, ax.get_ylim()[1] - 0.2, current_boundary,
                    ha='center', va='bottom',
                    color='black', fontsize=fontsize, fontweight='bold', rotation=90
                )
            prev_boundary = current_boundary

    # Y-axis styling (clone labels)
    ax.tick_params(axis='y', labelsize=8, labelrotation=0)
    for label in ax.get_yticklabels():
        label.set_weight('bold')

    # Colorbar styling
    if ax.collections:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(cbar_label, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=dpi)
    plt.show()
    plt.close()

    vprint(f"\n✓ Saved: {output_path}")
    vprint("="*60 + "\n")

def plot_integrated_cnv_baf_with_significance(
    df_AF: "pd.DataFrame",
    df_cnv_vars: "pd.DataFrame",
    df_ALT: "pd.DataFrame",
    df_REF: "pd.DataFrame",
    cytobands: "pd.DataFrame",
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
    df_cnv_all: "pd.DataFrame" = None,
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

    Ploidy-line behavior
    --------------------
    If plot_ploidy_median=True:

    1. For each cytoband, all CNV amplicons in that cytoband are identified.
    2. For the current clone, each amplicon gets one mean ploidy value across cells.
    3. Cytobands are classified as:
       - "extreme" if wilcoxon_results_var marks at least one displayed variant
         in that cytoband as informative=True for the clone.
       - "neutral" otherwise.
    4. Adjacent cytobands are merged only if they are:
       - on the same chromosome, and
       - have the same Wilcoxon-derived state.
    5. The final ploidy line for each contiguous merged block is calculated from
       all amplicon means in the block.

    Important
    ---------
    Extreme/no-extreme status is NOT recomputed from ploidy values.
    wilcoxon_results_var is the source of truth for merging only.

    Ploidy lines are NOT colored differently by extreme/neutral state.
    All final ploidy lines use ploidy_line_color.

    CI behavior
    -----------
    If plot_ploidy_ci=True, a grey confidence interval band is drawn behind each
    final ploidy line. If plot_ploidy_ci_caps=True, dashed horizontal cap lines
    are drawn at the lower and upper CI limits.
    """

    import os
    import re
    import sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    def vprint(msg):
        if verbose:
            print(msg)
            sys.stdout.flush()

    # =========================================================================
    # Helpers
    # =========================================================================
    def extract_chr_arm_pos(col, variant_to_arm):
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
        match = re.match(r"^(chr(?:\d+|X|Y))", str(chr_arm))
        return match.group(1) if match else str(chr_arm)

    def _summarise_values(values, stat="mean"):
        values = pd.Series(values).dropna().astype(float)

        if len(values) == 0:
            return np.nan

        if stat == "mean":
            return float(values.mean())

        if stat == "median":
            return float(values.median())

        raise ValueError("ploidy_line_stat must be either 'mean' or 'median'.")

    def _summarise_interval(values, method="percentile", percentiles=(25, 75)):
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

    def _draw_ci_band_and_caps(ax, x_start, x_end, ci_lo, ci_hi):
        if pd.isna(ci_lo) or pd.isna(ci_hi):
            return

        ax.fill_between(
            [x_start, x_end],
            [ci_lo, ci_lo],
            [ci_hi, ci_hi],
            color=ploidy_ci_color,
            alpha=ploidy_ci_alpha,
            linewidth=0,
            zorder=3.1,
        )

        if plot_ploidy_ci_caps:
            ax.hlines(
                y=ci_lo,
                xmin=x_start,
                xmax=x_end,
                colors=ploidy_ci_cap_color,
                linewidth=ploidy_ci_cap_linewidth,
                linestyles=ploidy_ci_cap_linestyle,
                alpha=ploidy_ci_cap_alpha,
                zorder=3.25,
            )
            ax.hlines(
                y=ci_hi,
                xmin=x_start,
                xmax=x_end,
                colors=ploidy_ci_cap_color,
                linewidth=ploidy_ci_cap_linewidth,
                linestyles=ploidy_ci_cap_linestyle,
                alpha=ploidy_ci_cap_alpha,
                zorder=3.25,
            )

    def _safe_label_for_filename(label):
        return (
            str(label)
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )

    baf_points = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    baf_colors_hex = ["#FFFFFF", "#dfe7fd", "#5e548e", "#eae4e9", "#FFFFFF"]
    baf_colors_rgb = np.array([mcolors.to_rgb(h) for h in baf_colors_hex])

    def baf_to_rgb(vals):
        v = np.clip(np.asarray(vals, dtype=float), 0.0, 1.0)
        r = np.interp(v, baf_points, baf_colors_rgb[:, 0])
        g = np.interp(v, baf_points, baf_colors_rgb[:, 1])
        b = np.interp(v, baf_points, baf_colors_rgb[:, 2])
        return np.stack([r, g, b], axis=-1)

    def _add_top_chr_arm_labels(ax_baf, arm_blocks, rotation, fontsize, pad):
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
        if not cytoband_list:
            return

        centers = []
        labels = []
        s = 0

        for i in range(1, len(cytoband_list) + 1):
            is_new_block = (
                i == len(cytoband_list)
                or cytoband_list[i] != cytoband_list[s]
                or chr_arm_list[i] != chr_arm_list[s]
            )

            if is_new_block:
                lab = cytoband_list[s]

                if lab not in ("", "NA", None):
                    centers.append((s + i - 1) / 2.0)
                    labels.append(lab)

                s = i

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

    def _draw_cytoband_separators(ax):
        for x in cytoband_sep_x:
            ax.axvline(
                x=x,
                color="#c0c0c0",
                linestyle="--",
                linewidth=cytoband_linewidth,
                alpha=cytoband_alpha,
                zorder=0.55,
            )

    def _draw_chr_arm_boundaries(ax):
        for x in chr_arm_sep_x:
            ax.axvline(
                x=x,
                color="black",
                linestyle="-",
                linewidth=chr_arm_linewidth,
                alpha=chr_arm_alpha,
                zorder=0.95,
            )

    def _draw_alt_arm_background(ax):
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

    default_type_to_color = {
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

    type_to_color = dict(default_type_to_color)

    if isinstance(cytoband_type_to_color, dict):
        type_to_color.update(cytoband_type_to_color)

    # =========================================================================
    # Logging
    # =========================================================================
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
    # AF variants
    # =========================================================================
    all_germline = [
        c for c in df_AF.columns
        if isinstance(c, str) and c.startswith("chr")
    ]

    if len(all_germline) == 0:
        vprint("❌ No variants found.")
        return

    # =========================================================================
    # Cytoband lookup
    # =========================================================================
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

    def variant_to_chr_arm(var: str) -> str:
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

    variant_to_arm = {v: variant_to_chr_arm(v) for v in all_germline}

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
    df_cnv_selected.loc[:, cnv_numeric.columns] = cnv_numeric.clip(upper=4.0)

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
        key=lambda x: extract_chr_arm_pos(x, variant_to_arm),
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

    chr_arm_list = [variant_to_arm.get(col, "") for col in sorted_cols]

    arm_blocks = []

    if chr_arm_list:
        s = 0

        for j in range(1, len(chr_arm_list) + 1):
            if j == len(chr_arm_list) or chr_arm_list[j] != chr_arm_list[s]:
                arm_blocks.append((chr_arm_list[s], s, j - 1))
                s = j

    chr_arm_sep_x = [
        i - 0.5
        for i in range(1, len(chr_arm_list))
        if chr_arm_list[i] != chr_arm_list[i - 1]
    ]

    def _band_and_type(chr_arm: str, pos: int):
        for s0, e0, b0, t0 in cb_intervals.get(str(chr_arm), []):
            if s0 <= pos <= e0:
                return b0, t0

        return "NA", "NA"

    cytoband_list = []
    cytoband_type_list = []

    for v, arm_name in zip(sorted_cols, chr_arm_list):
        try:
            pos = int(str(v).split(":")[1])
        except Exception:
            pos = -1

        b, t = _band_and_type(arm_name, pos)

        cytoband_list.append(b)
        cytoband_type_list.append(t)

    cytoband_sep_x = [
        i - 0.5
        for i in range(1, len(cytoband_list))
        if chr_arm_list[i] == chr_arm_list[i - 1]
        and cytoband_list[i] != cytoband_list[i - 1]
    ]

    # =========================================================================
    # Precompute all-amplicon CNV data per cytoband
    # =========================================================================
    _all_cnv_source = df_cnv_all if df_cnv_all is not None else df_cnv_vars

    _all_cnv_cols = [
        c for c in _all_cnv_source.columns
        if isinstance(c, str) and c.startswith("chr") and c != "label"
    ]

    _cnv_col_to_band = {}

    for col in _all_cnv_cols:
        _arm = variant_to_chr_arm(col)

        if chr_arms_to_plot is not None and _arm not in chr_arms_to_plot:
            continue

        try:
            _pos = int(str(col).split(":")[1])
        except Exception:
            continue

        _b, _t = _band_and_type(_arm, _pos)
        _cnv_col_to_band[col] = (_arm, _b)

    _band_to_cnv_cols = {}

    for col, key in _cnv_col_to_band.items():
        _band_to_cnv_cols.setdefault(key, []).append(col)

    vprint(
        f"  Ploidy line source: {len(_all_cnv_cols)} total CNV amplicons "
        f"({len(_cnv_col_to_band)} mapped to plotted cytobands)"
    )

    # =========================================================================
    # Informative / extreme BAF lookup from wilcoxon_results_var
    # =========================================================================
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

    x_positions = np.arange(len(sorted_cols))
    arm_bg_colors = ["#eaeaea", "#ffffff"]
    somatic_set = set(differential_variants)
    somatic_mask = np.array([col in somatic_set for col in sorted_cols], dtype=bool)

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

        low_dp_mask = (dp_mean_raw_vals < 5.0) | np.isnan(dp_mean_raw_vals)

        # ---------------------------------------------------------------------
        # BAF panel
        # ---------------------------------------------------------------------
        ax = ax_baf
        ax.set_facecolor("white")

        for name, spine in ax.spines.items():
            if name in ("left", "bottom", "top"):
                spine.set_visible(True)
                spine.set_linewidth(1.0)
                spine.set_edgecolor("black")
                spine.set_zorder(10)
            else:
                spine.set_visible(False)

        _draw_alt_arm_background(ax)

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

        informative_baf_mask = np.zeros(len(sorted_cols), dtype=bool)
        mean_colors = np.full(len(sorted_cols), "dimgray", dtype=object)

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

        jitter_width = 0.18

        for j, var in enumerate(sorted_cols):
            col_vals = df_label_num[var].values / 100.0
            col_vals = col_vals[~np.isnan(col_vals)]

            if col_vals.size == 0:
                continue

            if col_vals.size > max_jitter_cells:
                col_vals = col_vals[
                    np.random.choice(
                        col_vals.size,
                        max_jitter_cells,
                        replace=False,
                    )
                ]

            ax.scatter(
                j + np.random.uniform(
                    -jitter_width,
                    jitter_width,
                    size=col_vals.size,
                ),
                col_vals,
                c=baf_to_rgb(col_vals),
                s=8,
                alpha=0.1,
                edgecolors="none",
                zorder=1,
            )

        circle_big = (~somatic_mask) & informative_baf_mask
        circle_small = (~somatic_mask) & (~informative_baf_mask)

        ax.scatter(
            x_positions[circle_small & ~low_dp_mask],
            vals[circle_small & ~low_dp_mask],
            c=mean_colors_arr[circle_small & ~low_dp_mask],
            s=24,
            alpha=1.0,
            marker="o",
            edgecolors="black",
            linewidths=0.3,
            zorder=2,
        )

        ax.scatter(
            x_positions[circle_small & low_dp_mask],
            vals[circle_small & low_dp_mask],
            c=mean_colors_arr[circle_small & low_dp_mask],
            s=24,
            alpha=0.4,
            marker="o",
            edgecolors="black",
            linewidths=0.3,
            zorder=2,
        )

        ax.scatter(
            x_positions[circle_big & ~low_dp_mask],
            vals[circle_big & ~low_dp_mask],
            c=mean_colors_arr[circle_big & ~low_dp_mask],
            s=48,
            alpha=1.0,
            marker="o",
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )

        ax.scatter(
            x_positions[circle_big & low_dp_mask],
            vals[circle_big & low_dp_mask],
            c=mean_colors_arr[circle_big & low_dp_mask],
            s=48,
            alpha=0.4,
            marker="o",
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )

        ax.scatter(
            x_positions[somatic_mask & ~low_dp_mask],
            vals[somatic_mask & ~low_dp_mask],
            c=mean_colors_arr[somatic_mask & ~low_dp_mask],
            s=60,
            alpha=1.0,
            marker="D",
            edgecolors="black",
            linewidths=0.5,
            zorder=4,
        )

        ax.scatter(
            x_positions[somatic_mask & low_dp_mask],
            vals[somatic_mask & low_dp_mask],
            c=mean_colors_arr[somatic_mask & low_dp_mask],
            s=60,
            alpha=0.4,
            marker="D",
            edgecolors="black",
            linewidths=0.5,
            zorder=4,
        )

        ax.axhline(
            y=0.25,
            color="grey",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            zorder=0.6,
        )

        ax.axhline(
            y=0.75,
            color="grey",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            zorder=0.6,
        )

        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, len(sorted_cols) - 0.5)
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

        _draw_cytoband_separators(ax)
        _draw_chr_arm_boundaries(ax)

        # ---------------------------------------------------------------------
        # CNV panel
        # ---------------------------------------------------------------------
        ax2 = ax_cnv
        ax2.set_facecolor("white")

        for name, spine in ax2.spines.items():
            if name in ("left", "bottom", "top"):
                spine.set_visible(True)
                spine.set_linewidth(1.0)
                spine.set_edgecolor("black")
                spine.set_zorder(10)
            else:
                spine.set_visible(False)

        if color_cytoband_background_in_ploidy and cytoband_list:
            s = 0

            for i in range(1, len(cytoband_list) + 1):
                is_new_block = (
                    i == len(cytoband_list)
                    or cytoband_list[i] != cytoband_list[s]
                    or chr_arm_list[i] != chr_arm_list[s]
                )

                if is_new_block:
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

                    s = i
        else:
            _draw_alt_arm_background(ax2)

        df_cnv_label = (
            df_cnv_selected[df_cnv_selected["label"] == label]
            .drop(columns="label")
        )

        df_cnv_num = (
            df_cnv_label
            .apply(pd.to_numeric, errors="coerce")
            .reindex(columns=sorted_cols)
            .clip(upper=4.0)
        )

        show_cnv_points = not (
            plot_ploidy_median and hide_cnv_points_when_plotting_lines
        )

        # Single-cell CNV jitter points.
        if show_cnv_points:
            jitter_width_cnv = 0.18

            for j, var in enumerate(sorted_cols):
                if var not in df_cnv_num.columns:
                    continue

                col_vals = df_cnv_num[var].values
                col_vals = col_vals[~np.isnan(col_vals)]

                if col_vals.size == 0:
                    continue

                if col_vals.size > max_jitter_cells:
                    col_vals = col_vals[
                        np.random.choice(
                            col_vals.size,
                            max_jitter_cells,
                            replace=False,
                        )
                    ]

                ax2.scatter(
                    j + np.random.uniform(
                        -jitter_width_cnv,
                        jitter_width_cnv,
                        size=col_vals.size,
                    ),
                    col_vals,
                    s=8,
                    alpha=0.1,
                    color="#c9ada7",
                    edgecolors="none",
                    zorder=1,
                )

        # Per-amplicon / per-variant CNV median dots.
        if show_cnv_points:
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

        # ---------------------------------------------------------------------
        # Ploidy summary lines.
        # ---------------------------------------------------------------------
        if plot_ploidy_median:
            df_cnv_all_label = (
                _all_cnv_source[_all_cnv_source["label"] == label]
                .drop(columns="label", errors="ignore")
            )

            df_cnv_all_num = (
                df_cnv_all_label
                .apply(pd.to_numeric, errors="coerce")
                .clip(upper=4.0)
            )

            label_extreme_variants = {
                str(var)
                for (clone_label, var), is_info in informative_lookup.items()
                if str(clone_label).strip() == label_clean and bool(is_info)
            }

            cytoband_segments = []

            s = 0

            for i in range(1, len(cytoband_list) + 1):
                is_new_segment = (
                    i == len(cytoband_list)
                    or cytoband_list[i] != cytoband_list[s]
                    or chr_arm_list[i] != chr_arm_list[s]
                )

                if not is_new_segment:
                    continue

                chr_arm = chr_arm_list[s]
                cytoband_name = cytoband_list[s]
                chromosome = _chr_from_arm(chr_arm)

                band_key = (chr_arm, cytoband_name)
                band_cnv_cols = _band_to_cnv_cols.get(band_key, [])

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

                s = i

            if merge_contiguous_ploidy_lines:
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

                for seg in merged_segments:
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
                            ax=ax2,
                            x_start=seg["x_start"],
                            x_end=seg["x_end"],
                            ci_lo=ci_lo,
                            ci_hi=ci_hi,
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

            else:
                for seg in cytoband_segments:
                    if len(seg["amplicon_values"]) == 0:
                        continue

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
                            ax=ax2,
                            x_start=seg["x_start"],
                            x_end=seg["x_end"],
                            ci_lo=ci_lo,
                            ci_hi=ci_hi,
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

        _draw_cytoband_separators(ax2)
        _draw_chr_arm_boundaries(ax2)

        ax2.axhline(
            y=1.0,
            color="grey",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            zorder=0.6,
        )

        ax2.axhline(
            y=3.0,
            color="grey",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            zorder=0.6,
        )

        ax2.set_xlim(-0.5, len(sorted_cols) - 0.5)
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
        # ALT panel
        # ---------------------------------------------------------------------
        if plot_alt_ref and ax_alt is not None:
            ax3 = ax_alt

            for name, spine in ax3.spines.items():
                if name in ("left", "bottom", "top"):
                    spine.set_visible(True)
                    spine.set_linewidth(1.0)
                    spine.set_edgecolor("black")
                    spine.set_zorder(10)
                else:
                    spine.set_visible(False)

            ax3.set_facecolor("#f8eedb")

            alt_vals = (
                np.log10(df_alt_num + 1.0)
                .mean(axis=0, numeric_only=True)
                .values
                .astype(float)
            )

            good_alt = (~low_dp_mask) & (~np.isnan(alt_vals))
            low_alt = low_dp_mask & (~np.isnan(alt_vals))

            ax3.scatter(
                x_positions[good_alt],
                alt_vals[good_alt],
                c="#8b4513",
                s=20,
                alpha=1.0,
                edgecolors="black",
                linewidths=0.3,
                zorder=2,
            )

            ax3.scatter(
                x_positions[low_alt],
                alt_vals[low_alt],
                c="#8b4513",
                s=20,
                alpha=0.4,
                edgecolors="black",
                linewidths=0.3,
                zorder=2,
            )

            _draw_cytoband_separators(ax3)
            _draw_chr_arm_boundaries(ax3)

            ax3.set_xlim(-0.5, len(sorted_cols) - 0.5)

            alt_max = np.nanmax(alt_vals) if np.any(~np.isnan(alt_vals)) else 2.0
            alt_ceil = np.ceil(alt_max * 10) / 10
            alt_ceil = max(alt_ceil, 0.5)

            ax3.set_ylim(0.0, alt_ceil)

            ax3.set_yticks(
                np.arange(
                    0,
                    alt_ceil + 0.5,
                    max(1.0, np.floor(alt_ceil / 3)),
                )
            )

            if show_ylabel:
                ax3.set_ylabel(
                    "ALT\n(log10)",
                    fontsize=font_ylabel,
                    fontweight="bold",
                    rotation=90,
                )
            else:
                ax3.set_ylabel("")

            ax3.tick_params(axis="y", labelsize=font_yticks)
            ax3.tick_params(axis="x", bottom=False, labelbottom=False)

        # ---------------------------------------------------------------------
        # REF panel
        # ---------------------------------------------------------------------
        if plot_alt_ref and ax_ref is not None:
            ax4 = ax_ref

            for name, spine in ax4.spines.items():
                if name in ("left", "bottom", "top"):
                    spine.set_visible(True)
                    spine.set_linewidth(1.0)
                    spine.set_edgecolor("black")
                    spine.set_zorder(10)
                else:
                    spine.set_visible(False)

            ax4.set_facecolor("#dbf8ee")

            ref_vals = (
                np.log10(df_ref_num + 1.0)
                .mean(axis=0, numeric_only=True)
                .values
                .astype(float)
            )

            good_ref = (~low_dp_mask) & (~np.isnan(ref_vals))
            low_ref = low_dp_mask & (~np.isnan(ref_vals))

            ax4.scatter(
                x_positions[good_ref],
                ref_vals[good_ref],
                c="#459d5b",
                s=20,
                alpha=1.0,
                edgecolors="black",
                linewidths=0.3,
                zorder=2,
            )

            ax4.scatter(
                x_positions[low_ref],
                ref_vals[low_ref],
                c="#459d5b",
                s=20,
                alpha=0.4,
                edgecolors="black",
                linewidths=0.3,
                zorder=2,
            )

            _draw_cytoband_separators(ax4)
            _draw_chr_arm_boundaries(ax4)

            ax4.set_xlim(-0.5, len(sorted_cols) - 0.5)

            ref_max = np.nanmax(ref_vals) if np.any(~np.isnan(ref_vals)) else 2.0
            ref_ceil = np.ceil(ref_max * 10) / 10
            ref_ceil = max(ref_ceil, 0.5)

            ax4.set_ylim(0.0, ref_ceil)

            ax4.set_yticks(
                np.arange(
                    0,
                    ref_ceil + 0.5,
                    max(1.0, np.floor(ref_ceil / 3)),
                )
            )

            if show_ylabel:
                ax4.set_ylabel(
                    "REF\n(log10)",
                    fontsize=font_ylabel,
                    fontweight="bold",
                    rotation=90,
                )
            else:
                ax4.set_ylabel("")

            ax4.tick_params(axis="y", labelsize=font_yticks)
            ax4.tick_params(axis="x", bottom=False, labelbottom=False)

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
    if plot_separate_clones:
        out_dir = os.path.dirname(output_path)
        out_base = os.path.basename(output_path)
        out_name, out_ext = os.path.splitext(out_base)

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        for i, label in enumerate(unique_labels):
            vprint(f"\nPlotting clone {i + 1}: {label}")

            if plot_alt_ref:
                height_ratios = [height_baf, height_cnv, height_alt, height_ref]
                n_panels = 4
            else:
                height_ratios = [height_baf, height_cnv]
                n_panels = 2

            fig_h = height_multiplier * sum(height_ratios)

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

            if n_panels == 1:
                axes = [axes]

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

            fig.savefig(clone_output_path, dpi=dpi, bbox_inches="tight")
            fig.savefig(clone_output_path.replace('.png', '.pdf'), dpi=dpi, bbox_inches="tight")
            vprint(f"  ✓ Saved: {clone_output_path}")

            plt.show()
            plt.close(fig)

    else:
        raise NotImplementedError(
            "Stacked mode was removed in this patch. "
            "Set plot_separate_clones=True, or restore stacked mode separately."
        )

    vprint("\n" + "=" * 60)
    vprint("✓ PLOT COMPLETE")
    vprint("=" * 60)