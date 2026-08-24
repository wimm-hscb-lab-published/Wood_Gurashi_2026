import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def map_loh_to_cytobands(
    results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    cytobands: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Map LOH regions to specific cytobands based on genomic coordinates.
    
    IMPORTANT: If a cell has multiple segments on the same chromosome arm,
    they are combined into a single row with a semicolon-separated range
    (e.g., 'q11.1-q15; q22.1-q35.2' for biallelic/complex deletions).
    
    Parameters
    ----------
    results : Dict[str, Tuple]
        Results from process_all_chromosomes, keyed by chromosome arm
        Each value is (df_segments, df_loh, df_corr, df_homog)
    cytobands : pd.DataFrame
        Cytoband reference data with columns:
        - chromosome arm: e.g., 'chr1p', 'chr1q'
        - start: cytoband start position
        - end: cytoband end position
        - cytoband: cytoband name (e.g., 'p36.33')
    verbose : bool
        Print progress
        
    Returns
    -------
    df_loh_cytobands : pd.DataFrame
        One row per cell per chromosome arm, with combined ranges for cells with multiple segments
    """
    
    import numpy as np
    import pandas as pd
    from typing import Dict, List, Tuple
    
    def vprint(msg):
        if verbose:
            print(msg)
            import sys
            sys.stdout.flush()
    
    vprint("\n" + "="*60)
    vprint("MAPPING LOH TO CYTOBANDS")
    vprint("="*60)
    
    # Verify cytoband columns
    required_cols = ['chromosome arm', 'start', 'end', 'cytoband']
    missing_cols = [col for col in required_cols if col not in cytobands.columns]
    
    if missing_cols:
        vprint(f"⚠️  Missing columns in cytobands: {missing_cols}")
        vprint(f"   Available columns: {list(cytobands.columns)}")
        return pd.DataFrame()
    
    # Process each chromosome arm
    loh_cytoband_records = []
    
    for chr_arm, (df_segments, df_loh, df_corr, df_homog) in results.items():
        vprint(f"\n  Processing {chr_arm}...")
        
        if len(df_segments) == 0:
            vprint(f"    No segments found")
            continue
        
        # Determine correct column name for cell identifier
        if 'cell_id' in df_segments.columns:
            cell_col = 'cell_id'
        elif 'cell' in df_segments.columns:
            cell_col = 'cell'
        else:
            vprint(f"    ⚠️  No cell identifier column found")
            continue
        
        # Get cytobands for this chromosome arm
        chr_cytobands = cytobands[cytobands['chromosome arm'] == chr_arm].copy()
        
        if len(chr_cytobands) == 0:
            vprint(f"    ⚠️  No cytoband data for {chr_arm}")
            continue
        
        # Sort cytobands by position
        chr_cytobands = chr_cytobands.sort_values('start').reset_index(drop=True)
        
        vprint(f"    LOH segments: {len(df_segments)}")
        vprint(f"    Cytobands available: {len(chr_cytobands)}")
        
        # GROUP BY CELL - this is the key change!
        cell_groups = df_segments.groupby(cell_col)
        
        vprint(f"    Unique cells with LOH: {len(cell_groups)}")
        
        for cell_id, cell_segments in cell_groups:
            # Get cluster ID (should be same for all segments from this cell)
            cluster = cell_segments['cluster'].iloc[0] if 'cluster' in cell_segments.columns else 'Unknown'
            
            # Process each segment for this cell
            segment_ranges = []
            all_affected_cytobands = []
            segment_stats = []
            
            for idx, row in cell_segments.iterrows():
                loh_start = row['genomic_start']
                loh_end = row['genomic_end']
                
                # Find overlapping cytobands for this segment
                overlapping = chr_cytobands[
                    (chr_cytobands['start'] <= loh_end) & 
                    (chr_cytobands['end'] >= loh_start)
                ].copy()
                
                if len(overlapping) == 0:
                    # No exact overlap - find nearest
                    chr_cytobands['distance'] = chr_cytobands.apply(
                        lambda x: min(abs(x['start'] - loh_start), abs(x['end'] - loh_end)), axis=1
                    )
                    closest = chr_cytobands.nsmallest(1, 'distance')
                    affected_cytobands = list(closest['cytoband'])
                    cytoband_range = affected_cytobands[0] if len(affected_cytobands) > 0 else 'Unknown'
                else:
                    affected_cytobands = list(overlapping['cytoband'])
                    cytoband_start = affected_cytobands[0]
                    cytoband_end = affected_cytobands[-1]
                    
                    # Create range string for this segment
                    if len(affected_cytobands) == 1:
                        cytoband_range = affected_cytobands[0]
                    else:
                        cytoband_range = f"{cytoband_start}-{cytoband_end}"
                
                segment_ranges.append(cytoband_range)
                all_affected_cytobands.extend(affected_cytobands)
                segment_stats.append({
                    'start': loh_start,
                    'end': loh_end,
                    'length': loh_end - loh_start,
                    'n_variants': row.get('n_variants', 0),
                    'mean_af': row.get('mean_allele_freq', np.nan),
                    'extreme_frac': row.get('extreme_fraction', np.nan)
                })
            
            # CRITICAL: Combine multiple segments with semicolon
            combined_range = '; '.join(segment_ranges)
            
            # Get unique cytobands (maintain order)
            unique_cytobands = []
            seen = set()
            for cb in all_affected_cytobands:
                if cb not in seen:
                    unique_cytobands.append(cb)
                    seen.add(cb)
            
            # Aggregate stats
            total_length = sum(s['length'] for s in segment_stats)
            total_variants = sum(s['n_variants'] for s in segment_stats)
            mean_af = np.nanmean([s['mean_af'] for s in segment_stats])
            mean_extreme = np.nanmean([s['extreme_frac'] for s in segment_stats])
            
            # Get overall start/end
            genomic_start = min(s['start'] for s in segment_stats)
            genomic_end = max(s['end'] for s in segment_stats)
            
            # Determine cytoband_start and cytoband_end (from first and last unique cytobands)
            cytoband_start = unique_cytobands[0] if len(unique_cytobands) > 0 else 'Unknown'
            cytoband_end = unique_cytobands[-1] if len(unique_cytobands) > 0 else 'Unknown'
            
            loh_cytoband_records.append({
                'chr_arm': chr_arm,
                'cluster': cluster,
                'clone': f"LOH-C{cluster} {chr_arm}" if cluster != 'Unknown' else chr_arm,
                'cell_id': cell_id,
                'n_segments': len(cell_segments),  # NEW: Number of segments
                'genomic_start': genomic_start,
                'genomic_end': genomic_end,
                'length_mb': total_length / 1_000_000,
                'cytoband_start': cytoband_start,
                'cytoband_end': cytoband_end,
                'cytoband_range': combined_range,  # COMBINED with semicolons
                'affected_cytobands': ', '.join(unique_cytobands),
                'n_cytobands': len(unique_cytobands),
                'n_variants': total_variants,
                'mean_allele_freq': mean_af,
                'extreme_fraction': mean_extreme
            })
        
        vprint(f"    ✓ Mapped {len(cell_groups)} cells to cytobands")
    
    df_loh_cytobands = pd.DataFrame(loh_cytoband_records)
    
    if len(df_loh_cytobands) == 0:
        vprint("\n  ⚠️  No LOH-cytoband mappings created")
        return df_loh_cytobands
    
    # Summary statistics
    vprint("\n" + "="*60)
    vprint("SUMMARY")
    vprint("="*60)
    
    vprint(f"\n  Total cells mapped: {len(df_loh_cytobands)}")
    vprint(f"  Unique chromosome arms: {df_loh_cytobands['chr_arm'].nunique()}")
    
    # Count cells with multiple segments (biallelic/complex)
    n_multi_segment = (df_loh_cytobands['n_segments'] > 1).sum()
    if n_multi_segment > 0:
        vprint(f"  Cells with multiple segments: {n_multi_segment} ({100*n_multi_segment/len(df_loh_cytobands):.1f}%)")
        vprint(f"    (These show as combined ranges like 'q11.1-q15; q22.1-q35.2')")
    
    # Per chromosome arm summary
    vprint(f"\n  LOH by chromosome arm:")
    for chr_arm in sorted(df_loh_cytobands['chr_arm'].unique()):
        chr_data = df_loh_cytobands[df_loh_cytobands['chr_arm'] == chr_arm]
        n_cells = len(chr_data)
        n_clusters = chr_data['cluster'].nunique()
        n_multi = (chr_data['n_segments'] > 1).sum()
        
        # Get unique cytobands affected
        all_cytobands = []
        for cb_str in chr_data['affected_cytobands']:
            all_cytobands.extend(cb_str.split(', '))
        unique_cytobands = sorted(set(all_cytobands))
        
        vprint(f"    {chr_arm}: {n_cells} cells, {n_clusters} clusters")
        if n_multi > 0:
            vprint(f"      Cells with multiple segments: {n_multi} ({100*n_multi/n_cells:.1f}%)")
        vprint(f"      All affected cytobands: {', '.join(unique_cytobands)}")
        vprint(f"      Full range: {unique_cytobands[0]} to {unique_cytobands[-1]}")
        
        # Show most common cytoband ranges
        top_ranges = chr_data['cytoband_range'].value_counts().head(3)
        vprint(f"      Most common ranges:")
        for cb_range, count in top_ranges.items():
            vprint(f"        {cb_range}: {count} cells ({100*count/len(chr_data):.1f}%)")
    
    return df_loh_cytobands


def summarize_loh_cytobands_by_cluster(
    df_loh_cytobands: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create cluster-level summary of affected cytobands.
    
    Parameters
    ----------
    df_loh_cytobands : pd.DataFrame
        Output from map_loh_to_cytobands
    verbose : bool
        Print progress
        
    Returns
    -------
    df_cluster_summary : pd.DataFrame
        Cluster-level summary with detailed cytoband information
    """
    
    import pandas as pd
    import numpy as np
    
    def vprint(msg):
        if verbose:
            print(msg)
            import sys
            sys.stdout.flush()
    
    if len(df_loh_cytobands) == 0:
        vprint("⚠️  No data to summarize")
        return pd.DataFrame()
    
    vprint("\n" + "="*60)
    vprint("CLUSTER-LEVEL CYTOBAND SUMMARY")
    vprint("="*60)
    
    cluster_summaries = []
    
    for (chr_arm, cluster), group in df_loh_cytobands.groupby(['chr_arm', 'cluster']):
        # Collect all affected cytobands
        all_cytobands = []
        for cb_str in group['affected_cytobands']:
            all_cytobands.extend(cb_str.split(', '))
        unique_cytobands = sorted(set(all_cytobands))
        
        # Find consensus range (most common)
        consensus_range = group['cytoband_range'].mode()
        consensus_range = consensus_range.iloc[0] if len(consensus_range) > 0 else 'Variable'
        
        # Get min and max observed
        min_start = group['cytoband_start'].mode().iloc[0] if len(group['cytoband_start'].mode()) > 0 else unique_cytobands[0]
        max_end = group['cytoband_end'].mode().iloc[0] if len(group['cytoband_end'].mode()) > 0 else unique_cytobands[-1]
        full_range = f"{unique_cytobands[0]}-{unique_cytobands[-1]}" if len(unique_cytobands) > 1 else unique_cytobands[0]
        
        cluster_summaries.append({
            'chr_arm': chr_arm,
            'cluster': cluster,
            'clone': f"LOH-C{cluster} {chr_arm}",
            'n_cells': group['cell_id'].nunique(),
            'n_segments': len(group),
            'cytoband_min': unique_cytobands[0],
            'cytoband_max': unique_cytobands[-1],
            'cytoband_full_range': full_range,
            'cytoband_consensus': consensus_range,
            'consensus_frequency': (group['cytoband_range'] == consensus_range).sum() / len(group),
            'affected_cytobands': ', '.join(unique_cytobands),
            'n_unique_cytobands': len(unique_cytobands),
            'mean_length_mb': group['length_mb'].mean(),
            'median_length_mb': group['length_mb'].median(),
            'min_length_mb': group['length_mb'].min(),
            'max_length_mb': group['length_mb'].max(),
            'mean_n_variants': group['n_variants'].mean(),
            'mean_extreme_fraction': group['extreme_fraction'].mean()
        })
    
    df_cluster_summary = pd.DataFrame(cluster_summaries)
    
    # Sort by chromosome arm and number of cells
    df_cluster_summary = df_cluster_summary.sort_values(
        ['chr_arm', 'n_cells'], 
        ascending=[True, False]
    ).reset_index(drop=True)
    
    vprint(f"\n  Cluster summaries: {len(df_cluster_summary)}")
    
    # Show details per chromosome arm
    vprint(f"\n  Clusters by chromosome arm:")
    for chr_arm in sorted(df_cluster_summary['chr_arm'].unique()):
        chr_data = df_cluster_summary[df_cluster_summary['chr_arm'] == chr_arm]
        vprint(f"\n    {chr_arm}:")
        
        for idx, row in chr_data.iterrows():
            vprint(f"      Cluster {row['cluster']}: {row['n_cells']} cells")
            vprint(f"        Min cytoband: {row['cytoband_min']}")
            vprint(f"        Max cytoband: {row['cytoband_max']}")
            vprint(f"        Full range: {row['cytoband_full_range']}")
            vprint(f"        Consensus: {row['cytoband_consensus']} ({row['consensus_frequency']:.1%} of segments)")
            vprint(f"        All cytobands: {row['affected_cytobands']}")
            vprint(f"        Length: {row['min_length_mb']:.1f} - {row['max_length_mb']:.1f} Mb (median: {row['median_length_mb']:.1f} Mb)")
    
    return df_cluster_summary


def analyze_loh_heterogeneity(
    df_loh_cytobands: pd.DataFrame,
    chr_arm: str,
    cluster: int,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Analyze heterogeneity within an LOH cluster to identify potential subclones.
    
    Parameters
    ----------
    df_loh_cytobands : pd.DataFrame
        Output from map_loh_to_cytobands
    chr_arm : str
        Chromosome arm to analyze (e.g., 'chr5q')
    cluster : int
        Cluster ID to analyze
    verbose : bool
        Print detailed analysis
        
    Returns
    -------
    df_subclones : pd.DataFrame
        Summary of potential subclones based on cytoband ranges
    """
    
    import pandas as pd
    import numpy as np
    
    def vprint(msg):
        if verbose:
            print(msg)
            import sys
            sys.stdout.flush()
    
    # Filter to specific cluster
    cluster_data = df_loh_cytobands[
        (df_loh_cytobands['chr_arm'] == chr_arm) & 
        (df_loh_cytobands['cluster'] == cluster)
    ].copy()
    
    if len(cluster_data) == 0:
        vprint(f"⚠️  No data found for {chr_arm} cluster {cluster}")
        return pd.DataFrame()
    
    vprint("\n" + "="*60)
    vprint(f"LOH HETEROGENEITY ANALYSIS: {chr_arm} Cluster {cluster}")
    vprint("="*60)
    
    vprint(f"\n  Total cells: {cluster_data['cell_id'].nunique()}")
    vprint(f"  Total segments: {len(cluster_data)}")
    
    # Group by cytoband range to identify subclones
    range_groups = cluster_data.groupby('cytoband_range').agg({
        'cell_id': 'nunique',
        'length_mb': ['mean', 'median', 'min', 'max'],
        'n_variants': 'mean',
        'cytoband_start': 'first',
        'cytoband_end': 'first'
    }).reset_index()
    
    # Flatten column names
    range_groups.columns = ['cytoband_range', 'n_cells', 'mean_length_mb', 
                            'median_length_mb', 'min_length_mb', 'max_length_mb',
                            'mean_n_variants', 'cytoband_start', 'cytoband_end']
    
    # Sort by number of cells (descending)
    range_groups = range_groups.sort_values('n_cells', ascending=False).reset_index(drop=True)
    
    # Calculate percentage
    total_cells = cluster_data['cell_id'].nunique()
    range_groups['percent_cells'] = 100 * range_groups['n_cells'] / total_cells
    
    # Assign subclone labels
    range_groups['subclone'] = [f"Subclone {i+1}" for i in range(len(range_groups))]
    
    vprint(f"\n  Distinct LOH patterns: {len(range_groups)}")
    vprint(f"\n  Subclone breakdown:")
    
    for idx, row in range_groups.iterrows():
        vprint(f"\n    {row['subclone']}: {row['cytoband_range']}")
        vprint(f"      Cells: {row['n_cells']} ({row['percent_cells']:.1f}%)")
        vprint(f"      Length: {row['median_length_mb']:.1f} Mb (median)")
        vprint(f"      Variants: {row['mean_n_variants']:.1f} (mean)")
        vprint(f"      Range: {row['cytoband_start']} to {row['cytoband_end']}")
    
    # Analyze nested deletions
    vprint(f"\n  Deletion pattern analysis:")
    
    # Create a simplified representation for comparison
    range_groups['deletion_size_rank'] = range_groups['median_length_mb'].rank(ascending=False)
    
    # Check if smaller deletions are nested within larger ones
    for i in range(len(range_groups)):
        for j in range(i+1, len(range_groups)):
            range_i = range_groups.iloc[i]
            range_j = range_groups.iloc[j]
            
            # Check if one is nested in the other
            if (range_i['cytoband_start'] >= range_j['cytoband_start'] and 
                range_i['cytoband_end'] <= range_j['cytoband_end']):
                vprint(f"    • {range_i['subclone']} ({range_i['cytoband_range']}) is NESTED within")
                vprint(f"      {range_j['subclone']} ({range_j['cytoband_range']})")
            elif (range_j['cytoband_start'] >= range_i['cytoband_start'] and 
                  range_j['cytoband_end'] <= range_i['cytoband_end']):
                vprint(f"    • {range_j['subclone']} ({range_j['cytoband_range']}) is NESTED within")
                vprint(f"      {range_i['subclone']} ({range_i['cytoband_range']})")
    
    # Statistical summary
    vprint(f"\n  Statistical summary:")
    vprint(f"    Length range: {cluster_data['length_mb'].min():.1f} - {cluster_data['length_mb'].max():.1f} Mb")
    vprint(f"    Length median: {cluster_data['length_mb'].median():.1f} Mb")
    vprint(f"    Length std dev: {cluster_data['length_mb'].std():.1f} Mb")
    
    return range_groups


def compare_all_loh_heterogeneity(
    df_loh_cytobands: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare heterogeneity across all LOH clusters.
    
    Parameters
    ----------
    df_loh_cytobands : pd.DataFrame
        Output from map_loh_to_cytobands
    verbose : bool
        Print summary
        
    Returns
    -------
    df_heterogeneity : pd.DataFrame
        Heterogeneity metrics for each cluster
    """
    
    import pandas as pd
    import numpy as np
    
    def vprint(msg):
        if verbose:
            print(msg)
            import sys
            sys.stdout.flush()
    
    vprint("\n" + "="*60)
    vprint("LOH HETEROGENEITY COMPARISON")
    vprint("="*60)
    
    heterogeneity_records = []
    
    for (chr_arm, cluster), group in df_loh_cytobands.groupby(['chr_arm', 'cluster']):
        # Count distinct patterns
        n_patterns = group['cytoband_range'].nunique()
        n_cells = group['cell_id'].nunique()
        
        # Get most common pattern
        mode_range = group['cytoband_range'].mode().iloc[0]
        mode_count = (group['cytoband_range'] == mode_range).sum()
        mode_percent = 100 * mode_count / len(group)
        
        # Calculate length variability
        length_std = group['length_mb'].std()
        length_cv = length_std / group['length_mb'].mean() if group['length_mb'].mean() > 0 else 0
        
        # Get full range
        all_cytobands = []
        for cb_str in group['affected_cytobands']:
            all_cytobands.extend(cb_str.split(', '))
        unique_cytobands = sorted(set(all_cytobands))
        full_range = f"{unique_cytobands[0]}-{unique_cytobands[-1]}" if len(unique_cytobands) > 1 else unique_cytobands[0]
        
        heterogeneity_records.append({
            'chr_arm': chr_arm,
            'cluster': cluster,
            'n_cells': n_cells,
            'n_patterns': n_patterns,
            'heterogeneity_index': n_patterns / n_cells,  # Higher = more heterogeneous
            'consensus_range': mode_range,
            'consensus_percent': mode_percent,
            'full_range': full_range,
            'length_std_mb': length_std,
            'length_cv': length_cv,
            'min_length_mb': group['length_mb'].min(),
            'max_length_mb': group['length_mb'].max(),
            'median_length_mb': group['length_mb'].median()
        })
    
    df_heterogeneity = pd.DataFrame(heterogeneity_records)
    df_heterogeneity = df_heterogeneity.sort_values('heterogeneity_index', ascending=False)
    
    vprint(f"\n  Clusters ranked by heterogeneity:")
    for idx, row in df_heterogeneity.iterrows():
        vprint(f"\n    {row['chr_arm']} Cluster {row['cluster']}:")
        vprint(f"      Heterogeneity index: {row['heterogeneity_index']:.3f}")
        vprint(f"      Patterns: {row['n_patterns']} distinct ranges among {row['n_cells']} cells")
        vprint(f"      Consensus: {row['consensus_range']} ({row['consensus_percent']:.1f}%)")
        vprint(f"      Full range: {row['full_range']}")
        vprint(f"      Length: {row['min_length_mb']:.1f} - {row['max_length_mb']:.1f} Mb (CV: {row['length_cv']:.2f})")
        
        if row['heterogeneity_index'] > 0.1:
            vprint(f"      ⚠️  HIGH HETEROGENEITY - likely contains subclones")
    
    return df_heterogeneity

