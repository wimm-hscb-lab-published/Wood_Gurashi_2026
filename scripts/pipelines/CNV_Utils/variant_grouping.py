from collections import defaultdict


def group_informative_variants_by_chr_arm(filtered_germline_in_samples, cytobands, verbose=True):
    """
    Group informative variants by chromosome and chromosome arm using cytoband information.
    
    Parameters
    ----------
    filtered_germline_in_samples : array-like
        List of variant IDs in format 'chrN:position:REF/ALT'
    cytobands : pd.DataFrame
        Cytobands dataframe with columns: chromosome, arm, start, end
    verbose : bool, default=True
        Print progress and results
        
    Returns
    -------
    grouped_by_chr : dict
        Dictionary mapping chromosome names to list of variants
        Example: {'chr1': ['chr1:12345:A/T', ...], 'chr2': [...]}
    grouped_by_chr_arm : dict
        Dictionary mapping chromosome arms to list of variants
        Example: {'chr1p': ['chr1:12345:A/T', ...], 'chr1q': [...]}
    """
    from collections import defaultdict
    
    def vprint(msg):
        if verbose:
            print(msg)
    
    vprint("\n" + "="*60)
    vprint("GROUPING INFORMATIVE VARIANTS BY CHROMOSOME ARM")
    vprint("="*60)
    
    # ==========================================
    # STEP 1: GROUP BY CHROMOSOME
    # ==========================================
    vprint("\n  Step 1: Grouping by chromosome...")
    
    grouped_by_chr = defaultdict(list)
    for var in filtered_germline_in_samples:
        chr_part = var.split(':')[0]
        grouped_by_chr[chr_part].append(var)
    
    vprint(f"    ✓ Grouped {len(filtered_germline_in_samples)} variants into {len(grouped_by_chr)} chromosomes")
    
    if verbose:
        for chr_name in sorted(grouped_by_chr.keys(), key=lambda x: (len(x), x)):
            vprint(f"      • {chr_name}: {len(grouped_by_chr[chr_name])} variants")
    
    # ==========================================
    # STEP 2: BUILD ARM LOOKUP TABLE
    # ==========================================
    vprint("\n  Step 2: Building chromosome arm lookup table...")
    
    # Build a mapping from (chromosome, arm) to (start, end)
    chrom_arm_ranges = (
        cytobands
        .groupby(['chromosome', 'arm'])
        .agg({'start': 'min', 'end': 'max'})
        .reset_index()
    )
    
    # Create a lookup for fast assignment
    arm_lookup = {}
    for _, row in chrom_arm_ranges.iterrows():
        arm_lookup.setdefault(row['chromosome'], []).append({
            'arm': row['arm'],
            'start': row['start'],
            'end': row['end']
        })
    
    vprint(f"    ✓ Built lookup table for {len(arm_lookup)} chromosomes")
    vprint(f"    ✓ Total chromosome arms: {len(chrom_arm_ranges)}")
    
    # ==========================================
    # STEP 3: GROUP BY CHROMOSOME ARM
    # ==========================================
    vprint("\n  Step 3: Assigning variants to chromosome arms...")
    
    grouped_by_chr_arm = {}
    unmapped_variants = []
    
    for chr_name, variants in grouped_by_chr.items():
        # Remove 'chr' prefix for matching with cytobands
        chr_short = chr_name.replace('chr', '')
        
        if chr_short not in arm_lookup:
            vprint(f"      ⚠️  Warning: No cytoband data for {chr_name}")
            unmapped_variants.extend(variants)
            continue
        
        for variant in variants:
            # Extract position from variant string: chrN:position:REF/ALT
            try:
                pos = int(variant.split(':')[1])
            except Exception as e:
                vprint(f"      ⚠️  Warning: Could not parse position from '{variant}': {e}")
                unmapped_variants.append(variant)
                continue
            
            # Find which arm this variant belongs to
            assigned = False
            for arm_info in arm_lookup[chr_short]:
                if arm_info['start'] <= pos <= arm_info['end']:
                    arm_key = f"{chr_name}{arm_info['arm']}"
                    if arm_key not in grouped_by_chr_arm:
                        grouped_by_chr_arm[arm_key] = []
                    grouped_by_chr_arm[arm_key].append(variant)
                    assigned = True
                    break
            
            if not assigned:
                unmapped_variants.append(variant)
    
    vprint(f"    ✓ Grouped variants into {len(grouped_by_chr_arm)} chromosome arms")
    
    if unmapped_variants:
        vprint(f"    ⚠️  {len(unmapped_variants)} variants could not be mapped to chromosome arms")
    
    # ==========================================
    # STEP 4: SUMMARY
    # ==========================================
    if verbose:
        vprint("\n  Chromosome arm distribution:")
        for arm_key in sorted(grouped_by_chr_arm.keys(), key=lambda x: (len(x.replace('chr', '')), x)):
            vprint(f"      • {arm_key}: {len(grouped_by_chr_arm[arm_key])} variants")
    
    vprint("\n" + "="*60)
    vprint("GROUPING COMPLETE")
    vprint("="*60)
    vprint(f"\n  Total variants: {len(filtered_germline_in_samples)}")
    vprint(f"  Chromosomes: {len(grouped_by_chr)}")
    vprint(f"  Chromosome arms: {len(grouped_by_chr_arm)}")
    vprint(f"  Unmapped variants: {len(unmapped_variants)}")
    vprint("="*60 + "\n")
    
    return grouped_by_chr, grouped_by_chr_arm

