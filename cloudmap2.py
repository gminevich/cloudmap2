###############################################################################
# CloudMap2 
# 
# Bulked segregant mapping tool that generates maps of linked chromosomal 
# regions with color-coded SNPs of interest. Also uses kernel density 
# estimation on linked parental SNPs to predict position of most likely causal 
# SNP(s). 
###############################################################################

import argparse

from collections import OrderedDict
from itertools import islice

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

from scipy.stats import kde
from matplotlib.backends.backend_pdf import PdfPages


# Adjustable SNP ratios (adjust if your pool of mutants has been contaminated
# with WT). Values below this ratio (at mapping SNP positions only) will count 
# as non-SNP variants.
permissive_ratio_HA_SNP = .1

# Values above the ratio below (in regions of linkage only) will count as
# parental SNP variants
permissive_ratio_Homoz_Parental_SNP = .6
read_depth_for_SNP_consideration = 10
alt_alleles_for_SNP_consideration = 2

# Sliding window parameter
window_size = 10

# Express this as a ratio rather than an absolute number to make the parameter
# independent of the chosen window_size.
permissible_ratio_count_in_window = 0.7 

# Deals with cases where EMS mutation/random mutation in the parental strain
# exactly matches mapping strain SNP. Thus falsely interpreted as a mapping 
# strain SNP in analysis.
spurious_mapping_snp_position_ratio_threshold = .6  

# Whether to: 
# plot all parental variants (false)
# or EMS variants (G/C—>A/T) only (true).
ems_flag = 'false'

# For purposes of making plots for paper figures, can label the known causal
# variant.
# hu80: cic-1 chr III:2,401,031
# ot785: lin-13 chr III:7,701,091
# causal_chromosome_flag = 'chrIII'
# causal_variant_position = '2401031'

max_x = 0
max_y = 0


def main(Homozygous_Bulked_Segregant_SNPs_w_Mapping_Strain_SNPs_Subtracted_file, Bulked_Segregant_Allele_Ratios_at_Mapping_Strain_SNP_Positions_file, ofile):
    # ToDo: convert all hardcoded options to command line parameters

    mapping_strain_positions_df = parse_mapping_strain_positions_df(
        load_vcf_as_df(Bulked_Segregant_Allele_Ratios_at_Mapping_Strain_SNP_Positions_file)
        )
    parental_homozygous_snps_df = load_vcf_as_df(Homozygous_Bulked_Segregant_SNPs_w_Mapping_Strain_SNPs_Subtracted_file)    
    # ToDo: generalize this to handle all the different formats of chromosome
    # names. 
    all_chromosomes = pd.unique(mapping_strain_positions_df.CHROM.ravel()) 
    # Remove MtDNA from ndarray, right now only do it by index, but should
    # likely also plot MtDNA since there could be causal mutations there.
    all_chromosomes = np.delete(all_chromosomes, 4)

    mapping_plot_pdf, fig = initiate_plot(ofile)

    for chrom in all_chromosomes:
        # Debug
        print ("--------------------------------")
        print ("chrom: ", chrom)
        # Get longest region of linkage on each chromosome
        max_window_start, max_window_end = \
            longest_region_of_parental_linkage_via_mapping_snps(
                mapping_strain_positions_df, chrom
                )
        # Get parental strain SNPs in longest mapping region on each chromosome
        parental_strain_SNPs_in_longest_mapping_region = \
            parental_strain_variants_in_longest_non_crossing_strain_subsegment(
                parental_homozygous_snps_df,
                max_window_start, max_window_end, chrom
                )
        # Calculate KDE on the parental strain SNPs within the mapping region
        # Need at least 3 variants in order to use KDE
        if len(parental_strain_SNPs_in_longest_mapping_region.index) >= 3:
            kde_max_y, kde_max_x, xgrid, probablity_density_function = \
                kernel_density_estimation(
                    parental_strain_SNPs_in_longest_mapping_region,
                    chrom
                    )
            kde_output_plot(
                xgrid, probablity_density_function,
                chrom, parental_strain_SNPs_in_longest_mapping_region,
                kde_max_x, kde_max_y
                )              
        # If not enough SNPs to use KDE, plot them without kde.
        elif 1 >= len(parental_strain_SNPs_in_longest_mapping_region.index) < 3:
            minimal_snp_plot(
                chrom, parental_strain_SNPs_in_longest_mapping_region
                )
        else:
            minimal_snp_plot(
                chrom, parental_strain_SNPs_in_longest_mapping_region
                )

    finish_plot(mapping_plot_pdf, fig)
        
        
def initiate_plot(ofile):
    """Set up plotting parameters."""

    plt.style.use('ggplot')
    mapping_plot_pdf = PdfPages(ofile)
    
    fig = plt.figure(figsize=(11.69, 8.27), dpi=150)  
    return mapping_plot_pdf, fig

   
def finish_plot(mapping_plot_pdf=None, fig=None):
    """Finalize the plotting parameters and save."""

    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=25)
    
    plt.tight_layout(pad=.4, h_pad=.4, w_pad=.4)
    plt.subplots_adjust(wspace=.3, hspace=.7)

    mapping_plot_pdf.savefig(fig, pad_inches=.5, orientation='landscape')
    mapping_plot_pdf.close()


def load_vcf_as_df(vcf_file):
    """ Reads in a VCF, and converts key columns to a dataframe """
    vcf_as_df = pd.read_csv(vcf_file, header='infer', comment='#', sep='\t')
    vcf_as_df.columns = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER',  'INFO', 'FORMAT', 'SAMPLE']
    return vcf_as_df


def parse_mapping_strain_positions_df(mapping_strain_positions_df = None):
    """ Parse the key elements of the vcf file """
    vcf_read_group_data = mapping_strain_positions_df.SAMPLE.str.split(':')
    
    vcf_allele_freq_data = vcf_read_group_data.str[1]
    ref_allele_count = pd.to_numeric(vcf_allele_freq_data.str.split(',').str[0])
    alt_allele_count = pd.to_numeric(vcf_allele_freq_data.str.split(',').str[1])
    read_depth = vcf_read_group_data.str[2]

    # ToDo: Change to ratio of alt/read depth (only for cases where
    # read depth = alt+ref to account for sequencing error/noise).
    # For now, just use alt/read depth.
    ratio = (pd.to_numeric(alt_allele_count)) / (pd.to_numeric(read_depth))

    parsed_mapping_strain_positions_df = pd.concat(
        [mapping_strain_positions_df.CHROM,
         mapping_strain_positions_df.POS,
         alt_allele_count, ref_allele_count,
         pd.to_numeric(read_depth), ratio],
        axis=1
        )
    parsed_mapping_strain_positions_df.columns = [
        'CHROM', 'POS',
        'alt_allele_count', 'ref_allele_count',
        'read_depth', 'ratio'
        ]    
    return parsed_mapping_strain_positions_df


def sliding_window(seq, window_size=10):
    """ Returns a sliding window over data from the iterable """
    iterable = iter(seq)
    result = tuple(islice(iterable, window_size))
    if len(result) == window_size:
        yield result    
    for elem in iterable:
        result = result[1:] + (elem,)
        yield result


def longest_region_of_parental_linkage_via_mapping_snps(
    mapping_strain_positions_df, current_chrom
    ):
    """ 
    Calculates the longest region of parental linkage via analysis of SNP 
    ratios within a sliding window 
    """

    # either 0 ratio positions or cases where at least 2 alternate reads
    mapping_strain_positions_df = mapping_strain_positions_df[
        (mapping_strain_positions_df.alt_allele_count == 0)
        | (mapping_strain_positions_df.alt_allele_count
           > alt_alleles_for_SNP_consideration)
        ] 
    mapping_strain_positions_df = mapping_strain_positions_df.loc[
        (mapping_strain_positions_df.CHROM == current_chrom)
        & (mapping_strain_positions_df.read_depth
           > read_depth_for_SNP_consideration)
        ] 

    # Convert DF to dictionary and find max consecutive interval
    POS_ratio_DF = mapping_strain_positions_df[['POS','ratio']]
    POS_ratio_dict = dict(zip(POS_ratio_DF.POS, POS_ratio_DF.ratio))
    POS_ratio_dict_sorted = OrderedDict(sorted(POS_ratio_dict.items()))
    
    current_run_windows_start = 0
    current_run_windows_end = 0

    # the first/leftmost position in the window that begins the longest
    # interval below threshold
    max_window_start = 0
    # the last/rightmost position in the window that ends the longest interval
    # below threshold
    max_window_end = 0 
    consecutive_windows_below_threshold = 0
    max_consecutive_windows_below_threshold = 0
    # evaluate contents of each window
    for window in sliding_window(POS_ratio_dict_sorted.items(), window_size):
        ratios_accepted = ratios_rejected = 0
        # store POS info of first element of window
        current_window_start = window[0][0]
        for pos, ratio in window:
            # Discounts cases where parental mutations(EMS or random) exactly
            # match at an HA position, treat these as neutral
            if (consecutive_windows_below_threshold > 0) \
               & (ratio > spurious_mapping_snp_position_ratio_threshold): 
                pass
            elif ratio <.1:
                # ratios of < .1 we count as if a 0 ratio, with allowance for
                # sequencing error
                ratios_accepted += 1
            else:
                ratios_rejected += 1
        current_window_end = pos
        if ratios_accepted > 0:
            fraction_ratios_accepted = ratios_accepted / (ratios_accepted
                                                          + ratios_rejected)
        else:
            # do not risk a division by zero
            fraction_ratios_accepted = 0
        
        # e.g. If 4/5 SNPs are below .1, that window is a run of 0 ratio
        # positions and thus counted as "parental"
        if fraction_ratios_accepted < permissible_ratio_count_in_window:
            consecutive_windows_below_threshold = 0
        else:
            consecutive_windows_below_threshold += 1
            if consecutive_windows_below_threshold == 1:
                current_run_windows_start = current_window_start
            current_run_windows_end = current_window_end
            if consecutive_windows_below_threshold > max_consecutive_windows_below_threshold:
                max_consecutive_windows_below_threshold = consecutive_windows_below_threshold
                max_window_start = current_run_windows_start
                max_window_end = pos
    # Debug
    print(
        "max window: {:,} — {:,}"
        .format(max_window_start, max_window_end)
        )
    print("max_consecutive_windows_below_threshold:",
          max_consecutive_windows_below_threshold)
    print(
        "size of max window: {:,}"
        .format(max_window_end - max_window_start)
        )
    return max_window_start, max_window_end


def parental_strain_variants_in_longest_non_crossing_strain_subsegment(
    parental_homozygous_snps_df,
    start_largest_consecutive=None, end_largest_consecutive=None,
    current_chrom=None
    ):
    """
    Identifies parental strain variants in the longest region devoid of crossing
    strain snps 
    """    
    
    # Split SAMPLE Column with each field as a new column
    parental_homozygous_snps_read_group_data = \
        parental_homozygous_snps_df.SAMPLE.str.split(':')
    parental_homozygous_snps_allele_freq_data = \
        parental_homozygous_snps_read_group_data.str[1]
    parental_homozygous_ref_allele_count = \
        parental_homozygous_snps_allele_freq_data.str.split(',').str[0]
    parental_homozygous_ref_allele_count.name = 'ref_allele_count'
    parental_homozygous_alt_allele_count = \
        parental_homozygous_snps_allele_freq_data.str.split(',').str[1]
    parental_homozygous_alt_allele_count.name = 'alt_allele_count'    
    parental_homozygous_read_depth = \
        parental_homozygous_snps_read_group_data.str[2]

    # Ratio of alt/read depth (only for cases where read depth = alt+ref). 
    # For now, just use alt/read depth
    parental_homozygous_ratio = (
        pd.to_numeric(parental_homozygous_alt_allele_count)
        /pd.to_numeric(parental_homozygous_read_depth)
        )

    # Calculate EMS variants
    parental_homozygous_snps_df['EMS'] = np.where(
        np.logical_or(parental_homozygous_snps_df['REF']=='G',
                      parental_homozygous_snps_df['REF']=='C')
        & np.logical_or(parental_homozygous_snps_df['ALT']=='A',
                        parental_homozygous_snps_df['ALT']=='T'),
        'yes', 'no'
        )
    
    # Return DF of key VCF columns (header missing for newly created columns)
    parental_homozygous_snps_df_ems_depth_calc = pd.concat(
        [parental_homozygous_snps_df.EMS,
         parental_homozygous_snps_df.CHROM,
         parental_homozygous_snps_df.POS,
         parental_homozygous_ratio],
        axis=1
        )
    parental_homozygous_snps_df_ems_depth_calc.columns = [
        'EMS', 'CHROM', 'POS', 'ratio'
        ]

    # Get the linked chromosome. Currently there's a different chromosome
    # naming convention in the homozygous parental variants file.
    parental_homozygous_linked_chromosome = \
        parental_homozygous_snps_df_ems_depth_calc[
            (parental_homozygous_snps_df_ems_depth_calc.CHROM==current_chrom)
            & (parental_homozygous_snps_df_ems_depth_calc.ratio
               > permissive_ratio_Homoz_Parental_SNP)
            ]

    # For testing/paper figure purposes plot all parental or just ems variants
    if ems_flag == 'false':
        # All parental variants (EMS and genetic drift)
        parental_strain_SNPs_in_longest_mapping_region = \
            parental_homozygous_linked_chromosome[
                (parental_homozygous_linked_chromosome.POS
                 > start_largest_consecutive)
                & (parental_homozygous_linked_chromosome.POS
                   < end_largest_consecutive)
                ]
    elif ems_flag == 'true':
        # Filtered for EMS variants    
        parental_strain_SNPs_in_longest_mapping_region = \
            parental_homozygous_linked_chromosome[
                (parental_homozygous_linked_chromosome.POS
                 > start_largest_consecutive)
                & (parental_homozygous_linked_chromosome.POS
                   < end_largest_consecutive)
                & (parental_homozygous_linked_chromosome.EMS =="yes")
                ]

    # Debug
    #if current_chrom == causal_chromosome_flag:
    #    parental_strain_SNPs_in_longest_mapping_region.to_csv(current_chrom+'_parental_variants_in_mapping_region.csv')
    return parental_strain_SNPs_in_longest_mapping_region


def set_plot_axes(chrom=None, is_minimal_snp_plot=True):
    """ Creates layout for each figure on the pdf, sets the x-axis """
    chrom_length = 0
    
    if chrom == 'chrI':
        ax = plt.subplot2grid((3, 2), (0, 0))
        if is_minimal_snp_plot:
            chrom_length=16000000
    if chrom == 'chrII':
        ax = plt.subplot2grid((3, 2), (0, 1))
        if is_minimal_snp_plot:
            chrom_length=16000000
    if chrom == 'chrIII':
        ax = plt.subplot2grid((3, 2), (1, 0))
        if is_minimal_snp_plot:
            chrom_length=14000000
    if chrom == 'chrIV':
        ax = plt.subplot2grid((3, 2), (1, 1))
        if is_minimal_snp_plot:
            chrom_length=18000000
    if chrom == 'chrV':
        ax = plt.subplot2grid((3, 2), (2, 0))
        if is_minimal_snp_plot:
            chrom_length=21000000
    if chrom == 'chrX':
        ax = plt.subplot2grid((3, 2), (2, 1))
        if is_minimal_snp_plot:
            chrom_length=18000000
    
    return ax, chrom_length
    
    
def kde_output_plot(
    xgrid, probablity_density_function,
    chrom, parental_strain_SNPs_in_longest_mapping_region,
    kde_max_x, kde_max_y
    ):  
    """ 
    Calculates kernel density estimation on linked parental variants in
    cases where have at least 3 variants. 
    """
    ax, _ = set_plot_axes(chrom, is_minimal_snp_plot=False)
  
    plt.plot(xgrid, probablity_density_function, 'r-')
    plt.title(chrom)
        
    # Use/Remove scientific notation
    # plt.ticklabel_format(style='plain', axis='y')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Add commas to x-axis
    ax.get_xaxis().set_major_formatter(
        tkr.FuncFormatter(lambda x, p: format(int(x), ','))
        )
    # Annotate the predicted position
    ax.annotate(
        ("{:,}".format(round(kde_max_x))),
        (kde_max_x, kde_max_y),
        xytext=(0, -30),
        textcoords='offset points',
        arrowprops=dict(facecolor='black', shrink=0.05),
        horizontalalignment='center',
        verticalalignment='bottom'
        )      
    
    # ToDo: remove?
    # parental_strain_SNPs_in_longest_mapping_region_EMS = parental_homozygous_linked_chromosome[(parental_homozygous_linked_chromosome.POS > start_largest_consecutive) & (parental_homozygous_linked_chromosome.POS < end_largest_consecutive) & (parental_homozygous_linked_chromosome.EMS =="yes")]
    parental_strain_SNPs_in_longest_mapping_region_EMS = \
        parental_strain_SNPs_in_longest_mapping_region[
            (parental_strain_SNPs_in_longest_mapping_region.EMS =="yes")
            ]
    
    # Plot the scatter points on X axis
    plt.plot(
        parental_strain_SNPs_in_longest_mapping_region.POS,
        np.zeros_like(parental_strain_SNPs_in_longest_mapping_region.POS)+0,
        'o',
        color='grey'
        )
    
    # Overplot EMS variants in a different color
    plt.plot(
        parental_strain_SNPs_in_longest_mapping_region_EMS.POS,
        np.zeros_like(parental_strain_SNPs_in_longest_mapping_region_EMS.POS)+0,
        'd',
        color='blue'
        )
    
    # Debug
    # For paper figures, can plot the causal variant so it stands out
    #if chrom == causal_chromosome_flag:
    #    plt.plot(causal_variant_position, 0, 'd', color='red') 
    #if chrom == 'chrI': 
    #    plt.plot(causal_variant_position_I, 0, 'd', color='red') 
    #if chrom == 'chrV':
    #    plt.plot(causal_variant_position_II, 0, 'd', color='red')


def minimal_snp_plot(
    chrom, parental_strain_SNPs_in_longest_mapping_region=None
    ):
    """ 
    For cases where no parental variants in a region of linkage (defined as
    region devoid of crossing strain variants) or < 3 linked parental variants, 
    plot the positions without KDE b/c KDE on < 3 points doesn't work.
    """

    # Debug
    print("Plot performed on these SNPs: \n",
          parental_strain_SNPs_in_longest_mapping_region)
    
    ax, chrom_length = set_plot_axes(chrom, is_minimal_snp_plot=True)
   
    # Add commas to x-axis
    ax.get_xaxis().set_major_formatter(
        tkr.FuncFormatter(lambda x, p: format(int(x), ','))
        )
    
    # Debug: Remove scientific formatting
    #plt.ticklabel_format(style='plain', axis='x')
    
    # different X-axis depending on the amount of snps
    plt.axis([0, chrom_length, 0, 1])
    if len(parental_strain_SNPs_in_longest_mapping_region.index) == 0:
        #if 0 SNPs, then plot empty plot
        plt.axis([0, chrom_length, 0, 1])
    elif len(parental_strain_SNPs_in_longest_mapping_region.index) == 1:
        # if 1 snp, set axes to 1000bp+/-
        plt.axis(
            [parental_strain_SNPs_in_longest_mapping_region.POS.min() - 1000,
             parental_strain_SNPs_in_longest_mapping_region.POS.min() + 1000,
             0, 1]
            )
    elif len(parental_strain_SNPs_in_longest_mapping_region.index) == 2:
        # if 2 snps, set axes to 1000bp+/- each end
        plt.axis(
            [parental_strain_SNPs_in_longest_mapping_region.POS.min() - 1000,
             parental_strain_SNPs_in_longest_mapping_region.POS.max() + 1000,
             0, 1]
            )    

    # Identify the subset of EMS SNPs
    parental_strain_SNPs_in_longest_mapping_region_EMS = \
        parental_strain_SNPs_in_longest_mapping_region[
            (parental_strain_SNPs_in_longest_mapping_region.EMS =="yes")
            ]

    # Plot 1-2 variants without KDE
    if len(parental_strain_SNPs_in_longest_mapping_region.index) > 0:
        for positions in parental_strain_SNPs_in_longest_mapping_region:
            # plot non-EMS variants
            plt.plot(
                parental_strain_SNPs_in_longest_mapping_region.POS,
                np.zeros_like(
                    parental_strain_SNPs_in_longest_mapping_region.POS
                    ) + 0,
                'o', color='grey'
                )
            # overplot EMS variants in a different color
            plt.plot(
                parental_strain_SNPs_in_longest_mapping_region_EMS.POS,
                np.zeros_like(
                    parental_strain_SNPs_in_longest_mapping_region_EMS.POS
                    ) + 0,
                'd', color='blue'
                )

    plt.title(chrom)


def kernel_density_estimation(
    parental_strain_SNPs_in_longest_mapping_region=None, chrom=None
    ):
    """ 
    Perform kernel density estimation on a given set of linked parental 
    variants. scipy.stats module has automatic bandwidth determination: 
    """

    print("KDE performed on these SNPs: \n",
          parental_strain_SNPs_in_longest_mapping_region)
    kernel = kde.gaussian_kde(
        parental_strain_SNPs_in_longest_mapping_region.POS
        )
    xgrid = np.linspace(
        parental_strain_SNPs_in_longest_mapping_region.POS.min(),
        parental_strain_SNPs_in_longest_mapping_region.POS.max()
        )

    probablity_density_function = kernel.evaluate(xgrid)

    # Find the maximum y value and its corresponding x value.
    max_y = max(probablity_density_function)
    max_x = xgrid[probablity_density_function.argmax()] 
    # Print large numbers with commas
    print(
        "Predicted position of mutation: ",
        ("{:,}".format(round(max_x)))
        ) 
    return max_y, max_x, xgrid, probablity_density_function


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Map a causal variant and plot linkage evidence.'
    )
    parser.add_argument('Homozygous_Bulked_Segregant_SNPs_w_Mapping_Strain_SNPs_Subtracted_file', nargs='?',
                        metavar='<Homozygous Bulked Segregant SNPs with mapping strain SNPs subtracted vcf>',
                        help='VCF file with SNPs that appear homozygous '
                             'in the bulked segregants sample, but are '
                             'NOT found in the mapping strain.')
    parser.add_argument('Bulked_Segregant_Allele_Ratios_at_Mapping_Strain_SNP_Positions_file', nargs='?',
                        metavar='<Bulked Segregant Allele Ratios at Mapping Strain SNP Positions vcf>',
                        help='VCF file with allele ratios found in the bulked segregants sample '
                             'at mapping strain SNP positions only.')
    parser.add_argument('-o', '--ofile', required=True,
                        help="output file for the plot")
    args = vars(parser.parse_args())
    main(**args)
