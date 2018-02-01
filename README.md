## CloudMap2

A [bulked segregant mapping](https://en.wikipedia.org/wiki/Bulked_segregant_analysis) tool that finds regions of linkage on each chromosome, color-codes SNPs in that region according to their source (e.g. [EMS](https://en.wikipedia.org/wiki/Ethyl_methanesulfonate)), and uses [kernel density estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation) to predict position of most likely causal SNP. Provides more accurate mapping than the original [CloudMap](https://www.ncbi.nlm.nih.gov/pubmed/23051646).

Currently only supports [*C. elegans*](https://en.wikipedia.org/wiki/Caenorhabditis_elegans) genome.

<br>
#### Inputs:
1) A list of [GATK-called homozygous SNPs](https://software.broadinstitute.org/gatk/) from pooled F2 strains with known Hawaiian mapping strain SNPs subtracted. *< Bulked Segregant SNPs with mapping strain SNPs subtracted >*

2) A list of [GATK-called allele ratios](https://software.broadinstitute.org/gatk/) (e.g. [https://galaxyproject.org/](https://galaxyproject.org/)) from pooled F2 strains at mapping strain SNP positions only. For example, if crossing an [EMS](https://en.wikipedia.org/wiki/Ethyl_methanesulfonate) mutagenized *C. elegans* strain to a Hawaiian mapping strain (CB4856), you would call SNPs at Hawaiian SNP positions only. *< Allele ratios at mapping strain SNP positions >*




<br>
#### Output:

A map of linked chromosomal regions with color-coded variants of interest.

**gray**: Non-EMS derived parental SNPs (e.g. due to genetic drift with respect to N2)

**blue**: SNPs likely caused by [EMS](https://en.wikipedia.org/wiki/Ethyl_methanesulfonate).

**red**: For the purposes of paper figures, can plot the known causal variant in the context of other linked SNPs.

Linked regions with 3 or more SNPs have a predicted position of most likely causal SNP using [kernel density estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation). 

<br>
#### Usage:
```python
python cloudmap2.py < Bulked Segregant SNPs with mapping strain SNPs subtracted > < Allele ratios at mapping strain SNP positions > -o <output pdf>
```

<br>
#### Simple Demo:
```python
python cloudmap2.py Galaxy40-\[ot785HA_WS245_Homozygous_variants_SubtractedHobertHawaiianHomozygousAndHeterozygous\].vcf Galaxy45-\[ot785HA_VariantsAtHighQualityHAPositions_MQ30_WS245_DP_0_BiallelicPositions_SNPsOnly\].vcf -o ot785.pdf
```