```@meta
CurrentModule = HetDister
```

# HetDister

Documentation for [HetDister](https://github.com/ArndtLab/HetDister.jl).

Module to run demographic inference on diploid genomes, under the assumption of panmixia (i.e. the inferred effective population size is half the inverse of the observed mean coalescence rate).
See [this](https://github.com/ArndtLab/DemoInferDemo) repo for a demo of how to use it.

## Data form of input and output

The genome needs to be SNP-called and the genomic distance between consecutive heterozygous positions needs to be computed. Heterozygous positions are the ones with genotype 0/1 or 1/0 (Note that the phase is not important). The input is then a vector containind such distances. Additionally, mutation and recombination rates need to be chosen and passed as input as well.

For example, suppose you have a `.vcf` file with called variants you want to analyze. Then you may compute distances between heterozygous SNPs as follows:
```julia
using CSV
using DataFrames
using DataFramesMeta

f = "/myproject/myfavouritespecies.vcf"
df = CSV.read(f, DataFrame, 
    delim='\t', 
    comment="##",
    missingstring=[".", "NaN"],
    normalizenames=true,
    ntasks = 1,
    drop = [:INFO, :ID, :FILTER],
)

# remove homozygous variants
@chain df begin
    @rsubset! (:SampleName[1] == '1' && :SampleName[3] == '0') || (:SampleName[1] == '0' && :SampleName[3] == '1')
end

ils = df.POS[2:end] .- df.POS[1:end-1]
@assert all(ils .> 0)
```

The demographic model underlying the inference is composed of a variable number of epochs and the population size is constant along each epoch.

The output is a vector of parameters in the form `[L, N0, T1, N1, T2, N2, ...]` where `L` is the genome length,
`N0` is the ancestral population size in the furthermost epoch and extending to the infinite past, the subsequent pairs $(T_i, N_i)$ are the duration and size of following epochs going from past to present. This format is referred to as `TN` vector throughout.

```@index
```

```@autodocs
Modules = [HetDister, HetDister.Spectra]
```
