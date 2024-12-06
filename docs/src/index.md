```@meta
CurrentModule = DemoInfer
```

# DemoInfer

Documentation for [DemoInfer](https://github.com/ArndtLab/DemoInfer.jl).

Module to run demographic inference on diploid genomes, under the assumption of panmixia (i.e. the inferred effective population size is half the inverse of the observed mean coalescence rate).
See [this](https://github.com/ArndtLab/DemoInferDemo) repo for a demo of how to use it.

## Data form of input and output

The genome needs to be SNP-called and the genomic distance between consecutive heterozygous sites needs to be computed. The input is then a vector containind such distances. Additionally, mutation and recombination rates need to be chosen and passed as input as well.

The demographic model underlying the inference is composed of a variable number of epochs and the population size is constant along each epoch.

The output is a vector of parameters in the form `[L, N0, T1, N1, T2, N2, ...]` where `L` is the genome length,
`N0` is the ancestral population size in the infinite past and the subsequent pairs $(T_i, N_i)$ are the duration and size of following epochs going from past to present. This format is referred to as `TN` vector throughout.

```@index
```

```@autodocs
Modules = [DemoInfer]
```
