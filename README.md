# DemoInfer

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ArndtLab.github.io/DemoInfer.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ArndtLab.github.io/DemoInfer.jl/dev/)
[![Build Status](https://github.com/ArndtLab/DemoInfer.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ArndtLab/DemoInfer.jl/actions/workflows/CI.yml?query=branch%3Amain)

Module to run demographic inference on diploid genomes, under the assumption of panmixia (i.e. the inferred effective population size is half the inverse of the observed mean coalescence rate).
See [this](https://github.com/ArndtLab/DemoInferDemo) repo for a demo of how to use it.

To run the package, first install julia ([here](https://julialang.org/downloads/)).
To create a local environment with the package `cd` into your work directory and 
launch julia, then:
```julia
using Pkg; Pkg.activate(".")
```
```julia
Pkg.Registry.add(RegistrySpec(url = "https://github.com/ArndtLab/JuliaRegistry.git"))
```
```julia
Pkg.add("DemoInfer","HistogramBinnings","CSV","DataFrames")
using DemoInfer, HistogramBinnings, CSV, DataFrames
```

The tool require three inputs: a (binned) vector of IBS segments lengths, a mutation rate and 
a recombination rate (both per bp per generation).
```julia
data = CSV.read("path", header=0, DataFrame; delim="\t")
h_obs = HistogramBinnings.Histogram(LogEdgeVector(lo = 30, hi = 1_000_000, nbins = 200));
append!(h_obs, data[:,1])
```
You can read a vector from a `.csv` file (first line) and the create an histogram with it.

Set a value for the rates and run the inference:
```julia
mu = 2.36e-8
rho = 1e-8
Ltot = sum(data[:,1])
nepochs = 3
res, chains = fit(h_obs, nepochs, mu, rho, Ltot)
```

See the docs for more details.
