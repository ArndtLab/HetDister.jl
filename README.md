# DemoInfer

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ArndtLab.github.io/DemoInfer.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ArndtLab.github.io/DemoInfer.jl/dev/)
[![Build Status](https://github.com/ArndtLab/DemoInfer.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ArndtLab/DemoInfer.jl/actions/workflows/CI.yml?query=branch%3Amain)

Module to run demographic inference on diploid genomes.
The jupyter notebook at `example/notebook.ipynb` contains an overview of 
the main features one may be interested in.

To run the package, first install julia ([here](https://julialang.org/downloads/)).
To create a local environment with the package `cd`into your work directory and 
launch julia, then:
```julia
using Pkg; Pkg.activate(".")
```
```julia
Pkg.add("DemoInfer","HistogramBinnings","PyPlot","MLDs","CSV","DataFrames")
using DemoInfer, HistogramBinnings, PyPlot, MLDs, CSV, DataFrames
```

The tool require three inputs: a vector of IBS segments lengths, a mutation rate and 
a recombination rate (both per bp per generation).
```julia
data = CSV.read("path", header=0, DataFrame; delim="\t")
h_obs = HistogramBinnings.Histogram(LogEdgeVector(lo = 30, hi = 1_000_000, nbins = 200));
append!(h_obs, data[:,1])
```

Have a look at the data:
```julia
plot_hist(ĥ; s=3, color="red", label="observed")
legend()
xlim(1e0,1e7)
xscale("log")
yscale("log")
```

Set a value for the rates and run the inference:
```julia
mu = 2.36e-8
rho = 1e-8
res, chains = corrected_fit(ĥ, 3, 2.36e-8, 1e-8, start = 1, iters = 10, final_factor=100)
```

Plot residuals and demographic profile:
```julia
plot_residuals(ĥ, res[3], 2.36e-8, 1e-8; s = 4)
xscale("log")
```
```julia
_, ax = subplots(figsize=(7, 5))
plot_demography(res[3], ax, id="simulation")
legend()
```

See the docs and the example notebook for more details.