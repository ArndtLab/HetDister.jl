module DemoInfer

using PyPlot
using StatsBase, Distributions, HistogramBinnings
using PopSimIBX
using LinearAlgebra, Statistics
using Turing, Optim
using StatsAPI
using Printf
using Logging
import DynamicPPL, ForwardDiff, Accessors
using MLDs

include("mle_optimization.jl")
include("sequential_fit.jl")
include("simulate.jl")
include("corrections.jl")
include("plot_utils.jl")

end
