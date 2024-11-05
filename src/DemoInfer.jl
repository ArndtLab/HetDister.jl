module DemoInfer

using PyPlot
pushfirst!(PyPlot.pyimport("sys")."path", "")
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

export get_sim!, sequential_fit, corrected_fit, get_evidence, get_sds,
    plot_demography, plot_hist, plot_residuals, plot_naive_residuals,
    xy

end
