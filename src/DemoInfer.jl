module DemoInfer

using StatsBase, Distributions, HistogramBinnings
using PopSimIBX
using LinearAlgebra, Statistics
using Turing, Optim
using StatsAPI
using Printf
using Logging
import DynamicPPL, ForwardDiff, Accessors
using MLDs

using Logging
logger = ConsoleLogger(stdout, Logging.Error)
Base.global_logger(logger)

include("fitresult.jl")
include("mle_optimization.jl")
include("sequential_fit.jl")
include("simulate.jl")
include("corrections.jl")

export get_sim!,
    pre_fit, fit, compare_models, estimate_nepochs,
    get_para, evd, sds, pop_sizes, durations, get_chain,
    FitResult


function integral_ws(edges::Vector{T}, mu::Float64, TN::Vector) where {T <: Number}
    a = 0.5
    last_hid_I = hid_integral(TN, mu, edges[1] - a)
    weights = Vector{Float64}(undef, length(edges)-1)
    for i in eachindex(edges[1:end-1])
        @inbounds this_hid_I = hid_integral(TN, mu, edges[i+1] - a)
        weights[i] = this_hid_I - last_hid_I
        last_hid_I = this_hid_I
    end
    weights
end

end
