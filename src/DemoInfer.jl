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
using Random
using Base.Threads

using Logging
logger = ConsoleLogger(stdout, Logging.Error)
Base.global_logger(logger)

include("fitresult.jl")
include("mle_optimization.jl")
include("sequential_fit.jl")
include("simulate.jl")
include("corrections.jl")

export get_sim!,
    pre_fit, demoinfer, compare_models, estimate_nepochs,
    get_para, evd, sds, pop_sizes, durations, get_chain,
    compute_residuals,
    FitResult,
    Flat, Lin, Sq


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

function compute_residuals(h::Histogram, mu::Float64, TN::Vector)
    w_th = integral_ws(h.edges[1].edges, mu, TN)
    # w_ = zeros(length(h.weights))
    # w_ .= h.weights
    # when the observation is zero, we infer the rate of the Poisson process from the
    # neighbouring bins epxloiting the relation between them given by the model
    # w_[h.weights .== 0] .= w_th[h.weights .== 0]
    # residuals = (h.weights - w_th) ./ sqrt.(w_)
    residuals = (h.weights - w_th) ./ sqrt.(w_th) # probably better to use the "population" variance
    @assert all(isfinite.(residuals))
    return residuals
end

function compute_residuals(h1::Histogram, h2::Histogram; fc1 = 1.0, fc2 = 1.0)
    @assert length(h1.weights) == length(h2.weights)
    # when both observations are zero the residual is zero
    w_ = h1.weights / fc1 .+ h2.weights / fc2
    residuals = (h1.weights / fc1 - h2.weights / fc2) ./ sqrt.(w_)
    residuals[w_ .== 0] .= 0
    @assert all(isfinite.(residuals))
    return residuals
end

end
