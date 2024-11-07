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

include("mle_optimization.jl")
include("sequential_fit.jl")
include("simulate.jl")
include("corrections.jl")

export get_sim!,
    sequential_fit, corrected_fit,
    get_evidence, get_sds


function integral_weigths(edges::Vector{T}, mu::Float64, TN::Vector) where {T <: Number}
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

"""
    get_evidence(fit::FitResult)

Return the evidence of the fit.
"""
get_evidence(fit::FitResult) = fit.opt.evidence

"""
    get_sds(fit::FitResult)

Return the standard deviations of the parameters of the fit.
"""
get_sds(fit::FitResult) = fit.opt.stderrors

end
