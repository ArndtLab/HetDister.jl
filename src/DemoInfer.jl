module DemoInfer

using StatsBase, Distributions, HistogramBinnings
using PopSimIBX
using LinearAlgebra, Statistics
using Turing, Optim
using StatsAPI
using Printf
using DynamicPPL, ForwardDiff, Accessors
using MLDs
using Random
using Base.Threads
using Logging

include("utils.jl")
include("mle_optimization.jl")
include("sequential_fit.jl")
include("simulate.jl")
include("corrections.jl")

export get_sim!,
    pre_fit, demoinfer, compare_models,
    get_para, evd, sds, pop_sizes, durations, get_chain,
    compute_residuals,
    adapt_histogram,
    FitResult, FitOptions,
    Flat, Lin, Sq


function integral_ws(edges::AbstractVector{<:Real}, mu::Float64, TN::Vector)
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
    compute_residuals(h::Histogram, mu::Float64, TN::Vector)

Compute the residuals between the observed and expected weights.
"""
function compute_residuals(h::Histogram, mu::Float64, TN::Vector)
    w_th = integral_ws(h.edges[1], mu, TN)
    residuals = (h.weights - w_th) ./ sqrt.(w_th)
    @assert all(isfinite.(residuals))
    return residuals
end

"""
    compute_residuals(h1::Histogram, h2::Histogram)

When two histograms are given the std error of the difference is
used for normalization.
"""
function compute_residuals(h1::Histogram, h2::Histogram; fc1 = 1.0, fc2 = 1.0)
    @assert length(h1.weights) == length(h2.weights)
    # when both observations are zero the residual is zero
    w_ = h1.weights / fc1 .+ h2.weights / fc2
    residuals = (h1.weights / fc1 - h2.weights / fc2) ./ sqrt.(w_)
    residuals[w_ .== 0] .= 0
    @assert all(isfinite.(residuals))
    return residuals
end

"""
    adapt_histogram(segments::AbstractVector{<:Integer}; lo::Int=1, hi::Int=50_000_000, nbins::Int=200)

Build an histogram from `segments` logbinned between `lo` and `hi`
with `nbins` bins (see `HistogramBinnings.jl`).

The upper limit is adapted to the maximum observed length, so default value
is on purpose high.
"""
function adapt_histogram(segments::AbstractVector{<:Integer}; lo::Int=1, hi::Int=50_000_000, nbins::Int=200)
    h_obs = Histogram(LogEdgeVector(;lo, hi, nbins))
    append!(h_obs, segments)
    l = findlast(h_obs.weights .> 0)
    while h_obs.edges[1].edges[l+1] + 1 < hi
        hi = h_obs.edges[1].edges[l+1]
        h_obs = Histogram(LogEdgeVector(;lo, hi, nbins))
        append!(h_obs, segments)
        l = findlast(h_obs.weights .> 0)
    end
    edges = h_obs.edges[1].edges
    T = eltype(edges)
    nedges = T[]
    weights = h_obs.weights
    counter = 0
    for i in eachindex(weights)
        if i == 1
            push!(nedges, edges[i])
        elseif weights[i] == 0
            # record row of zeros
            counter += 1
        elseif counter > 0
            # enter here only when is not zero following a zero
            hi = edges[i+1]
            lo = edges[i-counter-1]
            mid = floor(sqrt(lo * hi))
            push!(nedges, mid)
            counter = 0
        elseif counter == 0
            # enter here when non zero following non zero
            push!(nedges, edges[i])
        end
    end
    push!(nedges, edges[end])
    h_obs = Histogram(LogEdgeVector(nedges))
    append!(h_obs, segments)
    return h_obs
end

"""
    compare_mlds(segs1::Vector{Int}, segs2::Vector{Int}; lo = 1, hi = 1_000_000, nbins = 200)

Build two histograms from `segs1` and `segs2`, rescale them for number and
average heterozygosity and return four vectors containing respectively
common midpoints for bins, the two rescaled weights and variances of
the difference between weights.
"""
function compare_mlds(segs1::AbstractVector{<:Integer}, segs2::AbstractVector{<:Integer}; lo = 1, hi = 1_000_000, nbins = 200)
    # 1 is the target lattice, i.e. with biggest theta
    segs1_ = copy(segs1)
    segs2_ = copy(segs2)
    theta1 = 1/mean(segs1_)
    theta2 = 1/mean(segs2_)
    swap = false
    if theta1 < theta2
        temp = copy(segs1_)
        segs1_ = copy(segs2_)
        segs2_ = temp
        theta1, theta2 = theta2, theta1
        swap = true
    end
    h1 = HistogramBinnings.Histogram(LogEdgeVector(lo = lo, hi = hi, nbins = nbins))
    append!(h1, segs1_)
    h2 = HistogramBinnings.Histogram(LogEdgeVector(lo = lo, hi = hi, nbins = nbins))
    append!(h2, segs2_)
    edges1 = h1.edges[1].edges * theta1
    edges2 = h2.edges[1].edges * theta2
    tw = zeros(Float64, length(h1.weights))
    w2 = h2.weights
    factor = length(segs1_) / length(segs2_)
    t = 1
    f = 1
    while t < length(edges1) && f < length(edges2)
        st, en = edges1[t], edges1[t+1]
        width = edges2[f+1] - edges2[f]
        if st <= edges2[f] < edges2[f+1] < en
            tw[t] += w2[f]
            f += 1
        elseif st <= edges2[f] < en < edges2[f+1]
            tw[t] += w2[f] * (en - edges2[f]) / width
            t += 1
        elseif edges2[f] < st <= edges2[f+1] < en
            if f == 1
                tw[t] += w2[f]
            else
                tw[t] += w2[f] * (edges2[f+1] - st) / width
            end
            f += 1
        elseif edges2[f] <= st < en < edges2[f+1]
            tw[t] += w2[f] * (en - st) / width
            t += 1
        else
            if edges2[f+1] < st && t == 1
                tw[t] += w2[f]
                f += 1
            else
                @error "disjoint bins"
            end
        end
    end
    
    rs = midpoints(h1.edges[1]) * theta1
    sigmasq = h1.weights .+ tw * factor^2
    if swap
        return rs, tw * factor, h1.weights, sigmasq
    else
        return rs, h1.weights, tw * factor, sigmasq
    end
end

end
