function ramp(iter, mu, rho)
    # min(mu/5 * iter, rho)
    rho
end

"""
    demoinfer(segments::AbstractVector{<:Integer}, epochrange::AbstractRange{<:Integer}, mu::Float64, rho::Float64; kwargs...)

Make an histogram with IBS `segments` and infer demographic models with
piece-wise constant epochs where the number of epochs is `epochrange`.

Return a named tuple which contains the fields:
- `fits`: a vector of `FitResult` (see [`FitResult`](@ref))
- `chains`: a vector of vectors of `FitResult`, one for each iteration
  of the correction procedure, and one chain per model
- `corrections`: a vector of vectors of corrections, one for each iteration
  of the correction procedure, and one vector of corrections per model.
  Corrections are histogram counts, therefore they have the same shape.
- `h_obs`: the histogram of the observed segments
- `h_mods`: a vector of modified histograms, one for each model, with
  higher order corrections applied.
- `yth`: a vector of vectors of the expected weights, one for each model
- `deltas`: a vector of vectors of the maximum absolute difference between
  corrections in consecutive iterations, and for each model.
- `conv`: a vector of booleans, one for each model, indicating whether the
  maximum iterations were reached (false) or whether the procedure 
  converged before (true).


# Optional Arguments
- `fop::FitOptions = FitOptions(sum(segments), mu, rho)`: the fit options, see [`FitOptions`](@ref).
- `lo::Int=1`: The lowest segment length to be considered in the histogram
- `hi::Int=50_000_000`: The highest segment length to be considered in the histogram
- `nbins::Int=fop.ndt`: The number of bins to use in the histogram
- `iters::Int=20`: The number of iterations to perform. It might converge earlier
- `reltol::Float64=1e-2`: The relative tolerance to use for convergence,
  i.e. the maximum absolute difference between corrections in consecutive iterations.
- `corcut::Int=fop.locut-1`: The index of the last histogram bin to apply corrections to.
"""
function demoinfer(segments::AbstractVector{<:Integer}, epochrange::AbstractRange{<:Integer}, mu::Float64, rho::Float64;
    fop::FitOptions = FitOptions(sum(segments), length(segments), mu, rho),
    lo::Int = 1, hi::Int = 50_000_000, nbins::Int = fop.ndt,
    kwargs...
)
    h = adapt_histogram(segments; lo, hi, nbins)
    if sum(segments) != fop.Ltot
        @warn "inconsistent Ltot and segments, taking sum(segments)"
        fop.Ltot = sum(segments)
    end
    return demoinfer(h, epochrange, fop; kwargs...)
end

"""
    demoinfer(h::Histogram, epochrange, fop::FitOptions; iters=20, reltol=1e-2, corcut=fop.locut-1, finalize=false)
    demoinfer(h, epochs, fop; iters=20, reltol=1e-2, corcut=fop.locut-1, finalize=false)

Take an histogram of IBS segments, fit options, and infer demographic models with
piece-wise constant epochs where the number of epochs is `epochrange`.
Return a named tuple as above.

If `epochrange` is a integer, then it fits only the model with that number of epochs.
In this case the returned named tuple contains only one element per field, instead of a vector.
"""
function demoinfer(h_obs::Histogram{T,1,E}, epochrange::AbstractRange{<:Integer}, fop_::FitOptions;
    kwargs...
) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    @assert length(epochrange) > 0
    results = Vector{NamedTuple}(undef, length(epochrange))
    @threads for i in eachindex(epochrange)
        results[i] = demoinfer(h_obs, epochrange[i], fop_; kwargs...)
    end
    return (;
        fits = map(r->r.f, results),
        chains = map(r->r.chain, results),
        corrections = map(r->r.corrections, results),
        h_obs = results[1].h_obs,
        h_mods = map(r->r.h_mod, results),
        yth = map(r->r.yth, results),
        deltas = map(r->r.deltas, results),
        conv = map(r->r.conv, results)
    )
end

function demoinfer(h_obs::Histogram{T,1,E}, epochs::Int, fop_::FitOptions;
    iters::Int = 20, reltol::Float64 = 1e-2, corcut::Int = fop_.locut-1, 
    finalize::Bool = false
) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    @assert !isempty(h_obs.weights) "histogram is empty"
    @assert epochs > 0 "epochrange has to be strictly positive"

    h_mod = Histogram(h_obs.edges)

    fop = deepcopy(fop_)
    rs = midpoints(h_obs.edges[1])
    bag = IntegralArrays(fop.order, fop.ndt, length(rs), Val{2epochs})

    chain = []
    corrections = []
    deltas = [Inf]

    h_mod.weights .= h_obs.weights
    corr = zeros(Float64, length(h_obs.weights))
    f = nothing
    yth = nothing
    for iter in 1:iters
        fits = pre_fit!(fop, h_mod, epochs)
        f = fits[end]
        if f.nepochs != epochs
            push!(chain, f)
            break
        end
        init = get_para(f)
        push!(chain, f)
        push!(corrections, corr)
        if iter > 1
            deltacorr = corrections[iter] .- corrections[iter-1]
            delta = maximum(abs.(deltacorr))
            push!(deltas, delta)
            if delta < reltol
                break
            end
        end

        weightsnaive = integral_ws(h_obs.edges[1], fop.mu, init)
        rho = ramp(iter, fop.mu, fop.rho)
        mldsmcp!(bag, 1:fop.order, rs, h_obs.edges[1], fop.mu, rho, init)

        h_mod.weights .= h_obs.weights

        yth = get_tmp(bag.ys, eltype(init))
        corr = yth .* diff(h_obs.edges[1]) .- weightsnaive
        corr[1:corcut] .= 0.
        temp = h_mod.weights .- corr
        temp .= round.(Int, temp)
        h_mod.weights .= max.(temp, 0)
        @assert all(isfinite, h_mod.weights)
        @assert all(!isnan, h_mod.weights)
    end

    conv = true
    if length(chain) == iters
        conv = false
        if finalize
            f = chain[argmin(deltas)]
            setOptimOptions!(fop; maxiters = 600, maxtime = 10000)
            setnaive!(fop, false)
            setnepochs!(fop, epochs)
            setinit!(fop, get_para(f))
            f = fit_model_epochs!(fop, h_obs)
        end
    end

    (;
        f,
        chain,
        corrections,
        h_obs,
        h_mod,
        yth,
        deltas,
        conv
    )
end

function correctestimate!(fop::FitOptions, fit::FitResult, h::Histogram)
    rs = midpoints(h.edges[1])
    bag = IntegralArrays(fop.order, fop.ndt, length(rs), Val{length(fit.para)}, 3)

    setnepochs!(fop, length(fit.para)÷2)
    setinit!(fop, fit.para)

    he = ForwardDiff.hessian(
        tn -> llsmcp!(bag, rs, h.edges[1].edges, h.weights, fop.mu, fop.rho, fop.locut, tn),
        get_para(fit)
    )
    return getFitResult(he, fit.para, fit.lp, fit.opt.optim_result, fop, h.weights, true)
end

"""
    compare_models(models[, mask])

Compare the models parameterized by `FitResult`s and return the best one.
Takes an iterable of `FitResult` as input and optionally a boolean mask
to reflect prior knowledge on models to discard.
"""
function compare_models(models, mask=trues(length(models)))
    ms = copy(models)
    ms = ms[mask]
    if isempty(ms)
        @warn "none of the models is meaningful"
        return nothing
    end
    best = 1
    lp = ms[1].lp
    ev = evd(ms[1])
    monotonic = true
    for i in eachindex(ms)
        if evd(ms[i]) > ev && ms[i].lp >= lp
            best = i
            lp = ms[i].lp
            ev = evd(ms[i])
        elseif ms[i].lp < lp && monotonic
            @warn """
                log-likelihood is not monotonic in the number of epochs.
                This means that at least one likelihood optimization
                has probably failed. See diagnostics.
            """
            monotonic = false
        end
    end
    if ms[best].converged == false
        @warn "the best model's optimization did not converge"
    end
    return ms[best]
end