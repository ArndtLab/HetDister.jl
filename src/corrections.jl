function fraction(mu, rho, n)
    mu/(mu+rho) * (rho/(mu+rho))^(n-1)
end

function ramp(iter, mu, rho)
    # min(mu/5 * iter, rho)
    rho
end

"""
    demoinfer(segments::AbstractVector{<:Integer}, epochrange::AbstractRange{<:Integer}, mu::Float64, rho::Float64; kwargs...)

Make an histogram with IBS `segments` and infer demographic histories with
piece-wise constant epochs where the number of epochs is `epochrange`.

Return a named tuple which contains a vector of `FitResult` in the field `fits`
(see [`FitResult`](@ref)).

# Optional Arguments
- `fop::FitOptions = FitOptions(sum(segments), mu, rho)`: the fit options, see [`FitOptions`](@ref).
- `lo::Int=1`: The lowest segment length to be considered in the histogram
- `hi::Int=50_000_000`: The highest segment length to be considered in the histogram
- `nbins::Int=400`: The number of bins to use in the histogram
- `iters::Int=10`: The number of iterations to perform. It might converge earlier
- `setorder::Bool=true`: the order at which the SMC' approximation is truncated will be set automatically according to the cutoff
- `cutoff=2e-5`: when `setorder` a fraction of segments smaller than `cutoff` will be ignored to set the order 
"""
function demoinfer(segments::AbstractVector{<:Integer}, epochrange::AbstractRange{<:Integer}, mu::Float64, rho::Float64;
    fop::FitOptions = FitOptions(sum(segments), mu, rho),
    lo::Int = 1, hi::Int = 50_000_000, nbins::Int = 400,
    setorder::Bool = true, cutoff = 2e-5,
    kwargs...
)
    h = adapt_histogram(segments; lo, hi, nbins)
    if sum(segments) != fop.Ltot
        @warn "inconsistent Ltot and segments, taking sum(segments)"
        fop.Ltot = sum(segments)
    end
    if setorder
        o = findfirst(map(i->fraction(fop.mu,fop.rho,i),1:30) .< cutoff)
        isnothing(o) && (o = 30)
        fop.order = o
        @info "setting order to $o"
    end
    return demoinfer(h, epochrange, fop; kwargs...)
end

"""
    demoinfer(h::Histogram, epochrange, fop::FitOptions; iters=15, reltol=1e-2, corcut=20, finalize=false)
    demoinfer(h, epochs, fop; iters=15, reltol=1e-2, corcut=20, finalize=false)

Take an histogram of IBS segments, fit options, and infer demographic histories with
piece-wise constant epochs where the number of epochs is `epochrange`.
See [`FitOptions`](@ref).
Return a named tuple which contains a vector of `FitResult` in the field `fits`
(see [`FitResult`](@ref)).

If `epochrange` is a integer, then it fits only the model with that number of epochs.
Return a named tuple with a `FitResult` object in the field `f`.
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
        yth = map(r->r.yth, results),
        deltas = map(r->r.deltas, results),
        conv = map(r->r.conv, results)
    )
end

function demoinfer(h_obs::Histogram{T,1,E}, epochs::Int, fop_::FitOptions;
    iters::Int = 20, reltol::Float64 = 1e-2, corcut::Int = 20, 
    finalize::Bool = false
) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    @assert !isempty(h_obs.weights) "histogram is empty"
    @assert epochs > 0 "epochrange has to be strictly positive"

    ho_mod = Histogram(h_obs.edges)

    fop = deepcopy(fop_)
    rs = midpoints(h_obs.edges[1])
    bag = IntegralArrays(fop.order, fop.ndt, length(rs), Val{2epochs})

    chain = []
    corrections = []
    deltas = [Inf]

    ho_mod.weights .= h_obs.weights
    corr = zeros(Float64, length(h_obs.weights))
    f = nothing
    yth = nothing
    for iter in 1:iters
        fits = pre_fit!(fop, ho_mod, epochs; require_convergence = false)
        f = fits[end]
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

        ho_mod.weights .= h_obs.weights

        yth = get_tmp(bag.ys, eltype(init))
        corr = yth .* diff(h_obs.edges[1]) .- weightsnaive
        corr[1:corcut] .= 0.
        temp = ho_mod.weights .- corr
        temp .= round.(Int, temp)
        ho_mod.weights .= max.(temp, 0)
        @assert all(isfinite, ho_mod.weights)
        @assert all(!isnan, ho_mod.weights)
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
    compare_models(models)

Compare the models parameterized by `FitResult`s and return the best one.
Takes an iterable of `FitResult` as input.

### Theoretical explanation
TBD
"""
function compare_models(models, flags=trues(length(models)))
    ms = copy(models)
    mask = map(eachindex(ms)) do i
        isfinite(evd(ms[i])) && ms[i].converged && flags[i]
    end
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
                has probably failed. You may want to change the fit options.
            """
            monotonic = false
        end
    end
    if !reduce(&, mask[1:best])
        @warn "a simpler model did not converge"
    end
    return ms[best]
end