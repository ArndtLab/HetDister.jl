"""
    demoinfer(segments::AbstractVector{<:Integer}, epochs::Int, mu::Float64, rho::Float64; kwargs...)

Make an histogram with IBS `segments` and infer demographic histories with
piece-wise constant epochs where the number of epochs is smaller or equal
to `epochs`.

Return a vector of `FitResult` of length smaller or equal to `epochs`, 
see [`FitResult`](@ref), `mu` and `rho` are respectively the mutation 
and recombination rates per base pair per generation.

# Optional Arguments
- `fop::FitOptions = FitOptions(sum(segments), mu, rho)`: the fit options, see [`FitOptions`](@ref).
- `lo::Int=1`: The lowest segment length to be considered in the histogram
- `hi::Int=50_000_000`: The highest segment length to be considered in the histogram
- `nbins::Int=400`: The number of bins to use in the histogram
- `iters::Int=10`: The number of iterations to perform. It might converge earlier
"""
function demoinfer(segments::AbstractVector{<:Integer}, epochs::Int, mu::Float64, rho::Float64;
    fop::FitOptions = FitOptions(sum(segments), mu, rho),
    lo::Int = 1, hi::Int = 50_000_000, nbins::Int = 400,
    kwargs...
)
    h = adapt_histogram(segments; lo, hi, nbins)
    if sum(segments) != fop.Ltot
        @warn "inconsistent Ltot and segments, taking sum(segments)"
        fop.Ltot = sum(segments)
    end
    return demoinfer(h, epochs, fop; kwargs...)
end

"""
    demoinfer_(h, epochrange, mu, rho, Ltot; kwargs...)

Infer demographic histories with piece-wise constant epochs
where the number of epochs is in `epochrange`. Takes a histogram
as input and the total length of the IBS segments.

It is much lighter to distribute the histogram than the vector of segments
which may also be streamed directly from disk into the histogram.
"""
function demoinfer(h_obs::Histogram{T,1,E}, epochs::Int, fop::FitOptions;
    iters::Int = 10
) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    @assert !isempty(h_obs.weights) "histogram is empty"
    @assert epochs > 0 "epochrange has to be strictly positive"

    ho_mod = Histogram(h_obs.edges)

    rs = midpoints(h_obs.edges[1])
    bag = IntegralArrays(fop.order, fop.ndt, length(rs), Val{2epochs})

    chain = []
    corrections = []

    fits = pre_fit!(fop, h_obs, epochs; require_convergence = false)
    f = compare_models(fits)
    init = get_para(f)
    for iter in 1:iters
        weightsnaive = integral_ws(h_obs.edges[1], fop.mu, init)
        mldsmcp!(bag, 1:fop.order, rs, h_obs.edges[1].edges, fop.mu, fop.rho, init)

        ho_mod.weights .= h_obs.weights

        yth = get_tmp(bag.ys, eltype(init))
        corr = yth .* diff(h_obs.edges[1]) .- weightsnaive
        corr[1:4] .= 0.
        temp = ho_mod.weights .- corr
        temp .= round.(Int, temp)
        ho_mod.weights .= max.(temp, 0)

        fits = pre_fit!(fop, ho_mod, epochs; require_convergence = false)
        f = compare_models(fits)

        init = f.para
        push!(chain, fits)
        push!(corrections, corr)
        if iter > 1
            deltacorr = (corrections[iter] .- corrections[iter-1]) ./ corrections[iter-1]
            if mean(abs.(deltacorr)) < 0.05
                break
            end
        end
    end

    (;
        fits,
        chain,
        corrections
    )
end

"""
    compare_models(models)

Compare the models parameterized by `FitResult`s and return the best one.
Takes an iterable of `FitResult` as input.

### Theoretical explanation
TBD
"""
function compare_models(models)
    ms = copy(models)
    filter!(m->isfinite(evd(m)) && m.converged, ms)
    if isempty(ms)
        @warn "none of the models is meaningful"
        return nothing
    end
    best = 1
    lp = ms[1].lp
    ev = evd(ms[1])
    mono = true
    for i in eachindex(ms)
        if evd(ms[i]) > ev && ms[i].lp >= lp
            best = i
            lp = ms[i].lp
            ev = evd(ms[i])
        elseif ms[i].lp < lp && mono
            @warn """
                log-likelihood is not monotonic in the number of epochs.
                This means that at least one likelihood optimization
                has probably failed. You may want to change the fit options.
            """
            mono = false
        end
    end
    return ms[best]
end