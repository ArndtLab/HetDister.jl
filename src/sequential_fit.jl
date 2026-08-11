function tcondr(r::Number, mu::Number)
    return 1 / (mu * r)
end

function timesplitter(h::Histogram, prev_para::Vector{T}, fop::FitOptions;
    frame::Number = 5
) where {T <: Number}

    # find approximate time of positive (negative) deviation from previous fit
    r = midpoints(h.edges[1])
    residuals = compute_residuals(h, fop.mu, fop.rho, prev_para, naive = isnaive(fop))

    found = zeros(1)
    j = fop.locut
    while j < length(residuals)
        z = j + 1
        while z < length(residuals) && residuals[j] * residuals[z] > 0
            z += 1
        end
        if z - j >= frame || (j == fop.locut) || (z == length(residuals))
            t1 = tcondr(r[j], fop.mu)
            t2 = tcondr(r[z], fop.mu)
            @debug "identified deviation " r[j] r[z]
            append!(found, t1, t2)
        end
        j = z
    end
    @debug "time splits results " found
    return found
end

function epochfinder!(init::Vector{T}, t, fop::FitOptions) where {T <: Number}
    nep = fop.nepochs - 1 # previous model
    # these are the absolute times of epochs changes
    # ordered from ancient to recent
    ts = [Spectra.getts(init,i) for i in nep:-1:1]
    split_epoch = findfirst(ts .< t)
    isnothing(split_epoch) && (split_epoch = 1)

    if split_epoch == 1
        newT = t - ts[1]
        # the floor keeps the new duration within the prior support of the
        # oldest T; when only the Ns are fitted the Ts are held fixed at the
        # proposed split, so it is used as is
        isonlyN(fop) || (newT = max(newT, 1000))
        newN = init[2]
        insert!(init, 3, newN)
        insert!(init, 3, newT)
    else
        newT1 = ts[split_epoch-1] - t
        newT2 = t - ts[split_epoch]
        newN = init[2split_epoch]
        init[2split_epoch-1] = newT1
        insert!(init, 2split_epoch, newT2)
        insert!(init, 2split_epoch, newN)
    end
    return init
end

"""
    midlineagetime(t0::Real, t1::Real, TN::AbstractVector{<:Real}, rho::Real; ngrid::Int = 1000)

Find the time `t` in `(t0, t1)` such that the number of lineages coalescing in
`[t0, t]` is roughly equal to the number of lineages coalescing in `[t, t1]`,
for the demographic scenario `TN` and recombination rate `rho`, see
[`cumulative_lineages`](@ref). All times are absolute, in generations before
present.

Return `0` when no such time is resolved on the logarithmic grid of `ngrid`
points used for the search.
"""
function midlineagetime(t0::Real, t1::Real, TN::AbstractVector{<:Real}, rho::Real;
    ngrid::Int = 1000
)
    t1 <= t0 && return 0.0
    cum0 = cumulative_lineages(t0, TN, rho)
    nlin = (cumulative_lineages(t1, TN, rho) - cum0) / 2
    nlin <= 0 && return 0.0
    for t in logrange(max(1, t0), t1, ngrid)
        if cumulative_lineages(t, TN, rho) - cum0 >= nlin
            return t
        end
    end
    return 0.0
end

function perturb_fit!(f::FitResult, fop::FitOptions, h::Histogram;
    by_pass::Bool = false
)
    f_ = deepcopy(f)
    reset_perturb!(fop)
    set_perturb!(fop, f)
    if any(fop.perturb)
        pinit = PInit(fop)
        for fct in fop.delta.factors
            next!(fop.delta)
            setinit!(fop, f.para)
            set_perturb!(fop, f)
            setinit!(fop, pinit)
            f = fit_model_epochs!(fop, h; stats = false)
            if f.converged
                if by_pass
                    break
                elseif !any(f.opt.at_lboundary[1:end-2])
                    break
                end
            end
        end
    end
    if f.lp < f_.lp || any(isnan, f.para)
        return f_
    end
    return f
end

"""
    pre_fit!(fop::FitOptions, h::Histogram, nfits)

Preliminarily fit `h` with an approximate model of piece-wise constant 
epochs for each number of epochs from 1 to `nfits`.

See [`FitOptions`](@ref) for how to specify them.
It modifies `fop` in place to adapt it to all the requested
epochs.
Return a vector of `FitResult`, one for each number of epochs,
see also [`FitResult`](@ref).
"""
function pre_fit!(fop::FitOptions, h::Histogram{T,1,E}, nfits::Int
) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    fits = FitResult[]
    @assert nfits > 0 "number of fits has to be strictly positive"
    for i in 1:nfits
        setnepochs!(fop, i)
        if i == 1
            f = fit_model_epochs!(fop, h)
        else
            ts = timesplitter(h, get_para(fits[i-1]), fop)
            if iszero(ts)
                @info "pre_fit: no split found, epoch $i"
                if !fop.force
                    return fits
                else
                    ts = [Spectra.getts(get_para(fits[i-1]), j) for j in 1:i-1]
                    push!(ts, 1e9)
                    ts[1] = 1
                    @debug ts
                    ts = sqrt.(ts[1:end-1] .* ts[2:end])
                end
            else
                filter!(t->t!=0, ts)
                push!(ts, 15.0)
                sort!(ts)
                unique!(ts)
                maxnts_ = min(fop.maxnts, length(ts))
                ts = ts[range(start=1, stop=length(ts), step=length(ts)÷maxnts_)]
            end
            fs = Vector{FitResult}(undef, length(ts))
            fops = Vector{FitOptions}(undef, length(ts))
            for j in eachindex(fops)
                fops[j] = deepcopy(fop)
            end
            @threads for j in eachindex(ts)
                init = get_para(fits[i-1])
                epochfinder!(init, ts[j], fops[j])
                setinit!(fops[j], init)
                f = fit_model_epochs!(fops[j], h; stats = false)
                fs[j] = f
            end
            lps = map(f->f.lp, fs)
            f = fs[argmax(lps)]
            @debug "best " ts[argmax(lps)] f.lp f.converged
            f = perturb_fit!(f, fop, h; by_pass=false)
            p = 1 .+ (rand(length(f.para)) .- 0.5) * 0.001
            setinit!(fop, get_para(f) .* p) # perturb slightly to avoid linesearch failure
            f = fit_model_epochs!(fop, h)
            if (f.lp < fits[i-1].lp) && f.converged
                @error "epoch $i ll not improved. Please report an issue"
            end
            @assert all(!isnan, f.para) """
                NaN parameters $(f.para)
                $(f.lp)
                $(f.opt.init)
                $(fop.upp)
                $(fs[argmax(lps)])
                $(f.para)
            """
        end
        push!(fits, f)
    end
    return fits
end

"""
    refine_model!(fop::FitOptions, h::Histogram, TN::AbstractVector{<:Real})

Iteratively refine the demographic model `TN` on the observed histogram `h` by
splitting its epochs, fitting only the population sizes, see [`fitNs!`](@ref).

In each round every epoch `[t0, t1)` of the current model proposes one new epoch
boundary, placed where half of the lineages coalescing within that epoch have
coalesced, see [`midlineagetime`](@ref). This gives as many candidate models as
there are epochs, each with one epoch more than the current one. They are all
fitted in parallel and the one with the highest evidence is adopted as the new
current model. Candidates whose log-likelihood is not better than the current
one are discarded: they contain the current model as a special case, so this
signals a failed optimization.

The refinement stops when the evidence stops increasing or when the number of
epochs reaches twice that of the input `TN`. Only the population sizes are ever
estimated: the total genome length and all epoch durations stay fixed at the
proposed values.

`fop` is modified in place to describe the returned model (`nepochs`, `init` and
`onlyN`). Return the final model as a `FitResult` with stats computed, see also
[`FitResult`](@ref).
"""
function refine_model!(fop::FitOptions, h::Histogram{T,1,E}, TN::AbstractVector{<:Real}
) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    nepochs0 = length(TN) ÷ 2
    @assert nepochs0 > 0 "TN has to contain at least one epoch"
    maxepochs = 2nepochs0
    tmax = 1e9 # the oldest epoch has no upper boundary

    setnepochs!(fop, nepochs0)
    setonlyN!(fop, true) # before setinit!, to keep the proposed L and Ts
    setinit!(fop, TN)
    best = fitNs!(fop, h)

    while best.nepochs < maxepochs
        cur = get_para(best)
        nep = best.nepochs
        # absolute times of epoch changes, from recent to ancient
        bounds = [Spectra.getts(cur, i) for i in 1:nep]
        ts = Float64[]
        for i in 1:nep
            t0 = bounds[i]
            t1 = i < nep ? bounds[i+1] : tmax
            t = midlineagetime(t0, t1, cur, fop.rho)
            if t0 + 1 < t < t1 - 1 && !(t in ts)
                push!(ts, t)
            end
        end
        if isempty(ts)
            @info "refine_model: no split found, $nep epochs"
            return best
        end
        @debug "refine_model: proposed splits " ts

        fs = Vector{FitResult}(undef, length(ts))
        fops = Vector{FitOptions}(undef, length(ts))
        for j in eachindex(ts)
            fops[j] = deepcopy(fop)
            setnepochs!(fops[j], nep + 1)
            setonlyN!(fops[j], true)
            init = get_para(best)
            epochfinder!(init, ts[j], fops[j])
            setinit!(fops[j], init)
        end
        @threads for j in eachindex(ts)
            fs[j] = fitNs!(fops[j], h)
        end

        kept = Int[]
        for j in eachindex(fs)
            if fs[j].lp < best.lp
                @error "refine_model: ll not improved splitting at $(ts[j]) with $nep epochs. Please report an issue"
            else
                push!(kept, j)
            end
        end
        isempty(kept) && return best
        b = kept[argmax(map(j -> evd(fs[j]), kept))]
        @debug "best " ts[b] fs[b].lp evd(fs[b]) fs[b].converged
        evd(fs[b]) <= evd(best) && return best

        best = fs[b]
        @assert all(!isnan, best.para) """
            NaN parameters $(best.para)
            $(best.lp)
            $(best.opt.init)
        """
        setnepochs!(fop, best.nepochs)
        setonlyN!(fop, true)
        setinit!(fop, get_para(best))
    end
    return best
end