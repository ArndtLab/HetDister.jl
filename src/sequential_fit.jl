function tcondr(r::Number, mu::Number)
    return [1 / (mu * r), 1 / (mu * (25r)^(1/1.3))]
end

function deviant(h::Histogram, mu::Float64, prev_para::Vector{T};
    frame::Number = 20,
    pos::Bool = true,
    threshold::Float64 = 0.,
    smallest_segment::Int = 1
) where {T <: Number}

    # find approximate time of positive (negative) deviation from previous fit
    r = midpoints(h.edges[1])
    residuals = compute_residuals(h, mu, prev_para)
    if !pos residuals = -residuals end

    divide = zeros(Int, length(residuals))
    divide[(residuals .> threshold) .& (r .>= smallest_segment)] .= 1
    divide[residuals .< threshold] .= 0
    j = 1
    while j < length(divide)
        z = 1
        while j+z <= length(divide) && divide[j+z] == divide[j]
            z += 1
        end
        if z <= frame
            divide[j:j+z-1] .= 0
        end
        j += z
    end
    found = zeros(1)
    for j in eachindex(divide[1:end-1])
        if divide[j] != divide[j+1]
            t = tcondr(r[j], mu)
            @debug "identified deviation " r[j]
            append!(found, t)
        end
    end
    return found
end

function timesplitter(h::Histogram, mu::Float64, prev_para::Vector{T};
    frame::Number = 20,
    threshold::Float64 = 0.,
    smallest_segment::Int = 1
) where {T <: Number}
    t1 = deviant(h, mu, prev_para; 
        frame, pos = true, threshold, smallest_segment
    )
    if iszero(t1)
        t1 = deviant(h, mu, prev_para;
            frame=frame/2, pos = true, threshold, smallest_segment
        )
    end
    if iszero(t1)
        t1 = deviant(h, mu, prev_para;
            frame=frame/4, pos = true, threshold, smallest_segment
        )
    end
    t2 = deviant(h, mu, prev_para;
        frame, pos = false, threshold, smallest_segment
    )
    if iszero(t2)
        t2 = deviant(h, mu, prev_para;
            frame=frame/2, pos = false, threshold, smallest_segment
        )
    end
    if iszero(t2)
        t2 = deviant(h, mu, prev_para;
            frame=frame/4, pos = false, threshold, smallest_segment
        )
    end
    @debug "deviant results " t1 t2
    return vcat(t1, t2)
end

function epochfinder!(init::Vector{T}, N0, t, fop::FitOptions) where {T <: Number}
    # these are the absolute times separating epochs
    # ordered from ancient to recent
    ts = reverse(pushfirst!(cumsum(init[end-1:-2:3]),0))
    split_epoch = findfirst(ts .< t)
    isnothing(split_epoch) && (split_epoch = 1)

    if split_epoch == 1
        newT = t - ts[1]
        newT = max(newT, 1000)
        newN = N0
        insert!(init, 3, newN)
        insert!(init, 3, newT)
    else
        newT1 = ts[split_epoch-1] - t
        newT1 = max(newT1, 20)
        newT2 = t - ts[split_epoch]
        newT2 = max(newT2, 20)
        newN = init[2split_epoch] * 0.99
        if newN > (fop.upp[2split_epoch] / 1.01)
            newN = N0
        end
        init[2split_epoch-1] = newT1
        insert!(init, 2split_epoch, newT2)
        insert!(init, 2split_epoch, newN)
    end
    return init
end

function perturb_fit!(f::FitResult, fop::FitOptions, h::Histogram, mu::Float64;
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
            f = fit_model_epochs!(fop, h, mu)
            if !isinf(evd(f)) & f.converged
                if by_pass
                    break
                elseif !any(f.opt.at_lboundary[1:end-2])
                    break
                end
            end
        end
    end
    if f.lp < f_.lp
        return f_
    end
    return f
end

"""
    pre_fit(h::Histogram, nfits::Int, mu::Float64, Ltot::Number; require_convergence=true)
    pre_fit!(fop::FitOptions, h::Histogram, nfits::Int, mu::Float64; require_convergence=true)

Preliminarily fit `h` with an approximate model of piece-wise constant 
epochs for each number of epochs from 1 to `nfits`.

If given the total length of the genome `Ltot` it initialize the fit 
options to default. See [`FitOptions`](@ref) for how to specify them.
Otherwise it modifies `fop` in place to adapt it to all the requested
epochs.
The mutation rate `mu` is assumed to be per base pairs per generation
and the total length of the genome `Ltot` is in base pairs. Return a 
vector of `FitResult`, one for each number of epochs,
see also [`FitResult`](@ref).
"""
function pre_fit(h::Histogram{T,1,E}, nfits::Int, mu::Float64, Ltot::Number; 
    require_convergence::Bool = true
) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    fop = FitOptions(Ltot)
    return pre_fit!(fop, h, nfits, mu; require_convergence)
end

function pre_fit!(fop::FitOptions, h::Histogram{T,1,E}, nfits::Int, mu::Float64;
    require_convergence::Bool = true
) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    fits = FitResult[]
    N0 = sum(h.weights) / fop.Ltot / 4mu
    @assert nfits > 0 "number of fits has to be strictly positive"
    for i in 1:nfits
        setnepochs!(fop, i)
        if i == 1
            f = fit_model_epochs!(fop, h, mu)
        else
            ts = timesplitter(h, mu, fits[i-1].para; fop.smallest_segment)
            if iszero(ts)
                @info "pre_fit: no split found, epoch $i"
                if !fop.force
                    return fits
                else
                    r = midpoints(h.edges[1])
                    append!(ts, tcondr(rand(r), mu))
                    append!(ts, tcondr(rand(r), mu))
                end
            end
            filter!(t->t!=0, ts)
            sort!(ts)
            unique!(ts)
            maxnts_ = min(fop.maxnts, length(ts))
            ts = ts[range(start=1, stop=length(ts), step=length(ts)÷maxnts_)]
            fs = Vector{FitResult}(undef, length(ts))
            fops = Vector{FitOptions}(undef, length(ts))
            for j in eachindex(fops)
                fops[j] = deepcopy(fop)
            end
            @threads for j in eachindex(ts)
                init = get_para(fits[i-1])
                epochfinder!(init, N0, ts[j], fops[j])
                setinit!(fops[j], init)
                f = fit_model_epochs!(fops[j], h, mu)
                f = perturb_fit!(f, fops[j], h, mu; by_pass=true)
                fs[j] = f
            end
            lps = map(f->f.lp, fs)
            f = fs[argmax(lps)]
            @debug "best " ts[argmax(lps)] f.lp f.converged
            f = perturb_fit!(f, fop, h, mu)
            if require_convergence && !f.converged
                @info "pre_fit: not converged, epoch $i"
                return fits
            end
        end

        if any(isnan.(f.para))
            @error "NaN parameters" f.para
        end

        push!(fits, f)
    end
    return fits
end