function tcondr(r::Number, para::Vector{T}, mu::Number) where T<:Number
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
            t = tcondr(r[j], prev_para, mu)
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

function perturb_fit!(f::FitResult, h::Histogram, mu::Float64, fop::FitOptions;
    by_pass::Bool = false
)
    if isinf(evd(f)) || any(f.opt.at_lboundary) || any(f.opt.at_uboundary[2:end]) || !f.converged
        factors = mapreduce( i->fill(i, 10), vcat, [0.001, 0.01, 0.1, 0.5, 0.5, 0.9, 2] )
        for fct in factors
            perturbations = Perturbation[]
            for i in eachindex(f.para)
                if f.opt.at_lboundary[i] || (f.opt.at_uboundary[i] && i > 1)
                    push!(perturbations, Perturbation(fct, i))
                end
            end
            setinit!(fop, f.para)
            fop.perturbations = perturbations
            f = fit_model_epochs(h, mu, fop)
            fop.perturbations = Perturbation[]
            if (evd(f) != Inf)
                if by_pass
                    break
                elseif !any(f.opt.at_lboundary[1:end-2])
                    break
                end
            end
        end
    end
    return f
end

"""
    pre_fit(h::Histogram, nfits::Int, mu::Float64, Ltot::Number; kwargs...)

Preliminarily fit `h` with an approximate model of piece-wise constant `nepochs`.

The mutation rate `mu` is assumed to be per base pairs per generation
and the total length of the genome `Ltot` is in base pairs. The fit
approximate the histogram with more and more epochs up to `nepochs`, so
the result is a vector of `FitResult`, one for each number of epochs.

See also [`FitResult`](@ref).

# Arguments
- `Tlow::Int=10`: The lower bound for the duration of epochs.
- `Nlow::Int=10`, `Nupp::Int=100000`: The lower and upper bounds for the population sizes.
- `smallest_segment::Int=30`: The smallest segment size to consider for the optimization,
same as in [`demoinfer`](@ref).
"""
function pre_fit(h::Histogram, nfits::Int, mu::Float64, Ltot::Number;
    Tlow::Number=10, Tupp::Number=1e7, Nlow::Number=10, Nupp::Number=1e8,
    smallest_segment::Int = 1,
    require_convergence::Bool = true,
    force::Bool = false
)
    fits = Vector{FitResult}(undef, nfits)
    N0 = sum(h.weights) / Ltot / 4mu
    for i in 1:nfits
        fop = FitOptions(Ltot; nepochs = i, Tlow, Tupp, Nlow, Nupp)
        if i == 1
            f = fit_model_epochs(h, mu, fop)
        else
            ts = timesplitter(h, mu, fits[i-1].para; smallest_segment)
            if iszero(ts)
                @info "pre_fit: no split found, epoch $i"
                if !force
                    return fits
                else
                    r = midpoints(h.edges[1])
                    append!(ts, tcondr(rand(r), fits[i-1].para, mu))
                    append!(ts, tcondr(rand(r), fits[i-1].para, mu))
                end
            end
            filter!(t->t!=0, ts)
            fs = Vector{FitResult}(undef, length(ts))
            fops = Vector{FitOptions}(undef, length(ts))
            for j in eachindex(fops)
                fops[j] = fop
            end
            @threads for j in eachindex(ts)
                init = get_para(fits[i-1])
                epochfinder!(init, N0, ts[j], fops[j])
                setinit!(fops[j], init)
                f = fit_model_epochs(h, mu, fops[j])
                if !f.converged
                    f = perturb_fit!(f, h, mu, fops[j])
                end
                fs[j] = f
            end
            conv = filter(f->f.converged, fs)
            if isempty(conv)
                # all possible splits did not converge, 
                # perturbing the best
                conv = fs
                @debug "pre_fit: no converged fits, epoch $i"
            end
            lps = map(f->f.lp, conv)
            f = fs[argmax(lps)]
            f = perturb_fit!(f, h, mu, fop)
            if require_convergence && !f.converged
                @info "pre_fit: not converged, epoch $i"
                return fits
            end
        end

        if any(isnan.(f.para))
            @error "NaN parameters" f.para
        end

        fits[i] = f
    end
    return fits
end