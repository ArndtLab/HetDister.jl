function initializer(h::Histogram, mu::Float64, prev_para::Vector{T};
    frame::Number = 20, 
    pos::Bool = true,
    threshold::Float64 = 0.,
    smallest_segment::Int = 30,
    previoust = []
) where {T <: Number}

    # find approximate time of positive (negative) deviation from previous fit
    w_th = integral_ws(h.edges[1].edges, mu, prev_para)
    r = midpoints(h.edges[1])
    residuals = pos ? (h.weights - w_th) ./ sqrt.(h.weights) : (w_th - h.weights) ./ sqrt.(h.weights)
    residuals[h.weights .== 0] .= 0

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

    for j in eachindex(divide[1:end-1])
        if divide[j] == 0 && divide[j+1] == 1
            t = 1 / (mu * (25r[j])^(1/1.3))
            return t
            # if all(previoust .!= t)
            #     return t
            # end
        end
    end
    return 0.
end

# only Tupp is allowed
function bounds_strict(fit::FitResult)
    any(fit.opt.at_lboundary) || any(fit.opt.at_uboundary[2:2:end])
end

# Nupp and Tupp can be reached
function bounds_weak(fit::FitResult)
    any(fit.opt.at_lboundary)
end

function recombine(f::FitResult, init)
    recomb = map(f.opt.at_lboundary, f.opt.at_uboundary, f.para, init) do l, u, p, i
        (l || u) ? i : p
    end
    return recomb
end

function perturb_fit!(f::FitResult, h::Histogram, mu::Float64, fop::FitOptions;
    by_pass::Bool = true,
    isrnd::Bool = true
)
    if (evd(f) == Inf) || bounds_strict(f)
        if isrnd
            factors = mapreduce( i->fill(i, 10), vcat, [0.001, 0.01, 0.1, 0.5, 0.5, 0.9, 2] )
            # factors = mapreduce( i->fill(i, 10), vcat, [0.01, 0.1, 0.5, 0.9, 1, 2, 2] )
            for fct in factors
                perturbations = Perturbation[]
                for i in eachindex(f.para)
                    if f.opt.at_lboundary[i] || f.opt.at_uboundary[i]
                        push!(perturbations, Perturbation(isrnd, fct, i))
                    end
                end
                setinit!(fop, f.para)
                fop.perturbations = perturbations
                f = fit_model_epochs(h, mu, fop)
                if (evd(f) != Inf) && !(bounds_weak(f) && by_pass)
                    break
                end
            end
        else
            perturbations = Perturbation[]
            for i in eachindex(f.para)
                factor = 1.
                if f.opt.at_lboundary[i]
                    factor = 2.
                    push!(perturbations, Perturbation(isrnd, factor, i))
                elseif f.opt.at_uboundary[i] && (i % 2 == 0)
                    factor = 0.5
                    push!(perturbations, Perturbation(isrnd, factor, i))
                end
            end
            setinit!(fop, f.para)
            fop.perturbations = perturbations
            f = fit_model_epochs(h, mu, fop)
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
same as in [`fit`](@ref).
"""
function pre_fit(h::Histogram, nfits::Int, mu::Float64, Ltot::Number;
    Tlow::Int = 10,
    Nlow::Int = 10, Nupp::Int = 100000,
    smallest_segment::Int = 30
)
    frame = 20
    fits = Vector{FitResult}(undef, nfits)
    tprevious = [0.]
    for i in 1:nfits
        fop = FitOptions(Ltot; nepochs = i)
        if i == 1
            f = fit_model_epochs(h, mu, fop)
            push!(tprevious, 20 * f.para[2])
        else
            t1 = initializer(h, mu, fits[i-1].para; 
                frame, pos = true, smallest_segment, previoust=tprevious
            )
            if iszero(t1)
                t1 = initializer(h, mu, fits[i-1].para;
                    frame=frame/2, pos = true, smallest_segment, previoust=tprevious
                )
            end
            t2 = initializer(h, mu, fits[i-1].para;
                frame, pos = false, smallest_segment, previoust=tprevious
            )
            if iszero(t2)
                t2 = initializer(h, mu, fits[i-1].para;
                    frame=frame/2, pos = false, smallest_segment, previoust=tprevious
                )
            end
            t = min(t1, t2)
            if any(t .== tprevious)
                t = max(t1, t2)
            end
            # d1 = abs(t1 - tprevious[end])
            # d2 = abs(t2 - tprevious[end])
            # if d1 > d2
            #     t = t1
            # else
            #     t = t2
            # end
            if iszero(t1)
                t = t2
            elseif iszero(t2)
                t = t1
            end
            @debug "initializer results " t1 t2 t

            if iszero(t1) && iszero(t2)
                @info "no more meaningful epochs can be added (residuals exausted)"
                return fits
            end
            push!(tprevious, t)

            init = copy(fits[i-1].para)
            N0 = fits[1].para[2]
            t = min(t, 12N0) # permissive upper bound to 12 times the ancestral population size. It can anyway vary during the optimization
            ts = reverse(pushfirst!(cumsum(init[end:-2:3]),0))
            split_epoch = findfirst(ts .< t)

            if split_epoch == 1
                newT = t - ts[1]
                newT = max(newT, 1000)
                newN = init[2]
                insert!(init, 3, newN)
                insert!(init, 3, newT)
            else
                if iszero(t)
                    newT1 = (ts[end-1] - ts[end]) / 2
                    newT1 = max(newT1, 20)
                    newT2 = newT1
                    @assert(split_epoch == length(ts), "split_epoch: $split_epoch, length(ts): $(length(ts))")
                else
                    newT1 = ts[split_epoch-1] - t
                    newT1 = max(newT1, 20)
                    newT2 = t - ts[split_epoch]
                    newT2 = max(newT2, 20)
                end
                newN = init[2split_epoch]
                # newN > (Nupp / 1.01) && (newN = fits[1].para[2])
                init[2split_epoch-1] = newT1
                insert!(init, 2split_epoch, newT2)
                insert!(init, 2split_epoch, newN)
            end
            setinit!(fop, init)
            updateTupp!(fop, 10init[2])
            f = fit_model_epochs(h, mu, fop)
            # init = recombine(f, init)
            f = perturb_fit!(f, h, mu, fop)
            if !f.converged
                @info "pre_fit: not converged, epoch $i"
                return fits
            end
        end

        if any(isnan.(f.para))
            @info "pre_fit: nan para, epoch $i" f.para
            return fits
        end

        fits[i] = f
    end
    return fits
end

"""
    estimate_nepochs(h::Histogram, mu::Float64, Ltot::Number; max_nepochs::Int = 10, kwargs...)

Estimate the number of epochs needed to fit the histogram `h`.

The mutation rate `mu` is assumed to be per base pairs per generation, 
and the total length of the genome `Ltot` is in base pairs.

The optional argument `max_nepochs` defines the maximum number of epochs that are explored,
while the other keyword arguments are passed to [`pre_fit`](@ref).
"""
function estimate_nepochs(h::Histogram, mu::Float64, Ltot::Number; max_nepochs::Int = 10, kwargs...)
    fits = pre_fit(h, max_nepochs, mu, Ltot; kwargs...)
    nepochs = findlast(i->isassigned(fits, i), eachindex(fits))
    fits = fits[1:nepochs]
    fits = filter(m-> !isinf(evd(m)) && m.converged, fits)
    bestev = argmax(evd.(fits))
    nepochs = fits[bestev].nepochs
    return nepochs
end