function initializer(h::Histogram, mu::Float64, prev_para::Vector{T};
    frame::Number = 20, 
    pos::Bool = true, 
    smallest_segment::Int = 30
) where {T <: Number}

    # find approximate time of positive (negative) deviation from previous fit
    w_th = integral_ws(h.edges[1].edges, mu, prev_para)
    r = midpoints(h.edges[1])
    residuals = pos ? (h.weights - w_th) ./ sqrt.(h.weights) : (w_th - h.weights) ./ sqrt.(h.weights)
    residuals[h.weights .== 0] .= 0

    divide = zeros(Int, length(residuals))
    divide[(residuals .> 0) .& (r .> smallest_segment)] .= 1
    divide[residuals .< 0] .= 0
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

    for j in eachindex(residuals[1:end-1])
        if divide[j] == 0 && divide[j+1] == 1
            return 2 / (2mu * r[j])
        end
    end
    return nothing
end

# only Tupp is allowed
function bounds_strict(fit::FitResult)
    any(fit.opt.at_lboundary) || any(fit.opt.at_uboundary[2:2:end])
end

# Nupp and Tupp can be reached
function bounds_weak(fit::FitResult)
    any(fit.opt.at_lboundary)
end

function perturb_fit!(f::FitResult, h::Histogram, mu::Float64, init::Vector{Float64}, 
    nepochs::Int, Ltot::Number; 
    by_pass::Bool = false, 
    kwargs...
)
    if (evd(f) == Inf) || (bounds_strict(f) && by_pass)
        perturbations = mapreduce( i->fill(i, 10), vcat, [0.001, 0.01, 0.1, 0.5, 0.5, 2, 2] )
        for perturbation in perturbations
            f = fit_epochs(h, mu; init, Ltot, perturbation, nepochs, Tupp = 10init[2], kwargs...)
            if (evd(f) != Inf) && !(bounds_weak(f) && by_pass)
                break
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
same as in [`fit`](@ref).
"""
function pre_fit(h::Histogram, nfits::Int, mu::Float64, Ltot::Number;
    Tlow::Int = 10,
    Nlow::Int = 10, Nupp::Int = 100000,
    smallest_segment::Int = 30
)
    frame = 20
    fits = Vector{FitResult}(undef, nfits)
    for i in 1:nfits
        if i == 1
            f = fit_epochs(h, mu; nepochs = i, Ltot, Tlow, Nlow, Nupp)
        elseif i == 2
            f = fit_epochs(h, mu; nepochs = i, Ltot, Tlow, Nlow, Nupp)
            f = perturb_fit!(f, h, mu, f.para, i, Ltot; Tlow, Nlow, Nupp)
        else
            t1 = initializer(h, mu, fits[i-1].para; frame, pos = true, smallest_segment)
            if isnothing(t1)
                t1 = initializer(h, mu, fits[i-1].para; frame=frame/2, pos = true, smallest_segment)
            end
            t1 = isnothing(t1) ? 10 : t1
            t2 = initializer(h, mu, fits[i-1].para; frame, pos = false, smallest_segment)
            if isnothing(t2)
                t2 = initializer(h, mu, fits[i-1].para; frame=frame/2, pos = false, smallest_segment)
            end
            t2 = isnothing(t2) ? 10 : t2[1]
            t = max(t1, t2)
            @debug "initializer results " t1 t2

            if (t1 == 10) && (t2 == 10)
                @info "no more meaningful epochs can be added"
                return fits
            end

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
                f = fit_epochs(h, mu; init, nepochs = i, Tlow, Tupp = 10init[2], Ltot, Nlow, Nupp)
                f = perturb_fit!(f, h, mu, init, i, Ltot; Tlow, Nlow, Nupp)
            else
                if t == 10
                    newT1 = (ts[end-1] - ts[end]) / 2
                    newT1 = max(newT1, 200)
                    newT2 = newT1
                    @assert(split_epoch == length(ts), "split_epoch: $split_epoch, length(ts): $(length(ts))")
                else
                    newT1 = ts[split_epoch-1] - t
                    newT1 = max(newT1, 200)
                    newT2 = t - ts[split_epoch]
                    newT2 = max(newT2, 200)
                end
                newN = init[2split_epoch]
                init[2split_epoch-1] = newT1
                insert!(init, 2split_epoch, newT2)
                insert!(init, 2split_epoch, newN)
                f = fit_epochs(h, mu; init, nepochs = i, Tlow, Tupp = 10init[2], Ltot, Nlow, Nupp)
                f = perturb_fit!(f, h, mu, init, i, Ltot; Tlow, Nlow, Nupp)
            end
            if !f.converged
                @info "pre_fit: not converged, epoch $i"
                f = perturb_fit!(f, h, mu, init, i, Ltot; by_pass = true, Tlow, Nlow, Nupp)
            end
        end

        if any(isnan.(f.para))
            @info "pre_fit: nan para, epoch $i" f.para
            f = perturb_fit!(f, h, mu, f.opt.init, i, Ltot; by_pass = true, Tlow, Nlow, Nupp)
        end

        if i > 1 && evd(f) < evd(fits[i-1])
            @info "no more meaningful epochs can be added"
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
    return nepochs
end