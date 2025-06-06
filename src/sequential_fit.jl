function tcondr(r::Number, para::Vector{T}, mu::Number) where T<:Number
    return 1 / (mu * (25r)^(1/1.3))
    # coalt = 1:40*para[2]
    # ts = MLDs.ordts(para)
    # ns = MLDs.ordns(para)
    # post = map(coalt) do t
    #     MLDs.approxposteriort(t, r, para[1], mu, ts, ns)
    # end
    # return coalt[argmax(post)]
end

function initializer(h::Histogram, mu::Float64, prev_para::Vector{T};
    frame::Number = 20, 
    pos::Bool = true,
    threshold::Float64 = 0.,
    smallest_segment::Int = 30
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

    for j in eachindex(divide[1:end-1])
        if divide[j] == 0 && divide[j+1] == 1
            t = tcondr(r[j], prev_para, mu)
            @debug "identified deviation " r[j]
            return t
        end
    end
    return 0.
end

function timesplitter(h::Histogram, mu::Float64, prev_para::Vector{T}, tprevious::Vector;
    frame::Number = 20,
    threshold::Float64 = 0.,
    smallest_segment::Int = 30
) where {T <: Number}
    t1 = initializer(h, mu, prev_para; 
        frame, pos = true, threshold, smallest_segment
    )
    if iszero(t1)
    t1 = initializer(h, mu, prev_para;
        frame=frame/2, pos = true, threshold, smallest_segment
    )
    end
    t2 = initializer(h, mu, prev_para;
        frame, pos = false, threshold, smallest_segment
    )
    if iszero(t2)
    t2 = initializer(h, mu, prev_para;
        frame=frame/2, pos = false, threshold, smallest_segment
    )
    end
    t = min(t1, t2)
    if any(t .== tprevious)
    t = max(t1, t2)
    end
    if iszero(t1)
    t = t2
    elseif iszero(t2)
    t = t1
    end
    @debug "initializer results " t1 t2 t
    return t
end

# TODO: recycle spline for splitting

# function timesplitter(h::Histogram, mu::Float64, prev_para::Vector{T}; 
#     n_nodes::Int = 35
# ) where {T <: Number}
#     N0 = prev_para[2]
#     xs = log.(midpoints(h.edges[1]))
#     w0 = integral_ws(h.edges[1].edges, mu, prev_para)
#     ys = (h.weights - w0) ./ sqrt.(w0)
#     nodes = LinRange(xs[1],xs[end],n_nodes)
#     fitsp = fit_nspline(xs,ys,nodes)
#     drs = ForwardDiff.derivative.(x->ForwardDiff.derivative(fitsp,x), nodes)

#     # spline 2nd derivative is piecewise linear, find zeros
#     inflections = Float64[]
#     for i in 1:length(drs)-1
#         if drs[i] * drs[i+1] < 0
#             m = (drs[i+1] - drs[i]) / (nodes[i+1] - nodes[i])
#             b = drs[i] - m * nodes[i]
#             x = -b/m
#             push!(inflections, x)
#         end
#     end
#     shoulder = argmin(abs.(inflections .- log(1/4mu/N0)))

#     # find maximum difference between spline and linear interpolation between inflections
#     deviations = Float64[]
#     for i in 1:length(inflections)
#         xp = i == 1 ? xs[1] : inflections[i-1]
#         xn = inflections[i]
#         m = (fitsp(xn) - fitsp(xp)) / (xn - xp)
#         b = fitsp(xp) - m * xp
#         devs = fitsp.(xp:0.01:xn) - (m * (xp:0.01:xn) .+ b)
#         argmaxdev = argmax(abs.(devs))
#         push!(deviations, devs[argmaxdev])
#     end
#     xp = inflections[end]
#     xn = xs[end]
#     m = (fitsp(xn) - fitsp(xp)) / (xn - xp)
#     b = fitsp(xp) - m * xp
#     devs = fitsp.(xp:0.01:xn) - (m * (xp:0.01:xn) .+ b)
#     argmaxdev = argmax(abs.(devs))
#     push!(deviations, devs[argmaxdev])

#     # rough prioritization of inflections, for splitting, may be unnecessary
#     # scores = similar(inflections)
#     # for i in 1:length(inflections)
#     #     scores[i] = abs(deviations[i])+abs(deviations[i+1])
#     #     if i == shoulder
#     #         scores[i] = abs(deviations[i]+deviations[i+1])
#     #     end
#     # end
#     # I = sortperm(scores, rev=true)
#     # inflections = inflections[I]
#     inflections .= exp.(inflections)
#     ts = (exp.((inflections*4N0*mu).^0.35) .+ 1e5) ./ exp.((inflections*4N0*mu).^0.35)
#     # ts = 1 ./ (mu * (25inflections).^(1/1.3))
#     return ts
# end

function epochfinder!(init::Vector{T}, N0, t, fop::FitOptions) where {T <: Number}
    # t = min(t, 12N0) # permissive upper bound to 12 times the ancestral population size
    ts = reverse(pushfirst!(cumsum(init[end-1:-2:3]),0))
    split_epoch = findfirst(ts .< t)
    # @debug "split epoch " split_epoch

    if split_epoch == 1
        newT = t - ts[1]
        newT = max(newT, 1000)
        newN = init[2]
        insert!(init, 3, newN)
        insert!(init, 3, newT)
    else
        newT1 = ts[split_epoch-1] - t
        newT1 = max(newT1, 20)
        newT2 = t - ts[split_epoch]
        newT2 = max(newT2, 20)
        newN = init[2split_epoch]
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
    by_pass::Bool = true,
    isrnd::Bool = true
)
    if (evd(f) == Inf) || any(f.opt.at_lboundary) || any(f.opt.at_uboundary[2:end])
        if isrnd
            factors = mapreduce( i->fill(i, 10), vcat, [0.001, 0.01, 0.1, 0.5, 0.5, 0.9, 2] )
            for fct in factors
                perturbations = Perturbation[]
                for i in eachindex(f.para)
                    if f.opt.at_lboundary[i] || (f.opt.at_uboundary[i] && i > 1)
                        push!(perturbations, Perturbation(isrnd, fct, i))
                    end
                end
                setinit!(fop, f.para)
                fop.perturbations = perturbations
                f = fit_model_epochs(h, mu, fop)
                if (evd(f) != Inf) && !(any(f.opt.at_lboundary[1:end-2]) && by_pass)
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
same as in [`demoinfer`](@ref).
"""
function pre_fit(h::Histogram, nfits::Int, mu::Float64, Ltot::Number;
    Tlow::Int=10, Nlow::Int=10, Nupp::Int=100000,
    smallest_segment::Int = 30,
    require_convergence::Bool = true
)
    fits = Vector{FitResult}(undef, nfits)
    tprevious = [0.]
    for i in 1:nfits
        fop = FitOptions(Ltot; nepochs = i, Tlow, Nlow, Nupp)
        if i == 1
            f = fit_model_epochs(h, mu, fop)
            push!(tprevious, 20 * f.para[2])
        else
            t = timesplitter(h, mu, fits[i-1].para, tprevious; smallest_segment=smallest_segment)
            if iszero(t)
                @info "pre_fit: no split found, epoch $i"
                return fits
            end
            push!(tprevious, t)
            init = copy(fits[i-1].para)
            epochfinder!(init, fits[1].para[2], t, fop)
            setinit!(fop, init)
            updateTupp!(fop, 10init[2])
            f = fit_model_epochs(h, mu, fop)
            f = perturb_fit!(f, h, mu, fop)
            if require_convergence && !f.converged
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
    fits = pre_fit(h, max_nepochs, mu, Ltot; require_convergence = false, kwargs...)
    nepochs = findlast(i->isassigned(fits, i), eachindex(fits))
    fits = fits[1:nepochs]
    fits = filter(m-> !isinf(evd(m)) && m.converged, fits)
    bestev = argmax(evd.(fits))
    nepochs = fits[bestev].nepochs
    return nepochs
end