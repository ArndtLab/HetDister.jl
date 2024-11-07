function initializer(h, μ, prev_para; frame = 20, pos = true)    # find approximate time of positive (negative) deviation from previous fit
    w_th = integral_ws(h.edges[1].edges, μ, prev_para)
    r = midpoints(h.edges[1])
    residuals = pos ? (h.weights - w_th) ./ sqrt.(h.weights) : (w_th - h.weights) ./ sqrt.(h.weights)
    residuals[h.weights .== 0] .= 0

    divide = zeros(Int, length(residuals))
    divide[(residuals .> 0) .& (r .> 30)] .= 1
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
            return 3 / (2μ * r[j])
        end
    end
    return nothing
end

function perturb_fit!(f, h, μ, init, nepochs; by_pass = false)
    tbounds(fit) = any(fit.para .== 10.)
    if (f.opt.evidence == Inf) || (tbounds(f) && by_pass)
        perturbations = mapreduce(i->fill(i, 10),vcat, [0.001, 0.01, 0.1, 0.5, 0.5, 2, 2])
        for perturbation in perturbations
            f = fit_epochs(h, μ; init, perturbation = perturbation, nepochs, Tupp = 5init[2], Tlow = 10)
            if (f.opt.evidence != Inf) && !(tbounds(f) && by_pass)
                break
            end
        end
    end
    return f
end

"""
    sequential_fit(h::Histogram, μ::Float64, nfits::Int; Tlow = 10)

Fit (uncorrected) the histogram `h` with an increasing number of epochs, starting from 1 up to `nfits`.

The function returns an array of `FitResult` objects, one for each fit.
`Tlow` is the optional lower bound for the duration of epochs.
"""
function sequential_fit(h::Histogram, μ::Float64, nfits::Int; Tlow = 10)
    fits = Vector{FitResult}(undef, nfits)
    for i in 1:nfits
        if i == 1
            f = fit_epochs(h, μ; nepochs = i)
        elseif i == 2
            f = fit_epochs(h, μ; nepochs = i, Tlow = Tlow)
            f = perturb_fit!(f, h, μ, f.para, i)
        else
            t1 = initializer(h, μ, fits[i-1].para, pos = true)
            if isnothing(t1)
                t1 = initializer(h, μ, fits[i-1].para, frame=10, pos = true)
            end
            t1 = isnothing(t1) ? 10 : t1
            t2 = initializer(h, μ, fits[i-1].para, pos = false)
            if isnothing(t2)
                t2 = initializer(h, μ, fits[i-1].para, frame=10, pos = false)
            end
            t2 = isnothing(t2) ? 10 : t2[1]
            t = max(t1, t2)

            init = copy(fits[i-1].para)
            meanN = fits[1].para[2]
            t = min(t, 12meanN) # permissive upper bound to 12 times the ancestral population size. It can anyway vary during the optimization
            ts = reverse(pushfirst!(cumsum(init[end:-2:3]),0))
            split_epoch = findfirst(ts .< t)

            if split_epoch == 1
                newT = t - ts[1]
                newT = max(newT, 1000)
                newN = init[2]
                insert!(init, 3, newN)
                insert!(init, 3, newT)
                f = fit_epochs(h, μ; init, nepochs = i, Tlow = Tlow, Tupp = 5init[2])
                f = perturb_fit!(f, h, μ, init, i)
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
                f = fit_epochs(h, μ; init, nepochs = i, Tlow = Tlow, Tupp = 5init[2])
                f = perturb_fit!(f, h, μ, init, i)
            end
            if !f.converged
                f = perturb_fit!(f, h, μ, init, i, by_pass = true)
            end
        end

        if any(isnan.(f.para))
            f = perturb_fit!(f, h, μ, f.opt.init, i, by_pass = true)
        end

        fits[i] = f
    end
    return fits
end