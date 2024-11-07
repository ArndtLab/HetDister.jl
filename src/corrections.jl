"""
    corrected_fit(h_obs::Histogram, nepochs::Int, mu::Float64, rho::Float64; start = 1, iters = 100, final_factor=100)

Fit iteratively the histogram `h_obs` with up to `nepochs` and return a vector of `FitResult` 
and a vector of matrices `chain` with the parameter vector of each iteration.

# Arguments
- `h_obs::Histogram`: The histogram to fit.
- `nepochs::Int`: The maximum number of epochs to fit.
- `start::Int=1`: The optional starting epoch.
- `iters::Int=100`: The number of iterations to perform. Suggested value is between 1 and 4.
- `final_factor::Int=100`: The factor for the final simulation (i.e. how many genomes are simulated). If fitting a cumulative histogram it should be 1.
"""
function corrected_fit(h_obs::Histogram, nepochs::Int, mu::Float64, rho::Float64; start = 1, iters = 100, final_factor=100)
    h_sim = HistogramBinnings.Histogram(h_obs.edges)
    ho_mod = HistogramBinnings.Histogram(h_obs.edges)
    raw_fits = sequential_fit(h_obs, mu, nepochs)
    chains = []
    results = []

    for i in start:nepochs
        init = raw_fits[i].para
        chain = Matrix{Float64}(undef, iters+1, 2i)
        chain[1, :] .= init

        for iters in 1:iters
            weights_th = integral_ws(h_obs.edges[1].edges, mu, init)
            factor = any(init[3:3:end] ./ init[4:2:end] .> 1) ? 20 : 1
            get_sim!(init, h_sim, mu, rho, factor=factor)
        
            ho_mod.weights .= h_obs.weights
        
            diff = (h_sim.weights .- weights_th)
            temp = ho_mod.weights .- diff
            temp .= max.(temp, 0)
            ho_mod.weights .= Int.(ceil.(temp))

            f = fit_epochs(ho_mod, mu; init, nepochs = i, Tlow = 10, Tupp = 5init[2])
            f = perturb_fit!(f, ho_mod, mu, f.para, i, by_pass=true)
            if (f.opt.evidence == Inf) || f.opt.at_any_boundary || isnothing(f.opt.coeftable)
                f = sequential_fit(ho_mod, mu, i)[i]
                f = perturb_fit!(f, ho_mod, mu, f.para, i, by_pass=true)
            end
            init = f.para
            chain[1+iters, :] .= init
        end

        estimate = mean(chain[2:end,:], dims=1)[1,:]

        weights_th = integral_ws(h_obs.edges[1].edges, mu, estimate)
        get_sim!(estimate, h_sim, mu, rho, factor=final_factor);

        ho_mod.weights .= h_obs.weights

        diff = h_sim.weights .- weights_th
        temp = ho_mod.weights .- diff
        temp .= max.(temp, 0)
        ho_mod.weights .= Int.(ceil.(temp))
        final_fit = fit_epochs(ho_mod, mu; init=estimate, nepochs = i, Tupp = 5estimate[2])
        final_fit = perturb_fit!(final_fit, ho_mod, mu, final_fit.para, i, by_pass=true)
        if (final_fit.opt.evidence == Inf) || final_fit.opt.at_any_boundary || isnothing(final_fit.opt.coeftable)
            final_fit = sequential_fit(ho_mod, mu, i)[i]
            final_fit = perturb_fit!(final_fit, ho_mod, mu, final_fit.para, i, by_pass=true)
        end
        println(final_fit.opt.coeftable)
        println("log-evidence: ", final_fit.opt.evidence)
        push!(results, final_fit)
        push!(chains, chain)
    end

    return results, chains
end