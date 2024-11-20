"""
    corrected_fit(h_obs::Histogram, nepochs::Int, mu::Float64, rho::Float64; kwargs...)

Fit iteratively the histogram `h_obs` with up to `nepochs` and return a vector of `FitResult` 
and a vector of matrices `chain` with the parameter vector of each iteration.

# Arguments
- `h_obs::Histogram`: The histogram to fit.
- `nepochs::Int`: The maximum number of epochs to fit.
- `mu::Float64`: The mutation rate.
- `rho::Float64`: The recombination rate.
- `Ltot::Number`: The total length of the genome.
- `start::Int=1`: The optional starting epoch.
- `iters::Int=100`: The number of iterations to perform. Suggested value is at least 10.
- `final_factor::Int=100`: The factor for the final simulation (i.e. how many genomes are simulated). 
If fitting a cumulative histogram it should be 1.
- `allow_boundary::Bool=false`: If true, the function will allow the fit to reach the upper 
boundaries of populations sizes. This can be useful if structure is expected because epochs of
separation between two subpopulations show up as a higher population size (smaller coalescence rate).
- `level::Float64=0.95`: The confidence level for the confidence intervals on the parameters estimates.
"""
function corrected_fit(h_obs::Histogram, nepochs::Int, mu::Float64, rho::Float64, Ltot::Number; 
    start = 1,
    iters = 100,
    final_factor=100, 
    allow_boundary=false,
    level = 0.95
    )
    h_sim = HistogramBinnings.Histogram(h_obs.edges)
    ho_mod = HistogramBinnings.Histogram(h_obs.edges)
    raw_fits = sequential_fit(h_obs, mu, nepochs, Ltot)
    chains = []
    results = FitResult[]

    for i in start:nepochs
        init = raw_fits[i].para
        chain = FitResult[]
        push!(chain, raw_fits[i])

        for iter in 1:iters
            weights_th = integral_ws(h_obs.edges[1].edges, mu, init)
            # factor = any( (init[3:2:end] ./ init[4:2:end]) .> 1 ) ? 20 : 1
            factor = 1
            get_sim!(init, h_sim, mu, rho, factor=factor)
        
            ho_mod.weights .= h_obs.weights
        
            diff = (h_sim.weights/factor .- weights_th)
            temp = ho_mod.weights .- diff
            temp .= round.(Int, temp)
            ho_mod.weights .= max.(temp, 0)

            f = fit_epochs(ho_mod, mu; init, Ltot, nepochs = i, Tlow = 10, Tupp = 5init[2])
            f = perturb_fit!(f, ho_mod, mu, f.para, i, Ltot, by_pass=true)
            if (f.opt.evidence == Inf) || any(f.opt.at_boundary[2:end]) || isnothing(f.opt.coeftable)
                f = sequential_fit(ho_mod, mu, i, Ltot)[i]
                f = perturb_fit!(f, ho_mod, mu, f.para, i, Ltot, by_pass=true)
            end
            
            init = f.para
            push!(chain, f)
        end

        estimate = zeros(length(chain[1].para))
        estimate_sd = zeros(length(chain[1].para))
        evidence = 0
        lp = 0
        burnin = iters > 1 ? 3 : 2
        sample_size = 0
        for j in burnin:iters
            if all(chain[j].para .!= 10) && chain[j].converged
                if allow_boundary
                    estimate .+= chain[j].para
                    estimate_sd .+= chain[j].opt.stderrors .^2
                    evidence += chain[j].opt.evidence
                    lp += chain[j].lp
                    sample_size += 1
                elseif all(chain[j].para != 1e5)
                    estimate .+= chain[j].para
                    estimate_sd .+= chain[j].opt.stderrors .^2
                    evidence += chain[j].opt.evidence
                    lp += chain[j].lp
                    sample_size += 1
                end
            end
        end
        estimate ./= sample_size
        estimate_sd .= sqrt.(estimate_sd./sample_size)
        evidence /= sample_size
        lp /= sample_size
        
        zscore = fill(0.0, length(estimate))
        p = fill(1, length(estimate))
        ci_low = fill(-Inf, length(estimate))
        ci_high = fill(Inf, length(estimate))
        try 
            zscore = estimate ./ estimate_sd
            p = map(z -> StatsAPI.pvalue(Distributions.Normal(), z; tail=:both), zscore)
        
            # Confidence interval (CI)
            q = Statistics.quantile(Distributions.Normal(), (1 + level) / 2)
            ci_low = para .- q .* stderrors
            ci_high = para .+ q .* stderrors
        catch
            # most likely computing stderrors failed
            # we stay with the default values
        end

        final_fit = FitResult(
            i,
            chain[end].bin,
            chain[end].mu,
            estimate,
            chain[end].para_name,
            estimate,
            "iterative fit",
            true,
            lp,
            (;
                stderrors = estimate_sd, 
                zscore,
                pvalues = p, ci_low, ci_high,
                evidence
            )
        )

        # weights_th = integral_ws(h_obs.edges[1].edges, mu, estimate)
        # get_sim!(estimate, h_sim, mu, rho, factor=final_factor);

        # ho_mod.weights .= h_obs.weights

        # diff = h_sim.weights/final_factor .- weights_th
        # temp = ho_mod.weights .- diff
        # temp .= round.(Int, temp)
        # ho_mod.weights .= max.(temp, 0)
        # final_fit = fit_epochs(ho_mod, mu; init=estimate, nepochs = i, Tupp = 5estimate[2], Ltot)
        # final_fit = perturb_fit!(final_fit, ho_mod, mu, final_fit.para, i, Ltot, by_pass=true)
        # if (final_fit.opt.evidence == Inf) ||  any(final_fit.opt.at_boundary[2:end]) || isnothing(final_fit.opt.coeftable)
        #     final_fit = sequential_fit(ho_mod, mu, i, Ltot)[i]
        #     final_fit = perturb_fit!(final_fit, ho_mod, mu, final_fit.para, i, Ltot, by_pass=true)
        # end
        # println(final_fit.opt.coeftable)

        println("log-evidence: ", final_fit.opt.evidence)
        push!(results, final_fit)
        push!(chains, chain)
    end

    return results, chains
end