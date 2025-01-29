"""
    fit(h_obs::Histogram, nepochs::Int, mu::Float64, rho::Float64, Ltot::Number; kwargs...)
    fit(h_obs, nepochs, mu, rho, Ltot, init::Vector{Float64}; kwargs...)

Fit iteratively `h_obs` with a demographic history of piece-wise constant `nepochs`.

Return a vector of `FitResult`, see [`FitResult`](@ref), `mu` and `rho` are
the mutation and recombination rates, respectively, per base pair per generation
and `Ltot` is the total length of the genome, in base pairs.
Optional argument `init` can be used to provide an initial point for the iterations.

# Arguments
- `iters::Int=100`: The number of iterations to perform. Suggested value is at least 10.
- `burnin::Int=5`: The number of iterations to discard as burnin. Notice that in general
    already the first iteration is a good sample.
- `allow_boundary::Bool=false`: If true, the function will allow the fit to reach the upper 
boundaries of populations sizes. This can be useful if structure is expected because epochs of
separation between two subpopulations show up as a higher population size (smaller coalescence rate).
- `level::Float64=0.95`: The confidence level for the confidence intervals on the parameters estimates.
- `Tlow::Int=10`: The lower bound for the duration of epochs.
- `Nlow::Int=10`, `Nupp::Int=100000`: The lower and upper bounds for the population sizes.
- `smallest_segment::Int=30`: The smallest segment size to consider for the optimization.
- `factor::Int=1`: correction is computed by simulating a genome of length `factor` times the length of
the input genome.
This dictates the lower bound which is considered in when initially guessing when to insert 
a new epoch, i.e. the upper bound for time splits. Notice that the optimization per se is
not affected by this limit.
"""
function fit(h_obs::Histogram, nepochs::Int, mu::Float64, rho::Float64, Ltot::Number, init::Vector{Float64}; 
    iters::Int = 100,
    burnin::Int = 0,
    allow_boundary::Bool = false,
    level::Float64 = 0.95,
    Tlow::Int = 10,
    Nlow::Int = 10, Nupp::Int = 100000,
    smallest_segment::Int = 30,
    factor::Int = 1
)

    burnin >= iters && @error "burnin must be smaller than iters"
    burnin += 1 
    length(init)÷2 == nepochs || @error "init must be in TN format, with 2*nepochs elements"

    h_sim = HistogramBinnings.Histogram(h_obs.edges)
    ho_mod = HistogramBinnings.Histogram(h_obs.edges)

    fop = FitOptions(Ltot; nepochs, init, Tlow, Nlow, Nupp)

    chain = FitResult[]
    corrections = []

    for iter in 1:iters
        weights_th = integral_ws(h_obs.edges[1].edges, mu, init)
        get_sim!(init, h_sim, mu, rho, factor=factor)
    
        ho_mod.weights .= h_obs.weights
    
        diff = (h_sim.weights/factor .- weights_th)
        temp = ho_mod.weights .- diff
        temp .= round.(Int, temp)
        ho_mod.weights .= max.(temp, 0)
        
        updateTupp!(fop, 10init[2])
        f = fit_model_epochs(ho_mod, mu, fop)
        f = perturb_fit!(f, ho_mod, mu, fop)
        if (evd(f) == Inf) || any(f.opt.at_lboundary) || any(f.opt.at_uboundary[2:end]) || isnothing(f.opt.coeftable)
            @info "fit failed, fallback on sequential fit"
            f_ = pre_fit(ho_mod, nepochs, mu, Ltot; smallest_segment)
            if !isassigned(f_, nepochs)
                @warn "fit failed, exiting at iter $iter,
                    consider reducing the number of epochs, currently set at $nepochs"
                # return nullFit(nepochs, mu, init)
            else
                f = f_[nepochs]
                f = perturb_fit!(f, ho_mod, mu, fop)
            end
        end

        init = f.para
        setinit!(fop, init)
        push!(chain, f)
        push!(corrections, diff)
    end

    estimate = zeros(length(chain[1].para))
    estimate_sd = zeros(length(chain[1].para))
    evidence = 0
    lp = 0
    sample_size = 0
    for j in burnin:iters
        if (!any(chain[j].opt.at_lboundary) &&
            !any(chain[j].opt.at_uboundary[3:2:end-1]) &&
            chain[j].converged &&
            !isinf(evd(chain[j])) &&
            all(chain[j].opt.pvalues .< 0.05)
        )
            if allow_boundary
                estimate .+= chain[j].para
                estimate_sd .+= sds(chain[j]) .^2
                evidence += evd(chain[j])
                lp += chain[j].lp
                sample_size += 1
            elseif !any(chain[j].opt.at_uboundary[2:2:end])
                estimate .+= chain[j].para
                estimate_sd .+= sds(chain[j]) .^2
                evidence += evd(chain[j])
                lp += chain[j].lp
                sample_size += 1
            end
        end
    end
    if sample_size == 0
        @debug "all fits discarded"
        return nullFit(nepochs, mu, init, [chain,corrections])
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
        ci_low = para .- q .* estimate_sd
        ci_high = para .+ q .* estimate_sd
    catch
        # most likely computing stderrors failed
        # we stay with the default values
    end

    final_fit = FitResult(
        nepochs,
        chain[end].bin,
        chain[end].mu,
        estimate,
        estimate_sd,
        chain[end].para_name,
        estimate,
        "iterative fit",
        true,
        lp,
        evidence,
        (; 
            init,
            chain, corrections, sample_size,
            zscore,
            pvalues = p, ci_low, ci_high,
        )
    )

    return final_fit
end

function fit(h_obs::Histogram, nepochs::Int, mu::Float64, rho::Float64, Ltot::Number;
    iters::Int = 100,
    burnin::Int = 0,
    allow_boundary::Bool = false,
    level::Float64 = 0.95,
    Tlow::Int = 10,
    Nlow::Int = 10, Nupp::Int = 100000,
    smallest_segment::Int = 30,
    factor::Int = 1
)
    f = pre_fit(h_obs, nepochs, mu, Ltot; Tlow, Nlow, Nupp, smallest_segment)
    if !isassigned(f, nepochs)
        @warn "fit failed, reduce the number of epochs"
        return nullFit(nepochs, mu, [], [])
    end
    f = f[nepochs]
    return fit(h_obs, nepochs, mu, rho, Ltot, get_para(f); 
        iters,
        burnin,
        allow_boundary,
        level,
        Tlow,
        Nlow, Nupp,
        smallest_segment,
        factor
    )
end

"""
    compare_models(models::Vector{FitResult})

Compare the models parameterized by `FitResult`s and return the best one.

### Theoretical explanation
TBD
"""
function compare_models(models::Vector{FitResult})
    ms = copy(models)
    ms = filter(m->!isinf(evd(m)), ms)
    ms = filter(m->all(m.opt.pvalues .< 0.05), ms)
    if length(ms) > 0
        evidences = evd.(ms)
        best = argmax(evidences)
        return ms[best]
    end
    @warn "none of the models is meaningful"
    return nothing
end