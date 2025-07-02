struct Flat end

function (a::Flat)(L, it)
    return 1
end

struct Lin end

function (a::Lin)(L, it)
    maxL = 3e11
    f = min(1 + 10*(it-1), 1000)
    return min(f, maxL/L)
end

struct Sq end

function (a::Sq)(L, it)
    maxL = 3e11
    f = min(1 + (3*(it-1))^2, 1000)
    return min(f, maxL/L)
end

"""
    demoinfer(h_obs::Histogram, nepochs::Int, mu::Float64, rho::Float64, Ltot::Number; kwargs...)
    demoinfer(h_obs, nepochs, mu, rho, Ltot, init::Vector{Float64}; kwargs...)

Fit iteratively `h_obs` with a demographic history of piece-wise constant `nepochs`.

Return a vector of `FitResult`, see [`FitResult`](@ref), `mu` and `rho` are
the mutation and recombination rates, respectively, per base pair per generation
and `Ltot` is the total length of the genome, in base pairs.
Optional argument `init` can be used to provide an initial point for the iterations.

# Arguments
- `iters::Int = 5`: The number of iterations to perform. Suggested value is at least 5, annealing plays a role.
- `burnin::Int = 3`: The number of iterations to discard as burnin. Notice that in general with no annealing,
    i.e. `Flat()`, already the first iteration is a good sample.
- `allow_boundary::Bool=false`: If true, the function will allow the fit to reach the upper 
boundaries of populations sizes. This can be useful if structure is expected because epochs of
separation between two subpopulations show up as a higher population size (smaller coalescence rate).
- `level::Float64=0.95`: The confidence level for the confidence intervals on the parameters estimates.
- `Tlow::Int=10`: The lower bound for the duration of epochs.
- `Nlow::Int=10`, `Nupp::Int=100000`: The lower and upper bounds for the population sizes.
- `smallest_segment::Int=30`: The smallest segment size to consider for the optimization.
- `annealing = Sq()`: correction is computed by simulating a genome of length `factor` times the length of
the input genome. At each iteration the factor is changed according to the annealing function. It can be `Flat()`,
`Lin()` or `Sq()`. It can be a user defined function with signature `(L, it) -> factor` with `L` the genome length
and `it` the iteration index.
"""
function demoinfer(h_obs::Histogram, nepochs::Int, mu::Float64, rho::Float64, Ltot::Number, init::Vector{Float64}; 
    iters::Int = 6,
    burnin::Int = 3,
    allow_boundary::Bool = false,
    level::Float64 = 0.95,
    Tlow::Int = 10, Tupp = 1e7,
    Nlow::Int = 10, Nupp::Int = 100000,
    smallest_segment::Int = 30,
    annealing = Sq()
)

    burnin >= iters && @error "burnin must be smaller than iters"
    burnin += 1 
    length(init)÷2 == nepochs || @error "init must be in TN format, with 2*nepochs elements"

    h_sim = HistogramBinnings.Histogram(h_obs.edges)
    ho_mod = HistogramBinnings.Histogram(h_obs.edges)

    fop = FitOptions(Ltot; nepochs, init, Tlow, Tupp, Nlow, Nupp)

    chain = FitResult[]
    corrections = []

    for iter in 1:iters
        weights_th = integral_ws(h_obs.edges[1].edges, mu, init)
        factor = annealing(Ltot, iter)
        get_sim!(init, h_sim, mu, rho; factor)
    
        ho_mod.weights .= h_obs.weights
    
        diff = (h_sim.weights/factor .- weights_th)
        temp = ho_mod.weights .- diff
        temp .= round.(Int, temp)
        ho_mod.weights .= max.(temp, 0)
        
        f = fit_model_epochs(ho_mod, mu, fop)
        # f = pre_fit(ho_mod, nepochs, mu, Ltot; Tlow, Tupp, Nlow, Nupp, smallest_segment)[nepochs]
        f = perturb_fit!(f, ho_mod, mu, fop)

        init = f.para
        setinit!(fop, init)
        push!(chain, f)
        push!(corrections, diff)
    end

    estimate = zeros(length(chain[1].para))
    estimate_sd = zeros(length(chain[1].para))
    evidence = 0
    lp = 0
    correction = zeros(length(h_obs.weights))
    sample_size = 0
    for j in burnin:iters
        if (chain[j].converged &&
            !isinf(evd(chain[j])) &&
            any(chain[j].opt.pvalues[2:2:end] .< 0.05)
        )
            estimate .+= chain[j].para
            estimate_sd .+= sds(chain[j]) .^2
            evidence += evd(chain[j])
            lp += chain[j].lp
            correction .+= corrections[j]
            sample_size += 1
        end
    end
    estimate ./= sample_size
    estimate_sd .= sqrt.(estimate_sd./sample_size)
    evidence /= sample_size
    lp /= sample_size
    correction ./= sample_size
    corrected_weights = integral_ws(h_obs.edges[1].edges, mu, estimate) .+ correction
    
    zscore = fill(0.0, length(estimate))
    p = fill(1, length(estimate))
    ci_low = fill(-Inf, length(estimate))
    ci_high = fill(Inf, length(estimate))
    try 
        zscore = estimate ./ estimate_sd
        p = map(z -> StatsAPI.pvalue(Distributions.Normal(), z; tail=:right), zscore)
    
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
            chain, corrections, #sample_size,
            zscore,
            pvalues = p, ci_low, ci_high,
            h_obs, corrected_weights
        )
    )

    return final_fit
end

function demoinfer(h_obs::Histogram, nepochs::Int, mu::Float64, rho::Float64, Ltot::Number;
    iters::Int = 5,
    burnin::Int = 3,
    allow_boundary::Bool = false,
    level::Float64 = 0.95,
    Tlow::Int = 10, Tupp = 1e7,
    Nlow::Int = 10, Nupp::Int = 100000,
    smallest_segment::Int = 30,
    annealing = Sq()
)
    f = pre_fit(h_obs, nepochs, mu, Ltot; Tlow, Tupp, Nlow, Nupp, smallest_segment, force = true)
    if !isassigned(f, nepochs)
        @warn "fit failed, reduce the number of epochs"
        return nullFit(nepochs, mu, [], [])
    end
    f = f[nepochs]
    return demoinfer(h_obs, nepochs, mu, rho, Ltot, get_para(f); 
        iters,
        burnin,
        allow_boundary,
        level,
        Tlow, Tupp,
        Nlow, Nupp,
        smallest_segment,
        annealing
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
    ms = filter(m->any(m.opt.pvalues[2:2:end] .< 0.05), ms)
    ms = filter(m->all(m.opt.pvalues[3:2:end-1] .< 0.05), ms)
    if length(ms) > 0
        best = 1
        lp = ms[1].lp
        ev = evd(ms[1])
        i = 2
        while (i <= length(ms)) && (evd(ms[i]) > ev) && (ms[i].lp > lp)
            best = i
            lp = ms[i].lp
            ev = evd(ms[i])
            i += 1
        end
        return ms[best]
    end
    @warn "none of the models is meaningful"
    return nothing
end