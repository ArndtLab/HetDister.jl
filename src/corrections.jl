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
    demoinfer(h_obs::Histogram, nepochs::Int, mu::Float64, rho::Float64, Ltot::Number, init::Vector{Float64}; kwargs...)

Fit iteratively `h_obs` with a demographic history of piece-wise constant `nepochs`
starting from an initial parameter vector.

Return a vector of `FitResult`, see [`FitResult`](@ref), `mu` and `rho` are
the mutation and recombination rates, respectively, per base pair per generation
and `Ltot` is the total length of the genome, in base pairs.
Optional argument `init` can be used to provide an initial point for the iterations.

# Arguments
- `iters::Int=9`: The number of iterations to perform. Due to stochasticity, the rate of success for the fit
will increase with the number of iterations.
- `level::Float64=0.95`: The confidence level for the confidence intervals on the parameters estimates.
- `Tlow::Number=10`, `Tupp::Number=1e7`: The lower and upper bounds for the duration of epochs.
- `Nlow::Number=10`, `Nupp::Number=1e5`: The lower and upper bounds for the population sizes.
- `annealing=Sq()`: correction is computed by simulating a genome of length `factor` times the length of
the input genome. At each iteration the factor is changed according to the annealing function. It can be `Flat()`,
`Lin()` or `Sq()`. It can be a user defined function with signature `(L, it) -> factor` with `L` the genome length
and `it` the iteration index.
- `s::Int=1234`: The random seed for the random number generator, used to compute the correction.
- `restart::Int=3`: The number of iterations after which the fit is restarted with a different seed.
- `top::Int=3`: the number of fits which is averaged for the final estimate, having best ranking likelihoods.
"""
function demoinfer(h_obs::Histogram, nepochs::Int, mu::Float64, rho::Float64, Ltot::Number, init::Vector{Float64}; 
    iters::Int = 9,
    level::Float64 = 0.95,
    Tlow::Number = 10, Tupp::Number = 1e7,
    Nlow::Number = 10, Nupp::Number = 1e5,
    annealing = Sq(),
    s::Int = 1234,
    restart::Int = 3,
    top::Int = 3
)
    Random.seed!(s)

    length(init)÷2 == nepochs || @error "init must be in TN format, with 2*nepochs elements"

    h_sim = HistogramBinnings.Histogram(h_obs.edges)
    ho_mod = HistogramBinnings.Histogram(h_obs.edges)

    fop = FitOptions(Ltot; nepochs, init, Tlow, Tupp, Nlow, Nupp)

    chain = FitResult[]
    corrections = []

    init_ = copy(init)
    for iter in 1:iters
        if iter % restart == 1
            Random.seed!(s + iter)
            init_ = copy(init)
        end
        weights_th = integral_ws(h_obs.edges[1].edges, mu, init_)
        factor = annealing(Ltot, iter)
        get_sim!(init_, h_sim, mu, rho; factor)
    
        ho_mod.weights .= h_obs.weights
    
        diff = (h_sim.weights/factor .- weights_th)
        temp = ho_mod.weights .- diff
        temp .= round.(Int, temp)
        ho_mod.weights .= max.(temp, 0)
        
        f = fit_model_epochs(ho_mod, mu, fop)
        f = perturb_fit!(f, ho_mod, mu, fop)

        init_ = f.para
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
    chain_ = filter(c -> c.converged && !isinf(evd(c)) && any(c.opt.pvalues[2:2:end] .< 0.05), chain)
    if isempty(chain_)
        @warn "no converged fits found, returning null fit"
        return nullFit(nepochs, mu, [], [])
    elseif length(chain_) < top
        @warn "not enough converged fits found, using only $(length(chain_)) fits"
    end
    sort!(chain_, by = c -> c.lp, rev = true)
    for j in 1:min(length(chain_),top)
        estimate .+= chain_[j].para
        estimate_sd .+= sds(chain_[j]) .^2
        evidence += evd(chain_[j])
        lp += chain_[j].lp
        correction .+= corrections[j]
        sample_size += 1
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
            chain, corrections, sample_size,
            zscore,
            pvalues = p, ci_low, ci_high,
            h_obs, corrected_weights
        )
    )

    return final_fit
end

"""
    demoinfer(h_obs::Histogram, nepochs::Int, mu::Float64, rho::Float64, Ltot::Number; kwargs...)

Fit iteratively `h_obs` with a demographic history of piece-wise constant `nepochs`.

Return a vector of `FitResult`, see [`FitResult`](@ref), `mu` and `rho` are
the mutation and recombination rates, respectively, per base pair per generation
and `Ltot` is the total length of the genome, in base pairs.
Optional argument `init` can be used to provide an initial point for the iterations.

# Arguments
- `iters::Int=9`: The number of iterations to perform. Due to stochasticity, the rate of success for the fit
will increase with the number of iterations.
- `level::Float64=0.95`: The confidence level for the confidence intervals on the parameters estimates.
- `Tlow::Number=10`, `Tupp::Number=1e7`: The lower and upper bounds for the duration of epochs.
- `Nlow::Number=10`, `Nupp::Number=1e5`: The lower and upper bounds for the population sizes.
- `smallest_segment::Int=30`: The smallest segment size present in the histogram to consider for the optimization.
- `annealing=Sq()`: correction is computed by simulating a genome of length `factor` times the length of
the input genome. At each iteration the factor is changed according to the annealing function. It can be `Flat()`,
`Lin()` or `Sq()`. It can be a user defined function with signature `(L, it) -> factor` with `L` the genome length
and `it` the iteration index.
- `force::Bool=true`: if `true`, the fit will try to add epochs even when no signal is found.
- `s::Int=1234`: The random seed for the random number generator, used to compute the correction.
- `restart::Int=3`: The number of iterations after which the fit is restarted with a different seed.
- `top::Int=3`: the number of fits which is averaged for the final estimate, having best ranking likelihoods.
"""
function demoinfer(h_obs::Histogram, nepochs::Int, mu::Float64, rho::Float64, Ltot::Number;
    iters::Int = 9,
    level::Float64 = 0.95,
    Tlow::Number = 10, Tupp::Number = 1e7,
    Nlow::Number = 10, Nupp::Number = 1e5,
    smallest_segment::Int = 30,
    annealing = Sq(),
    force::Bool = true,
    s::Int = 1234,
    restart::Int = 3,
    top::Int = 3
)
    f = pre_fit(h_obs, nepochs, mu, Ltot; Tlow, Tupp, Nlow, Nupp, smallest_segment, force)
    nepochs_ = findlast(i->isassigned(f, i), eachindex(f))
    if nepochs_ < nepochs
        @warn "models above $nepochs did not converge, stopping at $nepochs_"
    end

    results = Vector{FitResult}(undef, nepochs)
    @threads for n in 1:nepochs_
        results[n] = demoinfer(h_obs, n, mu, rho, Ltot, get_para(f[n]); 
            iters, level, Tlow, Tupp, Nlow, Nupp,
            annealing, s, restart, top
        )
    end

    return results
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