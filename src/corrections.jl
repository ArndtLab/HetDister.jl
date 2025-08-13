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

Fit iteratively histogram `h_obs` with a demographic history of piece-wise constant `nepochs`
starting from an initial parameter vector `init`.

Return a vector of `FitResult`, see [`FitResult`](@ref), `mu` and `rho` are
respectively the mutation and recombination rates, per base pair per generation,
and `Ltot` is the total length of the sequence, in base pairs.

# Optional Arguments
- `iters::Int=8`: The number of iterations to perform. Currently automatic check for convergence
is not implemented.
- `level::Float64=0.95`: The confidence level for the confidence intervals on the parameters estimates.
- `Tlow::Number=10`, `Tupp::Number=1e7`: The lower and upper bounds for the duration of epochs.
- `Nlow::Number=10`, `Nupp::Number=1e8`: The lower and upper bounds for the population sizes.
- `annealing=Sq()`: correction is computed by simulating a genome of length `factor` times the length of
the input genome. At each iteration the factor is changed according to the annealing function. It can
be `Flat()`, `Lin()` or `Sq()`. It can be a user defined function with signature `(L, it) -> factor`
with `L` the genome length and `it` the iteration index.
- `s::Int=1234`: The random seed for the random number generator, used to compute the correction.
- `restart::Int=100`: The number of iterations after which the fit is restarted with a different seed. Set
to a default high number, it should not be needed.
- `top::Int=1`: the number of fits at chain tail which is averaged for the final estimate.
"""
function demoinfer(h_obs::Histogram, nepochs::Int, mu::Float64, rho::Float64, Ltot::Number, init::Vector{Float64}; 
    iters::Int = 8,
    level::Float64 = 0.95,
    Tlow::Number = 10, Tupp::Number = 1e7,
    Nlow::Number = 10, Nupp::Number = 1e8,
    annealing = Sq(),
    s::Int = 1234,
    restart::Int = 100,
    top::Int = 1
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
        factor = annealing(Ltot, (iter-1) % restart + 1)
        get_sim!(init_, h_sim, mu, rho; factor)
    
        ho_mod.weights .= h_obs.weights
    
        diff = (h_sim.weights/factor .- weights_th)
        temp = ho_mod.weights .- diff
        temp .= round.(Int, temp)
        ho_mod.weights .= max.(temp, 0)
        
        setinit!(fop, init_)
        f = fit_model_epochs(ho_mod, mu, fop)
        f = perturb_fit!(f, ho_mod, mu, fop)

        init_ = f.para
        push!(chain, f)
        push!(corrections, diff)
        @show iter
    end

    estimate = zeros(length(chain[1].para))
    estimate_sd = zeros(length(chain[1].para))
    evidence = 0
    lp = 0
    correction = zeros(length(h_obs.weights))
    sample_size = 0
    mask = map(c -> c.converged && !isinf(evd(c)), chain)
    chain_ = chain[mask]
    conv = true
    if isempty(chain_)
        @warn "fits did not converge or have infinite evidence"
        conv = false
        chain_= chain
        mask = fill(true, length(chain))
    elseif length(chain_) < top
        @warn "not enough converged fits found, using only $(length(chain_)) fits"
    end
    for j in 1:min(length(chain_),top)
        estimate .+= chain_[end-j+1].para
        estimate_sd .+= sds(chain_[end-j+1]) .^2
        evidence += evd(chain_[end-j+1])
        lp += chain_[end-j+1].lp
        correction .+= corrections[mask][end-j+1]
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
        conv,
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
    demoinfer(segments::Vector{Int}, nepochs::Int, mu::Float64, rho::Float64; kwargs...)

Make an histogram with IBS `segments` and fit it iteratively with a demographic
history of piece-wise constant `nepochs`.

Return a vector of `FitResult`, see [`FitResult`](@ref), `mu` and `rho` are
respectively the mutation and recombination rates per base pair per generation.

# Optional Arguments
- `lo::Int=1`: The lowest segment length to be considered in the histogram
- `hi::Int=50_000_000`: The highest segment length to be considered in the histogram
- `nbins::Int=200`: The number of bins to use in the histogram
- `iters::Int=8`: The number of iterations to perform. Currently automatic check for convergence
is not implemented.
- `level::Float64=0.95`: The confidence level for the confidence intervals on the parameters estimates.
- `Tlow::Number=10`, `Tupp::Number=1e7`: The lower and upper bounds for the duration of epochs.
- `Nlow::Number=10`, `Nupp::Number=1e8`: The lower and upper bounds for the population sizes.
- `smallest_segment::Int=1`: The smallest segment size present in the histogram to consider 
for the optimization.
- `annealing=nothing`: correction is computed by simulating a genome of length `factor` times 
the length of the input genome. At each iteration the factor is changed according to the 
annealing function. It can be `Flat()`, `Lin()` or `Sq()`. It can be a user defined 
function with signature `(L, it) -> factor` with `L` the genome length and `it` the
iteration index. By default it is computed adaptively based on the input data, such 
that the total expected volume of the histogram is 2e8.
- `force::Bool=false`: if `true`, the fit will try to add epochs even when no signal is found.
- `s::Int=1234`: The random seed for the random number generator, used to compute the correction.
- `restart::Int=100`: The number of iterations after which the fit is restarted with
a different seed. Set to a default high number, it should not be needed.
- `top::Int=1`: the number of fits at chain tail which is averaged for the final estimate.
"""
function demoinfer(segments::Vector{Int}, nepochs::Int, mu::Float64, rho::Float64;
    lo::Int = 1, hi::Int = 50_000_000, nbins::Int = 200,
    iters::Int = 8,
    level::Float64 = 0.95,
    Tlow::Number = 10, Tupp::Number = 1e7,
    Nlow::Number = 10, Nupp::Number = 1e8,
    smallest_segment::Int = 1,
    annealing = nothing,
    force::Bool = false,
    s::Int = 1234,
    restart::Int = 100,
    top::Int = 1
)
    h_obs = adapt_histogram(segments; lo, hi, nbins)
    Ltot = sum(segments)

    f = pre_fit(h_obs, nepochs, mu, Ltot; 
        Tlow, Tupp, Nlow, Nupp, smallest_segment,
        force, require_convergence = false
    )
    nepochs_ = findlast(i->isassigned(f, i), eachindex(f))
    if nepochs_ < nepochs
        @warn "models above $nepochs did not converge, stopping at $nepochs_"
    end

    if isnothing(annealing)
        target = 2e8
        thetaL = length(segments)
        factor = target / thetaL
        annealing = (L, it) -> factor
    end

    results = Vector{FitResult}(undef, nepochs_)
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
    ms = filter(m->isfinite(evd(m)) && m.converged, ms)
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