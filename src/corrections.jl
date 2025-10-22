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
    demoinfer(segments::AbstractVector{<:Integer}, epochrange::Int, mu::Float64, rho::Float64; kwargs...)

Make an histogram with IBS `segments` and infer demographic histories with
piece-wise constant epochs where the number of epochs is smaller or equal 
to `epochrange`.

Return a vector of `FitResult` of length smaller or equal to `epochrange`, 
see [`FitResult`](@ref), `mu` and `rho` are respectively the mutation 
and recombination rates per base pair per generation.

# Optional Arguments
- `fop::FitOptions = FitOptions(sum(segments))`: the fit options, see [`FitOptions`](@ref).
- `lo::Int=1`: The lowest segment length to be considered in the histogram
- `hi::Int=50_000_000`: The highest segment length to be considered in the histogram
- `nbins::Int=200`: The number of bins to use in the histogram
- `iters::Int=8`: The number of iterations to perform. Currently automatic check for convergence
is not implemented.
- `annealing=nothing`: correction is computed by simulating a genome of length `factor` times 
the length of the input genome. At each iteration the factor is changed according to the 
annealing function. It can be `Flat()`, `Lin()` or `Sq()`. It can be a user defined 
function with signature `(L, it) -> factor` with `L` the genome length and `it` the
iteration index. By default it is computed adaptively based on the input data, such 
that the total expected volume of the histogram is 2e8.
- `s::Int=1234`: The random seed for the random number generator, used to compute the correction.
- `top::Int=1`: the number of fits at chain tail which is averaged for the final estimate.
- `level::Float64=0.95`: the confidence level for the confidence intervals.
"""
function demoinfer(segments::AbstractVector{<:Integer}, epochrange::Int, mu::Float64, rho::Float64;
    fop::FitOptions = FitOptions(sum(segments)),
    lo::Int = 1, hi::Int = 50_000_000, nbins::Int = 200,
    kwargs...
)
    h = adapt_histogram(segments; lo, hi, nbins)
    if sum(segments) != fop.Ltot
        @warn "inconsistent Ltot and segments, taking sum(segments)"
        fop.Ltot = sum(segments)
    end
    return demoinfer(h, epochrange, mu, rho, fop.Ltot; 
        fop = fop, kwargs...
    )
end

function demoinfer(h_obs::Histogram{T,1,E}, epochs::Int, mu::Float64, rho::Float64, Ltot::Number;
    fop::FitOptions = FitOptions(Ltot),
    annealing = nothing,
    iters::Int = 8,
    s::Int = 1234,
    top::Int = 1,
    level::Float64 = 0.95
) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    @assert !isempty(h_obs.weights) "histogram is empty"
    @assert epochs > 0 "epochrange has to be strictly positive"
    if Ltot != fop.Ltot
        @warn "inconsistent Ltot and fit options, taking Ltot"
        fop.Ltot = Ltot
    end
    if isnothing(annealing)
        target = 2e8
        thetaL = sum(h_obs.weights)
        factor = target / thetaL
        factor = max(1, factor)
        annealing = (L, it) -> factor
    end

    Random.seed!(s)

    h_sim = Histogram(h_obs.edges)
    ho_mod = Histogram(h_obs.edges)

    chain = []
    corrections = []
    corvars = []

    fits = pre_fit!(fop, h_obs, epochs, mu; require_convergence = false)
    f = compare_models(fits)
    init = get_para(f)
    for iter in 1:iters
        weights_th = integral_ws(h_obs.edges[1], mu, init)
        factor = annealing(Ltot, iter)
        get_sim!(init, h_sim, mu, rho; factor)

        ho_mod.weights .= h_obs.weights
    
        diff = (h_sim.weights/factor .- weights_th)
        vard = h_sim.weights/factor^2
        diff[1:4] .= 0.
        temp = ho_mod.weights .- diff
        temp .= round.(Int, temp)
        ho_mod.weights .= max.(temp, 0)
        
        fits = pre_fit!(fop, ho_mod, epochs, mu; require_convergence = false)
        f = compare_models(fits)

        init = f.para
        push!(chain, fits)
        push!(corrections, diff)
        push!(corvars, vard)
        if iter > 1
            deltacorr = (corrections[iter] .- corrections[iter-1]) ./ sqrt.(corvars[iter] .+ corvars[iter-1] .+ 1e-10)
            if abs(mean(deltacorr)) < 1/sqrt(length(deltacorr))
                break
            end
        end
    end

    bestchain = map(c -> compare_models(c), chain)
    f = bestchain[end]
    estimate = zeros(length(f.para))
    estimate_sd = zeros(length(f.para))
    evidence = 0
    lp = 0
    correction = zeros(length(h_obs.weights))
    sample_size = 0
    mask = map(c -> c.converged && !isinf(evd(c)), bestchain)
    chain_ = bestchain[mask]
    conv = true
    if isempty(chain_)
        @warn "fits did not converge or have infinite evidence"
        conv = false
        chain_= bestchain
        mask = fill(true, length(bestchain))
    elseif length(chain_) < top
        @warn "not enough converged fits found, using only $(length(chain_)) fits"
    end
    for j in 1:min(length(chain_),top)
        if length(chain_[end-j+1].para) != length(estimate)
            continue
        end
        estimate .+= chain_[end-j+1].para
        estimate_sd .+= sds(chain_[end-j+1]) .^2
        evidence += evd(chain_[end-j+1])
        lp += chain_[end-j+1].lp
        correction .+= corrections[end-j+1]
        sample_size += 1
    end
    estimate ./= sample_size
    estimate_sd .= sqrt.(estimate_sd./sample_size)
    evidence /= sample_size
    lp /= sample_size
    correction ./= sample_size
    corrected_weights = integral_ws(h_obs.edges[1].edges, mu, estimate) .+ correction
    
    zscore = estimate ./ estimate_sd
    p = map(z -> StatsAPI.pvalue(Distributions.Normal(), z; tail=:right), zscore)
    # Confidence interval (CI)
    q = Statistics.quantile(Distributions.Normal(), (1 + level) / 2)
    ci_low = estimate .- q .* estimate_sd
    ci_high = estimate .+ q .* estimate_sd

    FitResult(
        f.nepochs,
        f.bin,
        f.mu,
        estimate,
        estimate_sd,
        f.para_name,
        estimate,
        "iterative fit",
        conv,
        lp,
        evidence,
        (;
            chain, corrections, corvars, sample_size,
            zscore,
            pvalues = p, ci_low, ci_high,
            h_obs, corrected_weights
        )
    )
end

"""
    demoinfer_(h, epochrange, mu, rho, Ltot; kwargs...)

Infer demographic histories with piece-wise constant epochs
where the number of epochs is in `epochrange`. Takes a histogram
as input and the total length of the IBS segments.

It is much lighter to distribute the histogram than the vector of segments
which may also be streamed directly from disk into the histogram.
"""
function demoinfer_(h::Histogram{T,1,E}, epochrange::UnitRange{Int}, mu::Float64, rho::Float64, Ltot::Number;
    fop::FitOptions = FitOptions(Ltot),
    annealing = nothing,
    kwargs...
) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    @assert !isempty(h.weights) "histogram is empty"
    @assert first(epochrange) > 0 "epochrange has to be strictly positive"
    if Ltot != fop.Ltot
        @warn "inconsistent Ltot and fit options, taking Ltot"
        fop.Ltot = Ltot
    end
    f = pre_fit!(fop, h, last(epochrange), mu; require_convergence = false)
    nepochs = length(f)
    if nepochs < last(epochrange)
        @warn "for models above $nepochs no signal was found, stopping at $nepochs"
    end
    for i in f
        if !i.converged
            @warn "optimization with $(i.nepochs) did not converge"
        end
    end
    epochrange = first(epochrange):nepochs

    if isnothing(annealing)
        target = 2e8
        thetaL = sum(h.weights)
        factor = target / thetaL
        factor = max(1, factor)
        annealing = (L, it) -> factor
    end

    results = Vector{FitResult}(undef, length(epochrange))
    fops = Vector{FitOptions}(undef, length(epochrange))
    for i in eachindex(fops)
        fops[i] = deepcopy(fop)
    end
    @threads for n in eachindex(epochrange)
        epochs = epochrange[n]
        results[n] = demoinfer_(h, epochs, mu, rho, Ltot, get_para(f[epochs]);
            fop = fops[n], annealing, kwargs...
        )
    end

    return results
end

"""
    demoinfer_(h_obs::Histogram, nepochs::Int, mu::Float64, rho::Float64, Ltot::Number, init::Vector{Float64}; kwargs...)

Fit iteratively histogram `h_obs` with a single demographic model 
of piece-wise constant `nepochs` starting from an initial parameter vector `init`.

Return a `FitResult`, see [`FitResult`](@ref), above methods fall back to this,
which is called on multiple threads if available.
"""
function demoinfer_(h_obs::Histogram, nepochs::Int, mu::Float64, rho::Float64, Ltot::Number, init::Vector{Float64};
    fop::FitOptions = FitOptions(Ltot),
    annealing = Sq(),
    iters::Int = 8,
    s::Int = 1234,
    restart::Int = 100,
    top::Int = 1,
    level::Float64 = 0.95
)
    Random.seed!(s)

    length(init)÷2 == nepochs || @error "init must be in TN format, with 2*nepochs elements"
    setnepochs!(fop, nepochs)

    h_sim = Histogram(h_obs.edges)
    ho_mod = Histogram(h_obs.edges)

    if fop.Ltot != Ltot
        @warn "inconsistent Ltot and FitOptions, taking Ltot"
        fop.Ltot = Ltot
    end

    chain = FitResult[]
    corrections = []

    init_ = copy(init)
    for iter in 1:iters
        if iter % restart == 1
            Random.seed!(s + iter)
            init_ = copy(init)
        end
        weights_th = integral_ws(h_obs.edges[1], mu, init_)
        factor = annealing(Ltot, (iter-1) % restart + 1)
        get_sim!(init_, h_sim, mu, rho; factor)
    
        ho_mod.weights .= h_obs.weights
    
        diff = (h_sim.weights/factor .- weights_th)
        diff[1:4] .= 0.
        temp = ho_mod.weights .- diff
        temp .= round.(Int, temp)
        ho_mod.weights .= max.(temp, 0)
        
        setinit!(fop, init_)
        f = fit_model_epochs!(fop, ho_mod, mu)
        f = perturb_fit!(f, fop, ho_mod, mu)

        init_ = f.para
        push!(chain, f)
        push!(corrections, diff)
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
        correction .+= corrections[end-j+1]
        sample_size += 1
    end
    estimate ./= sample_size
    estimate_sd .= sqrt.(estimate_sd./sample_size)
    evidence /= sample_size
    lp /= sample_size
    correction ./= sample_size
    corrected_weights = integral_ws(h_obs.edges[1].edges, mu, estimate) .+ correction
    
    zscore = estimate ./ estimate_sd
    p = map(z -> StatsAPI.pvalue(Distributions.Normal(), z; tail=:right), zscore)
    # Confidence interval (CI)
    q = Statistics.quantile(Distributions.Normal(), (1 + level) / 2)
    ci_low = estimate .- q .* estimate_sd
    ci_high = estimate .+ q .* estimate_sd

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
    compare_models(models::Vector{FitResult})

Compare the models parameterized by `FitResult`s and return the best one.

### Theoretical explanation
TBD
"""
function compare_models(models::Vector{FitResult})
    ms = copy(models)
    filter!(m->isfinite(evd(m)) && m.converged, ms)
    if isempty(ms)
        @warn "none of the models is meaningful"
        return nothing
    end
    best = 1
    lp = ms[1].lp
    ev = evd(ms[1])
    mono = true
    for i in eachindex(ms)
        if evd(ms[i]) > ev && ms[i].lp >= lp
            best = i
            lp = ms[i].lp
            ev = evd(ms[i])
        elseif ms[i].lp < lp && mono
            @warn """
                log-likelihood is not monotonic in the number of epochs.
                This means that at least one likelihood optimization
                has probably failed. You may want to change the fit options.
            """
            mono = false
        end
    end
    return ms[best]
end