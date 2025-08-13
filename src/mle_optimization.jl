# helper functions

function correct_name(s)
    m = match(r"^TN\[(\d+)\]", s)
    if !isnothing(m)
        i = parse(Int, m.captures[1])
        if i == 1
            return "L"
        elseif iseven(i)
            return "N" * string((i-2) ÷ 2)
        else
            return "T" * string((i-1) ÷ 2)
        end
    end
    m = match(r"^N\[(\d+)\]", s)
    if !isnothing(m)
        return  "N" *string(parse(Int, m.captures[1]) - 1)
    end
    m = match(r"^T\[(\d+)\]", s)
    if !isnothing(m)
        return  "T" * m.captures[1]
    end
    return s
end

function getHessian(m::Turing.Optimisation.ModeResult; kwargs...)
    return Turing.Optimisation.StatsBase.informationmatrix(m; kwargs...)
end

# models

@model function model_epochs(edges::Vector, counts::Vector, mu::Float64, TNdists)
    TN ~ arraydist(TNdists)
    a = 0.5
    last_hid_I = hid_integral(TN, mu, edges[1] - a)
    for i in eachindex(counts)
        @inbounds this_hid_I = hid_integral(TN, mu, edges[i+1] - a)
        m = this_hid_I - last_hid_I
        last_hid_I = this_hid_I
        # if (m <= 0) || isnan(m)
        #     Turing.@addlogprob! -Inf
        #     # Exit the model evaluation early
        #     return
        # end
        @inbounds counts[i] ~ Poisson(m)
    end
end

# --- fitting

function fit_model_epochs(hist::StatsBase.Histogram, mu::Float64, options::FitOptions{T}) where {T<:FitKind}
    edges = hist.edges[1].edges
    counts = hist.weights

    # get a good initial guess
    iszero(options.init) && setinit!(options, hist, mu)
    
    # set the prior for the parameters
    TNd = Uniform.(options.low, options.upp)
    
    # perturb the initial guess
    pinit = copy(options.init)
    if !isempty(options.perturbations)
        for p in options.perturbations
            if p.factor < 1
                pinit[p.par] = rand(
                    truncated(
                        LogNormal(log(pinit[p.par]), p.factor),
                        options.low[p.par],
                        options.upp[p.par]
                    )
                )
            else
                rand(Uniform(options.low[p.par], options.upp[p.par]))
            end
        end
    end

    # run the optimization
    model = model_epochs(edges, counts, mu, TNd)

    mle = Optim.optimize(model, MLE(), pinit, options.solver, options.opt)
    
    para = vec(mle.values)
    para_name = DemoInfer.correct_name.(string.(names(mle.values, 1)))
    lp = -minimum(mle.optim_result)
    
    hess = DemoInfer.getHessian(mle)
    eigen_problem = eigen(hess)
    
    at_uboundary = map((x,u) -> (x>u/1.05), para, options.upp)
    at_lboundary = map((l,x) -> (x<l*1.05), options.low, para)
    maxchange = maximum(abs.(para .- options.init))
    stderrors = fill(Inf, length(para))
    zscore = fill(0.0, length(para))
    p = fill(1, length(para))
    ci_low = fill(-Inf, length(para))
    ci_high = fill(Inf, length(para))
    try 
        # stderrors = StatsBase.stderror(mle)
        stderrors = sqrt.(diag(pinv(hess)))
        zscore = para ./ stderrors
        p = map(z -> StatsAPI.pvalue(Distributions.Normal(), z; tail=:right), zscore)
    
        # Confidence interval (CI)
        q = Statistics.quantile(Distributions.Normal(), (1 + options.level) / 2)
        ci_low = para .- q .* stderrors
        ci_high = para .+ q .* stderrors
    catch
        # most likely computing stderrors failed
        # we stay with the default values
    end

    ct= try
        coeftable(mle)
    catch
        nothing
    end
    
    # assuming uniform prior
    lambdas = eigen_problem.values
    evidence = -Inf
    if isreal(lambdas)
        lambdas = real.(lambdas)
        lambdas[lambdas .< 0] .= eps()
        evidence = lp + sum(log.(1.0 ./ (options.upp.-options.low)) .+ 0.5*log(2*pi)) - 
            0.5 * sum(log.(lambdas))
    end

    FitResult(
        options.nepochs,
        length(counts),
        mu, 
        para,
        stderrors,
        para_name,
        para,
        summary(mle.optim_result),
        Optim.converged(mle.optim_result),
        lp,
        evidence,
        (; 
            mle.optim_result,
            at_any_boundary = any(at_uboundary) || any(at_lboundary), 
            at_uboundary, at_lboundary,
            options.low, options.upp, pinit, options.init,
            maxchange,
            coeftable = ct, 
            zscore, pvalues = p, ci_low, ci_high,
            eigen_problem.values)
    )
end