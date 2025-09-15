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

@model function model_epochs(edges::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer}, mu::Float64, TNdists::Vector{<:Distribution})
    TN ~ arraydist(TNdists)
    a = 0.5
    last_hid_I = hid_integral(TN, mu, edges[1] - a)
    for i in eachindex(counts)
        @inbounds this_hid_I = hid_integral(TN, mu, edges[i+1] - a)
        m = this_hid_I - last_hid_I
        last_hid_I = this_hid_I
        if (m < 0) || isnan(m)
            # this happens when evaluating the model
            # after optimization, in the unconstrained
            # space, using Bijectors.
            # I could not find a mwe, (TODO: find one)
            # probably out of domain, apply a penalty
            m = 0
        end
        @inbounds counts[i] ~ Poisson(m)
    end
end

# --- fitting

function fit_model_epochs!(options::FitOptions, h::Histogram{T,1,E}, mu::Float64) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    fit_model_epochs!(options, h.edges[1], h.weights, mu)
end


function fit_model_epochs!(
    options::FitOptions, edges::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer}, mu::Float64
)

    # get a good initial guess
    iszero(options.init) && setinit!(options, counts, mu)

    # run the optimization
    model = model_epochs(edges, counts, mu, options.prior)

    logger = ConsoleLogger(stdout, Logging.Error)
    mle = with_logger(logger) do
        Optim.optimize(model, MLE(), options.init, options.solver, options.opt)
    end

    para = vec(mle.values)
    para_name = DemoInfer.correct_name.(string.(names(mle.values, 1)))
    lp = -minimum(mle.optim_result)
    
    hess = DemoInfer.getHessian(mle)
    eigen_problem = eigen(hess)
    lambdas = eigen_problem.values

    at_uboundary = map((x,u) -> (x>u/1.05), para, options.upp)
    at_lboundary = map((l,x) -> (x<l*1.05), options.low, para)
    stderrors = fill(Inf, length(para))
    zscore = fill(0.0, length(para))
    p = fill(1, length(para))
    ci_low = fill(-Inf, length(para))
    ci_high = fill(Inf, length(para))
    evidence = -Inf
    manual_flag = true
    if isreal(lambdas)
        lambdas = real.(lambdas)
        if any(lambdas .< 0)
            manual_flag = false
        end
        lambdas[lambdas .<= 0] .= eps()
        vars_ = diag( eigen_problem.vectors *
            diagm(inv.(lambdas)) * eigen_problem.vectors'
        )
        stderrors = sqrt.(vars_)
        zscore = para ./ stderrors
        p = map(z -> StatsAPI.pvalue(Distributions.Normal(), z; tail=:right), zscore)
    
        # Confidence interval (CI)
        q = Statistics.quantile(Distributions.Normal(), (1 + options.level) / 2)
        ci_low = para .- q .* stderrors
        ci_high = para .+ q .* stderrors
    
        # assuming uniform prior
        evidence = lp + sum(log.(1.0 ./ (options.upp - options.low)) .+ 0.5*log(2*pi)) - 
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
        Optim.converged(mle.optim_result) && manual_flag,
        lp,
        evidence,
        (; 
            mle.optim_result,
            at_any_boundary = any(at_uboundary) || any(at_lboundary), 
            at_uboundary, at_lboundary,
            options.low, options.upp, options.init,
            zscore, pvalues = p, ci_low, ci_high,
            eigen_problem.values)
    )
end