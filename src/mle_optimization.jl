function getHessian(m::Turing.Optimisation.ModeResult; kwargs...)
    return Turing.Optimisation.StatsBase.informationmatrix(m; kwargs...)
end

# models

@model function model_epochs(edges::AbstractVector{<:Integer}, 
    counts::AbstractVector{<:Integer}, mu::Float64,
    TNdists::Vector{<:Distribution}
)
    TN ~ arraydist(TNdists)
    a = 0.5
    last_hid_I = laplacekingmanint(edges[1] - a, mu, TN)
    for i in eachindex(counts)
        @inbounds this_hid_I = laplacekingmanint(edges[i+1] - a, mu, TN)
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

@model function modelsmcp!(dc::IntegralArrays, rs::AbstractVector{<:Real}, 
    edges::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer},
    mu::Float64, rho::Float64, TNdists::Vector{<:Distribution}
)
    TN ~ arraydist(TNdists)
    mldsmcp!(dc, 1:dc.order, rs, edges, mu, rho, TN)
    m = get_tmp(dc.ys, eltype(TN))
    m .*= diff(edges)
    for i in eachindex(counts)
        if (m[i] < 0) || isnan(m[i])
            # this happens when evaluating the model
            # after optimization, in the unconstrained
            # space, using Bijectors.
            # I could not find a mwe, (TODO: find one)
            # probably out of domain, apply a penalty
            m[i] = 0
        end
        @inbounds counts[i] ~ Poisson(m[i])
    end
end

function llsmcp!(dc::IntegralArrays, rs::AbstractVector{<:Real}, 
    edges::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer},
    mu::Float64, rho::Float64, TN::AbstractVector{<:Real}
)
    mldsmcp!(dc, 1:dc.order, rs, edges, mu, rho, TN)
    m = get_tmp(dc.ys, eltype(TN))
    m .*= diff(edges)
    ll = 0
    for i in eachindex(counts)
        if (m[i] < 0) || isnan(m[i])
            # this happens when evaluating the model
            # after optimization, in the unconstrained
            # space, using Bijectors.
            # I could not find a mwe, (TODO: find one)
            # probably out of domain, apply a penalty
            m[i] = 0
        end
        @inbounds ll += logpdf(Poisson(m[i]),counts[i])
    end
    return -ll
end

# --- fitting

function fit_model_epochs!(options::FitOptions, h::Histogram{T,1,E}
) where {T<:Integer,E<:Tuple{AbstractVector{<:Integer}}}
    fit_model_epochs!(options, h.edges[1], h.weights, Val(isnaive(options)))
end


function fit_model_epochs!(
    options::FitOptions, edges::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer}, 
    ::Val{true}
)
    # get a good initial guess
    iszero(options.init) && initialize!(options, counts)

    model = model_epochs(edges, counts, options.mu, options.prior)
    logger = ConsoleLogger(stdout, Logging.Error)
    mle = with_logger(logger) do
        Optim.optimize(model, MLE(), options.init, options.solver, options.opt)
    end
    return getFitResult(mle, options, counts)
end

function fit_model_epochs!(
    options::FitOptions, edges::AbstractVector{<:Integer}, counts::AbstractVector{<:Integer}, 
    ::Val{false}
)

    # get a good initial guess
    iszero(options.init) && initialize!(options, counts)

    # run the optimization
    rs = midpoints(edges)
    dc = IntegralArrays(options.order, options.ndt, length(rs), Val{length(options.init)}, 3)
    model = modelsmcp!(dc, rs, edges, counts, options.mu, options.rho, options.prior)
    logger = ConsoleLogger(stdout, Logging.Error)
    mle = with_logger(logger) do
        Optim.optimize(model, MLE(), options.init, options.solver, options.opt)
    end
    return getFitResult(mle, options, counts)
end

function getFitResult(mle, options::FitOptions, counts)
    para = vec(mle.values)
    lp = -minimum(mle.optim_result)
    
    hess = getHessian(mle)
    return getFitResult(hess, para, lp, mle.optim_result, options, counts)
end

function getFitResult(hess, para, lp, optim_result, options::FitOptions, counts)
    eigen_problem = eigen(hess)
    lambdas = eigen_problem.values

    at_uboundary = map((x,u) -> (x>u/1.05), para, options.upp)
    at_lboundary = map((l,x) -> (x<l*1.05), options.low, para)
    stderrors = fill(Inf, length(para))
    zscore = fill(0.0, length(para))
    p = fill(1, length(para))
    ci_low = fill(-Inf, length(para))
    ci_high = fill(Inf, length(para))
    logevidence = -Inf
    manual_flag = true
    if isreal(lambdas)
        lambdas = real.(lambdas)
        # the smallest eigenvalue can be slightly negative
        # because L is not a demographic parameter
        if any(lambdas[2:end] .< 0) || lambdas[1] < -eps()
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
        logevidence = lp + sum(log.(1.0 ./ (options.upp - options.low)) .+ 0.5*log(2*pi)) - 
            0.5 * sum(log.(lambdas))
    end

    FitResult(
        options.nepochs,
        length(counts),
        options.mu,
        options.rho,
        para,
        stderrors,
        summary(optim_result),
        Optim.converged(optim_result) && manual_flag,
        lp,
        logevidence,
        (;
            optim_result,
            at_any_boundary = any(at_uboundary) || any(at_lboundary), 
            at_uboundary, at_lboundary,
            options.low, options.upp, options.init,
            zscore, pvalues = p, ci_low, ci_high,
            eigen_problem.values)
    )
end