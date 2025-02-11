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

function ll_epochs(edges::Vector, counts::Vector, mu::Float64, TN)
    a = 0.5
    ll = 0.
    last_hid_I = hid_integral(TN, mu, edges[1] - a)
    for i in eachindex(counts)
        @inbounds this_hid_I = hid_integral(TN, mu, edges[i+1] - a)
        m = this_hid_I - last_hid_I
        last_hid_I = this_hid_I
        if isnan(m) || (m <= 0)
            return Inf
        end
        @inbounds ll -= logpdf(Poisson(m), counts[i])
    end
    return ll #log ?
end

function ll_single(edges::Vector, counts::Vector, mu::Float64, fixedTN, par, parindex)
    temp = [fixedTN[1:parindex-1];[par];fixedTN[parindex+1:end]]
    return ll_epochs(edges, counts, mu, temp)
end

function chi_epochs(edges::Vector, counts::Vector, mu::Float64, TN)
    a = 0.5
    chi = 0.
    last_hid_I = hid_integral(TN, mu, edges[1] - a)
    for i in eachindex(counts)
        @inbounds this_hid_I = hid_integral(TN, mu, edges[i+1] - a)
        m = this_hid_I - last_hid_I
        last_hid_I = this_hid_I
        if (m <= 0) || isnan(m)
            return Inf
        end
        chi += (counts[i] - m)^2 / m
    end
    return chi
end

@model function model_sizes(edges::Vector, counts::Vector, mu::Float64, Ts::Vector, Ndists)
    # Ndists has L prior as the first element
    Ns ~ arraydist(Ndists)
    a = 0.5
    last_hid_I = hid_integral(Ns[2:end], Ts, Ns[1], mu, edges[1] - a)
    for i in eachindex(counts)
        @inbounds this_hid_I = hid_integral(Ns[2:end], Ts, Ns[1], mu, edges[i+1] - a)
        m = this_hid_I - last_hid_I
        last_hid_I = this_hid_I
        if (m <= 0) || isnan(m)
            Turing.@addlogprob! -Inf
            # Exit the model evaluation early
            return
        end
        @inbounds counts[i] ~ Poisson(m)
    end
end

function ll_sizes(edges::Vector, counts::Vector, mu::Float64, Ts::Vector, Ns)
    # Ns has L as the first element
    a = 0.5
    ll = 0.
    last_hid_I = hid_integral(Ns[2:end], Ts, Ns[1], mu, edges[1] - a)
    for i in eachindex(counts)
        @inbounds this_hid_I = hid_integral(Ns[2:end], Ts, Ns[1], mu, edges[i+1] - a)
        m = this_hid_I - last_hid_I
        last_hid_I = this_hid_I
        if (m <= 0) || isnan(m)
            ll = -Inf
            # Exit the model evaluation early
            return ll
        end
        @inbounds ll -= logpdf(Poisson(m), counts[i])
    end
    return ll
end


function chi_sizes(edges::Vector, counts::Vector, mu::Float64, Ts, Ns)
    a = 0.5
    chi = 0.
    last_hid_I = hid_integral(Ns[2:end], Ts, Ns[1], mu, edges[1] - a)
    for i in eachindex(counts)
        @inbounds this_hid_I = hid_integral(Ns[2:end], Ts, Ns[1], mu, edges[i+1] - a)
        m = this_hid_I - last_hid_I
        last_hid_I = this_hid_I
        if (m <= 0) || isnan(m)
            return Inf
        end
        chi += (counts[i] - m)^2 / m
    end
    return chi
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
            if p.isrnd 
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
            else
                pinit[p.par] *= p.factor
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
    lambdas[lambdas .< 0] .= eps()
    evidence = lp + sum(log.(1.0 ./ (options.upp.-options.low)) .+ 0.5*log(2*pi)) - 
        0.5 * sum(log.(lambdas))

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




















# --- unmaintained relicts

function fit_chill_epochs(hist::StatsBase.Histogram, mu::Float64, options::FitOptions{T}) where {T<:FitKind}
    edges = hist.edges[1].edges
    counts = hist.weights

    # get a good initial guess
    iszero(options.init) && setinit!(options, hist, mu)
    
    # perturb the initial guess
    pinit = copy(options.init)
    if !isempty(options.perturbations)
        for p in options.perturbations
            if p.isrnd 
                if p.factor < 1
                    pinit[p.par] = rand(
                        truncated(LogNormal(log(pinit[p.par]), p.factor), low[p.par], upp[p.par])
                        )
                else
                    rand(Uniform(low[p.par], upp[p.par])) # maybe TriangularDist(l, u, p) ?
                end
            else
                pinit[p.par] *= p.factor
            end
        end
    end

    # run the optimization
    model = x -> ll_epochs(edges, counts, mu, x)

    # mle = nothing
    # if options.solver == Optim.IPNewton()
    #     dfc = TwiceDifferentiableConstraints(options.low, options.upp)
    #     mle = Optim.optimize(model, dfc, pinit, options.solver; autodiff=:forward)
    # elseif options.solver == NelderMead()
    #     mle = Optim.optimize(model, options.low, options.upp, pinit, Fminbox(options.solver))
    # else
    #     mle = Optim.optimize(model, options.low, options.upp, pinit, Fminbox(options.solver); autodiff=:forward)
    # end

    f = OptimizationFunction((x,p) -> ll_epochs(edges, counts, mu, x))
    prob = Optimization.OptimizationProblem(f, pinit, nothing, lb = options.low, ub = options.upp)
    mle = solve(prob, ABC(), maxtime=600, maxiters=100_000)

    para = mle.minimizer
    lp = -minimum(mle)
    
    hess = ForwardDiff.hessian(model, para)
    dethess = det(hess)
    
    at_uboundary = map((x,u) -> (x>u/1.05), para, options.upp)
    at_lboundary = map((l,x) -> (x<l*1.05), options.low, para)
    maxchange = maximum(abs.(para .- options.init))
    stderrors = fill(Inf, length(para))
    zscore = fill(0.0, length(para))
    p = fill(1, length(para))
    ci_low = fill(-Inf, length(para))
    ci_high = fill(Inf, length(para))
    try 
        stderrors = diag(hess)
        zscore = para ./ stderrors
        p = map(z -> StatsAPI.pvalue(Distributions.Normal(), z; tail=:both), zscore)
    
        # Confidence interval (CI)
        q = Statistics.quantile(Distributions.Normal(), (1 + options.level) / 2)
        ci_low = para .- q .* stderrors
        ci_high = para .+ q .* stderrors
    catch
        # most likely computing stderrors failed
        # we stay with the default values
    end
    
    evidence = lp + sum(log.(1.0 ./ (options.upp.-options.low)) .+ log(2*pi)) - 0.5 * log(max(dethess, 0))

    FitResult(
        options.nepochs,
        length(counts),
        mu, 
        para,
        stderrors,
        nothing,
        para,
        summary(mle),
        false, #Optim.converged(mle),
        lp,
        evidence,
        (; 
            mle,
            at_any_boundary = any(at_uboundary) || any(at_lboundary), 
            at_uboundary, at_lboundary,
            options.low, options.upp, pinit, options.init,
            maxchange, 
            zscore, pvalues = p, ci_low, ci_high,
            dethess)
    )
end

function fit_ll_single(hist::StatsBase.Histogram, mu::Float64, options::FitOptions{T}, fixedTN, parindex) where {T<:FitKind}
    edges = hist.edges[1].edges
    counts = hist.weights

    # get a good initial guess
    iszero(options.init) && setinit!(options, hist, mu)
    
    # perturb the initial guess
    pinit = copy(options.init)
    if !isempty(options.perturbations)
        for p in options.perturbations
            if p.isrnd 
                if p.factor < 1
                    pinit[p.par] = rand(
                        truncated(LogNormal(log(pinit[p.par]), p.factor), low[p.par], upp[p.par])
                        )
                else
                    rand(Uniform(low[p.par], upp[p.par])) # maybe TriangularDist(l, u, p) ?
                end
            else
                pinit[p.par] *= p.factor
            end
        end
    end

    # run the optimization
    model(x) = ll_single(edges, counts, mu, fixedTN, x, parindex)

    mle = nothing
    if options.solver == Optim.IPNewton()
        dfc = TwiceDifferentiableConstraints([1e1], [1e5])
        mle = Optim.optimize(model, dfc, [1e3], options.solver; autodiff=:forward)
    elseif options.solver == NelderMead()
        mle = Optim.optimize(model, options.low, options.upp, pinit, Fminbox(options.solver))
    else
        mle = Optim.optimize(model, [1e1], [1e5], [1e3], Fminbox(options.solver); autodiff=:forward)
    end

    # f = OptimizationFunction((x,p) -> ll_epochs(edges, counts, mu, x))
    # prob = Optimization.OptimizationProblem(f, pinit, nothing, lb = options.low, ub = options.upp)
    # mle = solve(prob, WOA(;N=100), maxtime=600, maxiters=100_000)

    para = mle.minimizer
    lp = -minimum(mle)
    return para, lp
    
    hess = ForwardDiff.hessian(model, para)
    dethess = det(hess)
    
    at_uboundary = map((x,u) -> (x>u/1.01), para, options.upp)
    at_lboundary = map((l,x) -> (x<l*1.01), options.low, para)
    maxchange = maximum(abs.(para .- options.init))
    stderrors = fill(Inf, length(para))
    zscore = fill(0.0, length(para))
    p = fill(1, length(para))
    ci_low = fill(-Inf, length(para))
    ci_high = fill(Inf, length(para))
    try 
        stderrors = diag(hess)
        zscore = para ./ stderrors
        p = map(z -> StatsAPI.pvalue(Distributions.Normal(), z; tail=:both), zscore)
    
        # Confidence interval (CI)
        q = Statistics.quantile(Distributions.Normal(), (1 + options.level) / 2)
        ci_low = para .- q .* stderrors
        ci_high = para .+ q .* stderrors
    catch
        # most likely computing stderrors failed
        # we stay with the default values
    end
    
    evidence = lp + sum(log.(1.0 ./ (options.upp.-options.low)) .+ log(2*pi)) - 0.5 * log(max(dethess, 0))

    FitResult(
        options.nepochs,
        length(counts),
        mu, 
        para,
        stderrors,
        nothing,
        para,
        summary(mle),
        Optim.converged(mle),
        lp,
        evidence,
        (; 
            mle,
            at_any_boundary = any(at_uboundary) || any(at_lboundary), 
            at_uboundary, at_lboundary,
            options.low, options.upp, pinit, options.init,
            maxchange, 
            zscore, pvalues = p, ci_low, ci_high,
            dethess)
    )
end

function chain_model_epochs(hist::StatsBase.Histogram, mu::Float64, options::FitOptions{T}) where {T<:FitKind}
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
            if p.isrnd 
                if p.factor < 1
                    pinit[p.par] = rand(
                        truncated(LogNormal(log(pinit[p.par]), p.factor), low[p.par], upp[p.par])
                        )
                else
                    rand(Uniform(low[p.par], upp[p.par])) # maybe TriangularDist(l, u, p) ?
                end
            else
                pinit[p.par] *= p.factor
            end
        end
    end

    # run the optimization
    model = model_epochs(edges, counts, mu, TNd)

    # chain = sample(model, NUTS(1000, 0.9), 10_000, initial_params=pinit)

    # Running pathfinder
    draws = 1_000
    result_multi = multipathfinder(model, draws; nruns=8)

    # Estimating the metric
    inv_metric = result_multi.pathfinder_results[1].fit_distribution.Σ
    metric = DenseEuclideanMetric(Matrix(inv_metric))

    # Creating an AdvancedHMC NUTS sampler with the custom metric.
    tap = 0.5
    nuts = AdvancedHMC.NUTS(tap; metric=metric)

    # Sample
    chain = sample(model, externalsampler(nuts), 10_000; n_adapts=1_000)

    return chain

    burnin = 1
    para = map(names(chain)[1:npar(options)]) do k
        mean(chain[k][burnin:end])
    end

    para_name = correct_name.(string.(names(chain)[1:npar(options)]))
    lp = maximum(chain[:lp])
    stderrors = map(keys(ch)[1:end-1]) do k
        std(ch[k][burnin:end])
    end
    
    hess = DemoInfer.getHessian(mle)
    dethess = det(hess)
    
    at_uboundary = map((x,u) -> (x>u/1.01), para, options.upp)
    at_lboundary = map((l,x) -> (x<l*1.01), options.low, para)
    maxchange = maximum(abs.(para .- options.init))
    stderrors = fill(Inf, length(para))
    zscore = fill(0.0, length(para))
    p = fill(1, length(para))
    ci_low = fill(-Inf, length(para))
    ci_high = fill(Inf, length(para))
    try 
        stderrors = StatsBase.stderror(mle)
        zscore = para ./ stderrors
        p = map(z -> StatsAPI.pvalue(Distributions.Normal(), z; tail=:both), zscore)
    
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
    
    evidence = lp + sum(log.(1.0 ./ (options.upp.-options.low)) .+ log(2*pi)) - 0.5 * log(max(dethess, 0))

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
            dethess)
    )
end


# --- cross entropy optimization

function stopcrit(gs::Vector{Float64}, d::Int, reltol::Float64)
    cur = length(gs)
    pr = length(gs) - 1
    if pr > 0
        cond = true
    else
        cond = false
    end
    while (pr > 0) && (pr > length(gs) - d)
        if abs((gs[cur] - gs[pr]) / gs[cur]) < reltol
            cond = cond & true
        else
            cond = false
        end
        pr -= 1
        cur -= 1
    end
    return cond
end

function modentropy(xs::Vector{Vector{Float64}}, μ::Vector{T}, σ::Vector{T}, opt::FitOptions) where {T<:Real}
    n = length(xs)
    s = 0.
    for i in 1:n
        mvpdf = 1.
        for j in eachindex(μ)
            if σ[j] <= 0
                return -Inf
            end
            mvpdf *= pdf(truncated(LogNormal(μ[j], σ[j]), opt.low[j], opt.upp[j]), xs[i][j])
        end
        s += log(mvpdf)
    end
    return s
end

function CEOptimization(obj::Function, q::Float64, opt::FitOptions, samsize::Int=1000, d::Int=5;
    maxiters::Int = 1000,
    reltol::Float64 = 1e-3
)
    dim = npar(opt)
    σ = [log(opt.upp[i] - opt.low[i]) / 3 for i in 1:dim]
    μ = log.(opt.init)
    chain = Vector{Float64}[]
    push!(chain, μ)

    gs = Float64[]
    it = 1 
    while it < maxiters && !stopcrit(gs, d, reltol)
        # sample from the current distribution (μ,σ)
        xs = map(1:samsize) do _
            map(1:dim) do i 
                rand(truncated(LogNormal(μ[i], σ[i]), opt.low[i], opt.upp[i]))
            end
        end
        # evaluate loglikelihood for each sample x and sort the (loglike,x) pair
        lls = obj.(xs)
        p = sortperm(lls)
        xs = xs[p]
        lls = lls[p]
        # compute the q quantile γ
        thr = ceil(Int, samsize * (1-q))
        γ = lls[thr]
        subset = xs[thr:end]
        # optimize distribution to concentrate on the q quantile
        model = x -> -modentropy(subset, x[1:dim], x[dim+1:end], opt)
        lower = [log.(opt.low); zeros(dim)]
        upper = [log.(opt.upp); σ*1.01]
        x0 = [μ; σ]
        optimum = Optim.optimize(model, lower, upper, x0, Fminbox(LBFGS()))
        μ = optimum.minimizer[1:dim]
        σ = optimum.minimizer[dim+1:end]
        if any(μ .< log.(opt.low)) || any(μ .> log.(opt.upp))
            @error "Optimization step went out of bounds"
            for i in 1:dim
                if μ[i] < log(opt.low[i])
                    μ[i] = log(opt.low[i])
                elseif μ[i] > log(opt.upp[i])
                    μ[i] = log(opt.upp[i])
                end
            end
        end
        push!(chain, μ)
        push!(gs, γ)
        it += 1
    end
    return chain, gs
end


































function fit_model_sizes(hist::StatsBase.Histogram, mu::Float64, Ts;
    nepochs::Int = 1,
    Ltot = nothing,
    init = nothing,
    perturbation = nothing,
    solver = LBFGS(),
    opt = Optim.Options(;iterations = 20000, allow_f_increases=true, 
        time_limit = 600, g_tol = 5e-8),
    range_factor = 10,
    Tlow = 10, Tupp = 100000,
    Nlow = 10, Nupp = 100000,
    low = nothing, upp = nothing,
    level = 0.95
)

    edges = hist.edges[1][:]
    counts = hist.weights
    @assert length(edges) - 1 == length(counts)

    if isnothing(Ltot)
        Ltot = sum(midpoints(hist.edges[1]) .* hist.weights)
        @warn "optional parameter Ltot inferred from histogram: this could lead to wrong results"
    end

    # get a good initial guess
    if isnothing(init)
        N = 1/(4*mu*(Ltot/sum(hist.weights)))
        init = [Ltot, N]
        for i in 2:nepochs
            push!(init, N)
        end
        # @show init
    else
        @assert length(init) == nepochs + 1
        @assert length(Ts) == nepochs - 1
    end
    
    # set the range for the parameters
    if isnothing(low) || isnothing(upp)
        low = [init[1]/2, init[2]/range_factor] # changed L boundaries
        upp = [init[1]+10., init[2]*range_factor]
        for i in 2:nepochs
            push!(low, Nlow)
            push!(upp, Nupp)
        end
    end
    # low .= log.(low)
    # upp .= log.(upp)
    # init = log.(init)
    Nd = Uniform.(low, upp)
    
    # perturb the initial guess
    pinit = copy(init)
    if !isnothing(perturbation)
        pinit = map(pinit, low, upp) do p, l, u
            if perturbation < 1
                rand(truncated(LogNormal(log(p), perturbation), l, u))
            else
                rand(Uniform(l, u)) # maybe TriangularDist(l, u, p) ?
            end
        end
    end

    # run the optimization
    model = model_sizes(edges, counts, mu, Ts, Nd)

    mle = optimize(model, MLE(), pinit, solver, opt)
    # chain = sample(model, Gibbs(MH(),MH(),MH(),MH(),MH(),MH()), 100_000, init_params=pinit)
    # chain = sample(model, NUTS(1000, 0.65), 10_000, initial_params=pinit)
    
    para = vec(mle.values)
    tn = copy(para)
    for i in eachindex(Ts)
        insert!(tn, 2i+1, Ts[i])
    end
    # para = exp.(para)
    # burnin = 1
    # para = map(names(chain)[1:2nepochs]) do k
        # mean(chain[k][burnin:end])
    # end
    para_name = DemoInfer.correct_name.(string.(names(mle.values, 1)))
    # para_name = correct_name.(string.(names(chain)[1:2nepochs]))
    lp = -minimum(mle.optim_result)
    # lp = maximum(chain[:lp])
    
    
    hess = DemoInfer.getHessian(mle)
    dethess = det(hess)
    # dethess = 0
    
    at_uboundary = map((x,u) -> (x>u/1.01), para, upp)
    at_lboundary = map((l,x) -> (x<l*1.01), low, para)
    maxchange = 0 #maximum(abs.(para .- init))
    tnstd = fill(Inf, length(para))
    zscore = fill(0.0, length(para))
    p = fill(1, length(para))
    ci_low = fill(-Inf, length(para))
    ci_high = fill(Inf, length(para))
    try 
        tnstd = StatsBase.stderror(mle)
        stderrors = copy(tnstd)
        for i in eachindex(Ts)
            insert!(tnstd, 2i+1, 1)
        end
        # stderrors = sqrt.( exp.(para).^2 .* StatsBase.stderror(mle) .^2 ) #temporary rough approximation
        # stderrors = map(keys(ch)[1:end-1]) do k
        #     std(ch[k][burnin:end])
        # end
        zscore = para ./ stderrors
        p = map(z -> StatsAPI.pvalue(Distributions.Normal(), z; tail=:both), zscore)
    
        # Confidence interval (CI)
        q = Statistics.quantile(Distributions.Normal(), (1 + level) / 2)
        ci_low = para .- q .* stderrors
        ci_high = para .+ q .* stderrors
    catch
        # most likely computing stderrors failed
        # we stay with the default values
        tnstd = fill(1., 2nepochs)
    end

    ct= try
        coeftable(mle)
    catch
        nothing
    end
    
    evidence = lp + sum(log.(1.0 ./ (upp.-low)) .+ log(2*pi)) - 0.5 * log(max(dethess, 0))
    # evidence = 0

    FitResult(
        nepochs,
        length(counts),
        mu, 
        tn,
        tnstd,
        para_name,
        tn,
        # "HMC",
        summary(mle.optim_result),
        # true,
        Optim.converged(mle.optim_result),
        lp,
        evidence,
        (; 
            mle.optim_result,
            # chain,
            at_any_boundary = any(at_uboundary) || any(at_lboundary), 
            at_uboundary, at_lboundary,
            low, upp, pinit, init,
            maxchange,
            coeftable = ct, 
            zscore, pvalues = p, ci_low, ci_high,
            dethess)
    )
end

function fit_chi_sizes(hist::StatsBase.Histogram, mu::Float64, Ts; 
    nepochs::Int = 1,
    Ltot = nothing,
    init = nothing,
    perturbation = nothing,
    solver = LBFGS(),
    opt = Optim.Options(;iterations = 20000, allow_f_increases=true, 
        time_limit = 600, g_tol = 5e-8),
    range_factor = 10,
    Tlow = 10, Tupp = 100000,
    Nlow = 10, Nupp = 100000,
    low = nothing, upp = nothing,
    level = 0.95,
    kwargs...
)

    edges = hist.edges[1][:]
    counts = hist.weights
    @assert length(edges) - 1 == length(counts)

    if isnothing(Ltot)
        Ltot = sum(midpoints(hist.edges[1]) .* hist.weights)
        @warn "optional parameter Ltot inferred from histogram: this could lead to wrong results"
    end

    # get a good initial guess
    if isnothing(init)
        N = 1/(4*mu*(Ltot/sum(hist.weights)))
        init = [Ltot, N]
        for i in 2:nepochs
            push!(init, N)
        end
        # @show init
    else
        @assert length(init) == nepochs + 1
        @assert length(Ts) == nepochs - 1
    end
    
    # set the range for the parameters
    if isnothing(low) || isnothing(upp)
        low = [init[1]/2, init[2]/range_factor] # changed L boundaries
        upp = [init[1]+10., init[2]*range_factor]
        for i in 2:nepochs
            push!(low, Nlow)
            push!(upp, Nupp)
        end
    end
    # low .= log.(low)
    # upp .= log.(upp)
    # init = log.(init)
    Nd = Uniform.(low, upp)
    
    # perturb the initial guess
    pinit = copy(init)
    if !isnothing(perturbation)
        pinit = map(pinit, low, upp) do p, l, u
            if perturbation < 1
                rand(truncated(LogNormal(log(p), perturbation), l, u))
            else
                rand(Uniform(l, u)) # maybe TriangularDist(l, u, p) ?
            end
        end
    end

    # run the optimization
    model = x -> chi_sizes(edges, counts, mu, Ts, x)

    opt = optimize(model, pinit, Adam(; kwargs...); autodiff=:forward)
    # chain = sample(model, Gibbs(MH(),MH(),MH(),MH(),MH(),MH()), 100_000, init_params=pinit)
    # chain = sample(model, NUTS(1000, 0.65), 10_000, initial_params=pinit)
    
    para = Optim.minimizer(opt)
    tn = copy(para)
    for i in eachindex(Ts)
        insert!(tn, 2i+1, Ts[i])
    end
    # para = exp.(para)
    # burnin = 1
    # para = map(names(chain)[1:2nepochs]) do k
        # mean(chain[k][burnin:end])
    # end
    # para_name = DemoInfer.correct_name.(string.(names(mle.values, 1)))
    # para_name = correct_name.(string.(names(chain)[1:2nepochs]))
    lp = Optim.minimum(opt)
    # lp = maximum(chain[:lp])
    
    
    # hess = DemoInfer.getHessian(mle)
    # dethess = det(hess)
    dethess = 0
    
    at_uboundary = map((x,u) -> (x>u/1.01), para, upp)
    at_lboundary = map((l,x) -> (x<l*1.01), low, para)
    maxchange = 0 #maximum(abs.(para .- init))
    tnstd = fill(Inf, length(para))
    zscore = fill(0.0, length(para))
    p = fill(1, length(para))
    ci_low = fill(-Inf, length(para))
    ci_high = fill(Inf, length(para))
    try 
        tnstd = StatsBase.stderror(mle)
        stderrors = copy(tnstd)
        for i in eachindex(Ts)
            insert!(tnstd, 2i+1, 1)
        end
        # stderrors = sqrt.( exp.(para).^2 .* StatsBase.stderror(mle) .^2 ) #temporary rough approximation
        # stderrors = map(keys(ch)[1:end-1]) do k
        #     std(ch[k][burnin:end])
        # end
        zscore = para ./ stderrors
        p = map(z -> StatsAPI.pvalue(Distributions.Normal(), z; tail=:both), zscore)
    
        # Confidence interval (CI)
        q = Statistics.quantile(Distributions.Normal(), (1 + level) / 2)
        ci_low = para .- q .* stderrors
        ci_high = para .+ q .* stderrors
    catch
        # most likely computing stderrors failed
        # we stay with the default values
        tnstd = fill(1., 2nepochs)
    end

    ct= try
        coeftable(mle)
    catch
        nothing
    end
    
    evidence = lp + sum(log.(1.0 ./ (upp.-low)) .+ log(2*pi)) - 0.5 * log(max(dethess, 0))
    # evidence = 0

    FitResult(
        nepochs,
        length(counts),
        mu, 
        tn,
        tnstd,
        nothing,
        tn,
        # "HMC",
        summary(opt),
        # true,
        Optim.converged(opt),
        lp,
        evidence,
        (; 
            # mle.optim_result,
            # chain,
            at_any_boundary = any(at_uboundary) || any(at_lboundary), 
            at_uboundary, at_lboundary,
            low, upp, pinit, init,
            maxchange,
            coeftable = ct, 
            zscore, pvalues = p, ci_low, ci_high,
            dethess)
    )
end



















# chain = sample(model, Gibbs(MH(),MH(),MH(),MH(),MH(),MH()), 100_000, init_params=pinit)
    # chain = sample(model, NUTS(1000, 0.65), 10_000, initial_params=pinit)

    # burnin = 1
    # para = map(names(chain)[1:2nepochs]) do k
        # mean(chain[k][burnin:end])
    # end

    # para_name = correct_name.(string.(names(chain)[1:2nepochs]))
# lp = maximum(chain[:lp])
# stderrors = map(keys(ch)[1:end-1]) do k
        #     std(ch[k][burnin:end])
        # end