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

function getHessian(m; hessian_function=ForwardDiff.hessian, kwargs...)
    # Calculate Hessian. Adapted from Turing/src/optimisation/Optimisation.jl

    # Convert the values to their unconstrained states to make sure the
    # Hessian is computed with respect to the untransformed parameters.
    linked = DynamicPPL.istrans(m.f.varinfo)
    if linked
        # Setfield.@set! m.f.varinfo = DynamicPPL.invlink!!(m.f.varinfo, m.f.model)
        m = Accessors.@set m.f.varinfo = DynamicPPL.invlink!!(m.f.varinfo, m.f.model)
    end

    # Calculate the Hessian.
    H = hessian_function(m.f, m.values.array[:, 1])

    # Link it back if we invlinked it.
    if linked
        # Setfield.@set! m.f.varinfo = DynamicPPL.link!!(m.f.varinfo, m.f.model)
        m = Accessors.@set m.f.varinfo = DynamicPPL.link!!(m.f.varinfo, m.f.model)
    end

    return H
end

# models

@model function model_epochs_integral(edges::Vector, counts::Vector, mu::Float64, TNdists)
    TN ~ arraydist(TNdists)
    a = 0.5
    last_hid_I = hid_integral(TN, mu, edges[1] - a)
    for i in eachindex(counts)
        @inbounds this_hid_I = hid_integral(TN, mu, edges[i+1] - a)
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

# --- fitting

fit_epochs(hist::StatsBase.Histogram, mu::Float64; kwargs...) = fit_epochs_integral(hist, mu; kwargs...)

function fit_epochs_integral(hist::StatsBase.Histogram, mu::Float64; 
    nepochs::Int = 1,
    Ltot = nothing,
    init = nothing,
    perturbation = nothing,
    solver = LBFGS(),
    opt = Optim.Options(;iterations = 20000, allow_f_increases=true, 
        time_limit = 600, g_tol = 5e-8),
    range_factor = 10,
    Tlow = 10, Tupp = 10000,
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
            append!(init, [1000, N])
        end
        # @show init
    else
        @assert length(init) == 2 * nepochs
    end
    
    # set the range for the parameters
    if isnothing(low) || isnothing(upp)
        low = [init[1]/2, init[2]/range_factor] # changed L boundaries
        upp = [init[1]+10., init[2]*range_factor]
        for i in 2:nepochs
            append!(low, [Tlow, Nlow])
            append!(upp, [Tupp, Nupp])
        end
    end
    TNd = Uniform.(low, upp)
    
    # perturb the initial guess
    pinit = copy(init)
    if !isnothing(perturbation)
        pinit = map(pinit, low, upp) do p, l, u
            if perturbation < 1
                rand(truncated(LogNormal(log(p), perturbation), l, u))
            else
                rand(Uniform(l, u))
            end
        end
    end

    # run the optimization
    model = model_epochs_integral(edges, counts, mu, TNd)

    mle = optimize(model, MLE(), pinit, solver, opt)

    
    para = vec(mle.values)
    para_name = DemoInfer.correct_name.(string.(names(mle.values, 1)))
    lp = -minimum(mle.optim_result)
    
    
    hess = DemoInfer.getHessian(mle)
    dethess = det(hess)
    
    at_uboundary = map((x,u) -> (x>u/1.01), para, upp)
    at_lboundary = map((l,x) -> (x<l*1.01), low, para)
    maxchange = maximum(abs.(para .- init))
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
        q = Statistics.quantile(Distributions.Normal(), (1 + level) / 2)
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
    
    evidence = lp + sum(log.(1.0 ./ (upp.-low)) .+ log(2*pi)) - 0.5 * log(max(dethess, 0))

    FitResult(
        nepochs,
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
            low, upp, pinit, init,
            maxchange,
            coeftable = ct, 
            zscore, pvalues = p, ci_low, ci_high,
            dethess)
    )
end
