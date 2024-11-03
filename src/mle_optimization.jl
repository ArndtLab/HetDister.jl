struct FitResult
    nepochs::Int
    bin::Int
    mu::Float64
    para
    para_name
    TN::Vector
    method::String
    converged::Bool
    lp::Float64
    opt
end

function Base.show(io::IO, f::FitResult) 
    model = (f.nepochs == 1 ? "stationary" : "$(f.nepochs) epochs") *
            (f.bin > 1 ? " (binned $(f.bin))" : "")
    print(io, "Fit ", model, " ")
    print(io, f.method, " ")
    print(io, f.converged ? "●" : "○", " ")
    print(io, "[", @sprintf("%.1e",f.para[1]))
    for i in 2:length(f.para)
        print(io, " ,", @sprintf("%.1f",f.para[i]))
    end
    print(io, "] ", @sprintf("%.3f",f.lp))
end


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

@model function model_epochs(r::Vector, counts::Vector, weights::Vector, mu::Float64, TNdists)
    TN ~ arraydist(TNdists)
    
    for i in eachindex(counts)
        m = weights[i] * hid(TN, mu, r[i])
        if (m <= 0) || isnan(m)
            Turing.@addlogprob! -Inf
            # Exit the model evaluation early
            return
        end
        counts[i] ~ Poisson(m)
    end
end
    
@model function model_epochs_log(r::Vector, logdensity::Vector, mu::Float64, TNdists)
    TN ~ arraydist(TNdists)
    sigma = 1
    for i in eachindex(r)
        m = hid(TN, mu, r[i])
        if (m <= 0) || isnan(m)
            Turing.@addlogprob! -Inf
            # Exit the model evaluation early
            return
        end
        logdensity[i] ~ Normal(log(m), sigma)
    end
end

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

@model function model_Ns_integral(edges::Vector, counts::Vector, mu::Float64, Ndists, Ts::Vector, L::Float64)
    Ns ~ arraydist(Ndists)

    a = 0.5
    last_hid_I = hid_integral(Ns[2:end], Ts, L, mu, edges[1] - a)
    for i in eachindex(counts)
        @inbounds this_hid_I = hid_integral(Ns[2:end], Ts, L, mu, edges[i+1] - a)
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

    # get a good initial guess
    if isnothing(init)
        Ltot = sum(midpoints(hist.edges[1]) .* hist.weights)
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
        low = [init[1]/range_factor, init[2]/range_factor]
        upp = [init[1]*range_factor, init[2]*range_factor]
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
                rand(Truncated(LogNormal(log(p), perturbation), l, u))
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
    
    at_boundary = map((l,x,u) -> (x<l*1.01) || (x>u/1.01), low, para, upp)
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
        para_name,
        para,
        summary(mle.optim_result),
        Optim.converged(mle.optim_result),
        lp,
        (; 
            mle.optim_result,
            at_any_boundary = any(at_boundary), 
            at_boundary,
            low, upp, pinit, init,
            maxchange,
            coeftable = ct, 
            stderrors, zscore, pvalues = p, ci_low, ci_high,
            evidence, dethess)
    )
end

function fit_Ns_integral(hist::StatsBase.Histogram, mu::Float64, Ts::Vector{Float64}, Nstat::Float64, L::Float64;
    perturbation = nothing,
    solver = LBFGS(),
    opt = Optim.Options(;iterations = 20000, allow_f_increases=true, 
        time_limit = 600, g_tol = 5e-8),
    Nlow = 10., Nupp = 100000.,
    range_factor = 10,
    level = 0.95
)

    edges = hist.edges[1][:]
    counts = hist.weights
    @assert length(edges) - 1 == length(counts)
    
    # set the range for the parameters
    low = [Nlow for _ in 1:length(Ts)+1]
    upp = [Nupp for _ in 1:length(Ts)+1]
    # pushfirst!(low, L/range_factor)
    # pushfirst!(upp, L*range_factor)
    Nd = Uniform.(low, upp)

    init = [Nstat]
    append!(init, [Nstat for _ in 1:length(Ts)])
    
    # perturb the initial guess
    pinit = copy(init)
    if !isnothing(perturbation)
        pinit = map(pinit, low, upp) do p, l, u
            if perturbation < 1
                rand(Truncated(LogNormal(log(p), perturbation), l, u))
            else
                rand(Uniform(l, u))
            end
        end
    end

    # run the optimization
    model = model_Ns_integral(edges, counts, mu, Nd, Ts, L)

    mle = optimize(model, MLE(), pinit, solver, opt)

    
    para = vec(mle.values)
    para_name = DemoInfer.correct_name.(string.(names(mle.values, 1)))
    lp = -minimum(mle.optim_result)
    
    
    hess = DemoInfer.getHessian(mle)
    dethess = det(hess)
    
    at_boundary = map((l,x,u) -> (x<l*1.01) || (x>u/1.01), low, para, upp)
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
        length(Ts)+1,
        length(counts),
        mu, 
        para,
        para_name,
        para,
        summary(mle.optim_result),
        Optim.converged(mle.optim_result),
        lp,
        (; 
            mle.optim_result,
            at_any_boundary = any(at_boundary), 
            at_boundary,
            low, upp, pinit, init,
            maxchange,
            coeftable = ct, 
            stderrors, zscore, pvalues = p, ci_low, ci_high,
            evidence, dethess)
    )
end

function fit_epochs_mids(hist::StatsBase.Histogram, mu::Float64; kwargs...)

    obs_x = midpoints(hist.edges[1])
    obs_y = hist.weights
    obs_w = StatsBase.binvolume.(Ref(hist), 1:length(hist.edges[1])-1)
    fit_epochs_mids(obs_x, obs_y, obs_w, mu; kwargs...)
end

function fit_epochs_mids(obs_x::Vector, obs_y::Vector, mu::Float64; kwargs...)
    fit_epochs_mids(obs_x, obs_y, ones(length(obs_x)), mu; kwargs...)
end

function fit_epochs_mids(obs_x::Vector, obs_y::Vector, obs_w::Vector, mu::Float64;
    nepochs::Int = 1,
    init = nothing,
    solver = LBFGS(),
    opt = Optim.Options(;iterations = 20000, allow_f_increases=true, 
        time_limit = 600, g_tol = 5e-8),
    range_factor = 10,
    Tlow = 10, Tupp = 10000,
    Nlow = 10, Nupp = 100000,
    low = nothing, upp = nothing,
    level = 0.95
)

    if isnothing(init)
        Ltot = sum(obs_x .* obs_y)
        N = 1/(4*mu*(Ltot/sum(obs_y)))
        pinit = [Ltot, N]
        for i in 2:nepochs
            append!(pinit, [1000, N])
        end
    else
        pinit = copy(init)
        @assert length(pinit) == 2 * nepochs
    end

    low = [pinit[1]/range_factor, pinit[2]/range_factor]
    upp = [pinit[1]*range_factor, pinit[2]*range_factor]
    for i in 2:nepochs
        append!(low, [Tlow, Nlow])
        append!(upp, [Tupp, Nupp])
    end
    TNd = Uniform.(low, upp)


  

   
    model = DemoInfer.model_epochs(obs_x, obs_y, obs_w, mu, TNd)

    mle = optimize(model, MLE(), pinit, solver, opt)

    
    para = vec(mle.values)
    para_name = DemoInfer.correct_name.(string.(names(mle.values, 1)))
    lp = -minimum(mle.optim_result)
    
    im = informationmatrix(mle)
    detim = det(im)

    hess = getHessian(mle)
    dethess = det(hess)
    

    at_boundary = map((l,x,u) -> (x<l*1.01) || (x>u/1.01), low, para, upp)
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
        length(obs_x),
        mu, 
        para,
        para_name,
        para,
        summary(mle.optim_result),
        Optim.converged(mle.optim_result),
        lp,
        (; 
            mle.optim_result,
            at_any_boundary = any(at_boundary), 
            at_boundary,
            low, upp, pinit,
            coeftable = ct, 
            stderrors, zscore, pvalues = p, ci_low, ci_high,
            evidence, detim, dethess)
    )
end




