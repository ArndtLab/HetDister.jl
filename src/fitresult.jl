"""
    struct FitResult

A data structure to store the results of a fit.

Some methods are defined for this type to get the vector of parameters, std errors, 
model evidence, etc. See [`get_para`](@ref), [`sds`](@ref), [`evd`](@ref), 
[`pop_sizes`](@ref), [`durations`](@ref).
"""
struct FitResult
    nepochs::Int
    bin::Int
    mu::Float64
    para::Vector
    stderrors::Vector
    para_name
    TN::Vector
    method::String
    converged::Bool
    lp::Float64
    evidence::Float64
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
        print(io, ", ", @sprintf("%.1f",f.para[i]))
    end
    print(io, "] ", @sprintf("logL %.3f",f.lp), @sprintf(" | evidence %.3f",f.evidence))
end

"""
    pars(fit::FitResult)

Return the parameters of the fit.
"""
get_para(fit::FitResult) = copy(fit.para)

"""
    sds(fit::FitResult)

Return the standard deviations of the parameters of the fit.
"""
sds(fit::FitResult) = copy(fit.stderrors)

"""
    evd(fit::FitResult)

Return the evidence of the fit.
"""
evd(fit::FitResult) = fit.evidence

"""
    pop_sizes(fit::FitResult)

Return the fitted population sizes, from past to present.
"""
pop_sizes(fit::FitResult) = fit.para[2:2:end]

"""
    durations(fit::FitResult)

Return the fitted durations of the epochs.
"""
durations(fit::FitResult) = fit.para[3:2:end-1]

"""
    get_chain(fit::FitResult)

Return two matrices containing the chain of fitted parameters
and std errors respectively (both as columns).
"""
function get_chain(fit::FitResult)
    if isempty(findall(keys(fit.opt) .== :chain))
        return fit.para, fit.stderrors
    end
    p = mapreduce(hcat, fit.opt.chain) do x
        get_para(x)
    end
    sd = mapreduce(hcat, fit.opt.chain) do x
        sds(x)
    end
    return p, sd
end

mutable struct Perturbation
    factor::Float64
    par::Int
end

abstract type FitKind end

struct EpochsFit <: FitKind end
struct SizesFit <: FitKind end

npar(nepochs::Int, ::EpochsFit) = 2nepochs
npar(nepochs::Int, ::SizesFit) = nepochs + 1

mutable struct FitOptions{T<:FitKind}
    nepochs::Int
    Ltot::Number
    init::Vector{Float64}
    Ts::Vector{Float64}
    perturbations::Vector{Perturbation}
    solver
    opt
    low::Vector{Float64}
    upp::Vector{Float64}
    level::Float64
end

npar(fop::FitOptions{T}) where {T<:FitKind} = npar(fop.nepochs, T())

function FitOptions(Ltot::Number;
    kind::FitKind = EpochsFit(),
    nepochs = 1,
    init = nothing,
    Ts = nothing,
    perturbations = Perturbation[],
    solver = LBFGS(),
    opt = Optim.Options(;iterations = 6000, allow_f_increases=true, time_limit = 60, g_tol = 5e-8),
    Tlow = 10, Tupp = 1e7,
    Nlow = 10, Nupp = 1e8,
    level = 0.95
)
    N = npar(nepochs, kind)
    if isnothing(init)
        init = zeros(N)
    else
        @assert length(init) == N
    end
    if isnothing(Ts)
        Ts = zeros(N-2)
    else
        @assert length(Ts) == N-2
    end
    @assert length(perturbations) <= N
    upp = zeros(N)
    low = zeros(N)
    if kind isa EpochsFit
        upp[2:2:end] .= Nupp
        low[2:2:end] .= Nlow
        upp[3:2:end-1] .= Tupp
        low[3:2:end-1] .= Tlow
    elseif kind isa SizesFit
        upp[2:end] .= Nupp
        low[2:end] .= Nlow
    end
    upp[1] = Ltot * 1.001
    low[1] = Ltot * 0.5

    return FitOptions{typeof(kind)}(
        nepochs,
        Ltot,
        init,
        Ts,
        perturbations,
        solver,
        opt,
        low,
        upp,
        level
    )
end

function setinit!(fop::FitOptions{T}, h::Histogram, mu::Float64) where {T<:FitKind}
    N = 1/(4*mu*(fop.Ltot/sum(h.weights))) # can be rough estimate depending on binning
    fop.init[1] = fop.Ltot
    n = npar(fop.nepochs, T())
    fop.init[2:end] .= N .* (0.99 .+ rand(n-1) .* 0.02)
    return nothing
end

function setinit!(fop::FitOptions{T}, init::Vector{Float64}) where {T<:FitKind}
    fop.init .= init
    return nothing
end