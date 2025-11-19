"""
    struct FitResult

A data structure to store the results of a fit.

See the introduction for how the model is 
parameterized [Data form of input and output](@ref).
Some methods are defined for this type to get the vector of parameters, std errors, 
model evidence, etc. See [`get_para`](@ref), [`sds`](@ref), [`evd`](@ref), 
[`pop_sizes`](@ref), [`durations`](@ref).
"""
struct FitResult
    nepochs::Int
    bin::Int
    mu::Float64
    rho::Float64
    para::Vector
    stderrors::Vector
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

npar(fit::FitResult) = 2fit.nepochs

mutable struct Deltas
    factors::Vector{Float64}
    state::Integer
end

function next!(d::Deltas)
    # assumes to be called from iteration over factors
    d.state += 1
    if d.state > length(d.factors)
        d.state = 1
    end
end

struct LBound <: AbstractVector{Float64}
    Ltot::Float64
    Nlow::Float64
    Tlow::Float64
    pars::Int
end
LBound(Ltot::Number,Nlow::Number,Tlow::Number,pars::Int) = LBound(
    Float64(Ltot), 
    Float64(Nlow), 
    Float64(Tlow),
    pars
)

Base.size(lb::LBound) = (lb.pars,)

function Base.getindex(lb::LBound, i::Int)
    if i == 1
        return lb.Ltot * 0.5
    elseif i%2 == 0
        return lb.Nlow
    else
        return lb.Tlow
    end
end

struct UBound <: AbstractVector{Float64}
    Ltot::Float64
    Nupp::Float64
    Tupp::Float64
    pars::Int
end
UBound(Ltot::Number,Nupp::Number,Tupp::Number,pars::Int) = UBound(
    Float64(Ltot), 
    Float64(Nupp), 
    Float64(Tupp),
    pars
)

Base.size(ub::UBound) = (ub.pars,)

function Base.getindex(ub::UBound, i::Int)
    if i == 1
        return ub.Ltot * 1.001
    elseif i%2 == 0
        return ub.Nupp
    else
        return ub.Tupp
    end
end

mutable struct FitOptions
    nepochs::Int
    mu::Float64
    rho::Float64
    Ltot::Real
    init::Vector{Float64}
    perturb::BitVector
    delta::Deltas
    solver
    opt
    low::LBound
    upp::UBound
    prior::Vector{<:Distribution}
    level::Float64
    smallest_segment::Int
    force::Bool
    maxnts::Int
    naive::Bool
    order::Int
    ndt::Int
end

function Base.show(io::IO, fop::FitOptions)
    println(io, "FitOptions with:")
    println(io, "total genome length: ", fop.Ltot)
    println(io, "μ / bp / g: ", fop.mu)
    println(io, "ρ / bp / g: ", fop.rho)
    println(io, "N lower bound: ", fop.low.Nlow)
    println(io, "N upper bound: ", fop.upp.Nupp)
    println(io, "T lower bound: ", fop.low.Tlow)
    println(io, "T upper bound: ", fop.upp.Tupp)
    println(io, "solver: ", summary(fop.solver))
end

npar(fop::FitOptions) = 2fop.nepochs

"""
    FitOptions(Ltot, mu, rho; kwargs...)

Construct an an object of type FitOptions, requiring 
total genome length `Ltot` in base pairs,
mutation rate and recombination rate per base pair per generation.

## Optional Arguments
- `Tlow::Number=10`, `Tupp::Number=1e7`: The lower and upper bounds for the duration of epochs.
- `Nlow::Number=10`, `Nupp::Number=1e8`: The lower and upper bounds for the population sizes.
- `level::Float64=0.95`: The confidence level for the confidence intervals on the parameters estimates.
- `solver`: The solver to use for the optimization, default is `LBFGS()`.
- `smallest_segment::Int=1`: The smallest segment size present in the histogram to consider 
  for the signal search.
- `force::Bool=true`: if true try to fit further epochs even when no signal is found.
- `maxnts::Int=10`: The maximum number of new time splits to consider when adding a new epoch.
  Higher is greedier.
- `naive::Bool=true`: if true the expected weights are computed
  using the closed form integral, otherwise using higher order transition
  probabilities from SMC' theory (slower).
- `order::Int=10`: maximum number of higher order corrections to use
  when `naive` is false, i.e. number of intermediate recombination events
  plus one.
- `ndt::Int=800`: number of Legendre nodes to use when `naive` is false.

## Optim Arguments
Additional keywords are passed to the `Optim.Options` constructor 
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl). Default are:
- `iterations = 6000`
- `time_limit = 60`
- `g_tol = 5e-8`
- `show_warnings = false`
"""
function FitOptions(Ltot, mu, rho;
    Tlow = 10, Tupp = 1e7,
    Nlow = 10, Nupp = 1e8,
    level = 0.95,
    solver = LBFGS(),
    nepochs::Int = 1,
    smallest_segment::Int = 1,
    force::Bool = true,
    maxnts::Int = 15,
    naive::Bool = true,
    order = 10,
    ndt = 800,
    iterations = 6000,
    time_limit = 60,
    g_tol = 5e-8,
    show_warnings = false,
    kwargs...
)
    N = 2nepochs
    init = zeros(N)
    # set bounds and prior for the parameters
    upp = UBound(Ltot,Nupp,Tupp,N)
    low = LBound(Ltot,Nlow,Tlow,N)
    prior = Uniform.(low,upp)
    perturb = falses(N)
    factors = mapreduce( i->fill(i, 10), vcat, [0.001, 0.01, 0.1, 0.5, 0.5, 0.9, 2] )
    delta = Deltas(factors, 0)

    return FitOptions(
        nepochs,
        mu,
        rho,
        Ltot,
        init,
        perturb,
        delta,
        solver,
        Optim.Options(;iterations, time_limit, g_tol, show_warnings, kwargs...),
        low,
        upp,
        prior,
        level,
        smallest_segment,
        force,
        maxnts,
        naive,
        order,
        ndt
    )
end

function setinit!(fop::FitOptions, weights::Vector{<:Integer})
    vol = sum(weights)
    @assert vol != 0 "Empty histogram!"
    N = 1/(4*fop.mu*(fop.Ltot/vol)) # can be rough estimate depending on binning
    n = npar(fop)
    fop.init[1] = fop.Ltot
    fop.init[2:end] .= N .* (0.99 .+ rand(n-1) .* 0.02)
    setinit!(fop, fop.init)
    return nothing
end

function setinit!(fop::FitOptions, init::AbstractVector{Float64})
    @assert length(init) == npar(fop)
    fop.init .= init
    for i in eachindex(fop.init)
        fop.init[i] < fop.low[i] ? fop.init[i] = fop.low[i] * 1.001 : nothing
        fop.init[i] > fop.upp[i] ? fop.init[i] = fop.upp[i] * 0.999 : nothing
    end
    return nothing
end

function setnepochs!(fop::FitOptions, nepochs::Int)
    N = 2nepochs
    fop.nepochs = nepochs
    fop.init = zeros(N)
    fop.perturb = falses(N)
    L = fop.Ltot
    Nlow = fop.low.Nlow
    Nupp = fop.upp.Nupp
    Tlow = fop.low.Tlow
    Tupp = fop.upp.Tupp
    fop.low = LBound(L, Nlow, Tlow, N)
    fop.upp = UBound(L, Nupp, Tupp, N)
    fop.prior = Uniform.(fop.low, fop.upp)
end

function set_perturb!(fop::FitOptions, fit::FitResult)
    @assert npar(fop) == npar(fit)
    for i in eachindex(fop.perturb)
        fop.perturb[i] = fit.opt.at_lboundary[i] || 
            (fit.opt.at_uboundary[i] && i > 1) ||
            isinf(evd(fit)) ||
            !fit.converged
    end
end

function reset_perturb!(fop::FitOptions)
    fop.perturb .= falses(npar(fop))
    fop.delta.state = 0
end

struct PInit <: AbstractVector{Float64}
    fop::FitOptions
end

Base.size(p::PInit) = (npar(p.fop),)

getdelta(fop::FitOptions) = fop.delta.factors[fop.delta.state]

function Base.getindex(p::PInit, i::Int)
    if !p.fop.perturb[i]
        return p.fop.init[i]
    else
        dl = getdelta(p.fop)
        low = p.fop.low[i]
        upp = p.fop.upp[i]
        if dl < 1
            return rand(
                truncated(
                    LogNormal(log(p.fop.init[i]), dl),
                    low,
                    upp
                )
            )
        else
            return rand(Uniform(low, upp))
        end
    end
end

function isnaive(fop::FitOptions)
    return fop.naive
end

function setnaive!(fop::FitOptions, flag::Bool)
    fop.naive = flag
end

function setOptimOptions!(fop::FitOptions;
    iterations = 6000,
    time_limit = 60,
    g_tol = 5e-8,
    show_warnings = false,
    kwargs...
)
    fop.opt = Optim.Options(; iterations, time_limit, g_tol, show_warnings, kwargs...)
end