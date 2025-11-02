module CoalescentBase
using ForwardDiff

export getts, getns,
    Nt, cumcr,
    coalescent, extbps, 
    laplace_n, 
    lineages, cumulative_lineages

function getts(TN::AbstractVector{T}, i::Int) where T
    # TN = [L, N0, T1, N1, T2, N2, ...]
    # returns the ordered times in reverse order
    (i < 1 || i > length(TN) ÷ 2) && throw(ArgumentError("index out of bounds"))
    s = zero(T)
    for j in 2:i
        s += TN[end-1-2*(j-2)]
    end
    return s
end

function getns(TN::AbstractVector{T}, i::Int) where T
    # TN = [L, N0, T1, N1, T2, N2, ...]
    # returns the ordered population sizes in reverse order
    (i < 1 || i > length(TN) ÷ 2) && throw(ArgumentError("index out of bounds"))
    return TN[end-2*(i-1)]
end

function Nt(t::Real, TN::AbstractVector{<:Real})
    pnt = 1
    while pnt < length(TN)÷2 && getts(TN, pnt+1) <= t
        pnt += 1
    end
    return getns(TN, pnt)
end

# cumulative coalescence rate between t1 and t2
function cumcr(t1::Real, t2::Real, TN::AbstractVector{<:Real})
    @assert t2 >= t1
    @assert t1 >= 0
    pnt = 1
    while pnt < length(TN)÷2 && getts(TN, pnt+1) <= t1
        pnt += 1
    end
    c = 0.
    while pnt < length(TN)÷2 && getts(TN, pnt) < t2
        gens = min(t2, getts(TN, pnt+1)) - max(t1, getts(TN, pnt))
        c += gens / getns(TN, pnt)
        pnt += 1
    end
    if getts(TN, pnt) < t2
        gens = t2 - max(t1, getts(TN, pnt))
        c += gens / getns(TN, pnt)
        pnt += 1
    end
    return c
end

"""
    coalescent(t::Number, TN::Vector)

Calculate the probability of coalescence at time `t` generations in
the past.

It is computed for two alleles in the absence of recombinaiton 
and for a demographic scenario encoded in `TN`. The distribution 
of such `t`s is geometric as introduced by Hudson and Kingman.

### References
"""
function coalescent(t::Number, TN::Vector)
    return exp(-cumcr(0, t, TN)/2) / (2 * Nt(t, TN))
end

"""
    extbps(t::Number, TN::Vector)

Calculate the the expected number of basepairs that still have to
reach coalescence at time `t` generations in the past. 

The demographic scenario is encoded in `TN`.

### Reference
"""
function extbps(t::Number, TN::Vector)
    return round(TN[1]*exp(-cumcr(0, t, TN)/2))
end

function laplace_n(TN::Vector, s::Number)
    N = TN[2]
    y = 2 * N^2 / (1 + 2*N*s)
    # stationary solution in first epoch
    for k in 3:2:length(TN) # loop over further epochs
        T  = TN[k]
        Np = TN[k-1]
        N = TN[k+1]
        # step up or down
        gamma = N / Np
        w1 = gamma >= 1 ? gamma^2 - (gamma^2 - 1)/(2 * Np) : gamma^2
        w2 = gamma >= 1 ? (gamma^2 - 1) * Np               : zero(gamma)
        # propagate in time
        v1 = exp((-T/(2*N)) - s*T)
        v2 = (1 - v1) * ((2*N^2) / (1 + 2*N*s))
        # update value
        y = (w1 * y + w2) * v1 + v2
    end
    y
end

function laplace_n(Nv::Vector, Tv::Vector, s::Number)
	# T =        [T1, T2, ...]
	# N = [Nstat, N1, N2, ...]
    Nstat = Nv[1]
    y = 2 * Nstat^2 / (1 + 2*Nstat*s)
    # stationary solution in first epoch
    Np = Nstat
    for (T, N) in zip(Tv, Iterators.drop(Nv, 1)) # loop over further epochs
        # T  = Tv[k-2]
        # Np = Nv[k-1]
        # N =  Nv[k]
        # step up or down
        gamma = N / Np
        w1 = gamma >= 1 ? gamma^2 - (gamma^2 - 1)/(2 * Np) : gamma^2
        w2 = gamma >= 1 ? (gamma^2 - 1) * Np               : zero(gamma)
        # propagate in time
        v1 = exp((-T/(2*N)) - s*T)
        v2 = (1 - v1) * ((2*N^2) / (1 + 2*N*s))
        # update value
        y = (w1 * y + w2) * v1 + v2
        Np = N
    end
    y
end

"""
    lineages(t::Float64, TN::Vector, rho::Float64; k::Int = 0)

Calculate the expected number of genomic segments which are Identical by Descent 
and coalesce at time `t` generations in the past having a genomic length longer
than `k` basepairs. 

The demographic scenario is encoded in `TN` and the recombination rate is `rho`
in unit per bp per generation.
"""
function lineages(t::Number, TN::Vector, rho::Number; k::Number = 0)
    return 2 * TN[1] * rho * t * exp(-2 * rho * t * k - cumcr(0, t, TN)/2) / (2 * Nt(t, TN))
end

function cumulative_lineages(t::Number, TN::Vector, rho::Float64; k::Number = 0)
    s = 0.
    cum = 0.
    pnt = 1
    while pnt < length(TN)÷2 && getts(TN, pnt+1) < t
        pnt += 1
        t_ = getts(TN, pnt)
        aem = 1/2getns(TN, pnt-1)
        aep = 1/2getns(TN, pnt)
        cum += (t_ - getts(TN, pnt-1)) / 2getns(TN, pnt-1)
        s += ( 
            t_*(aep/(aep+2rho*k) - aem/(aem+2rho*k)) 
            + (aep/(aep+2rho*k)^2 - aem/(aem+2rho*k)^2)
        ) * exp(-2rho * k * t_ - cum)
    end
    ae = 1/2getns(TN, pnt)
    cum += (t - getts(TN, pnt)) / 2getns(TN, pnt)
    s -= ( 
        t*ae/(ae+2rho*k) + ae/(ae+2rho*k)^2
    ) * exp(-2rho * k * t - cum)
    s += 2 * getns(TN, 1) / (1 + 4*getns(TN, 1) * rho * k)^2
    return s * 2 * TN[1] * rho
end

end