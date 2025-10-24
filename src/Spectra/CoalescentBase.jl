module CoalescentBase
using ForwardDiff

export getts, getns,
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
    s = zero(T)
    for j in 1:i
        s += TN[end-2*(j-1)]
    end
    return s
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
    pnt = 1
    c = 0.
    while (pnt < length(TN)÷2) && (getts(TN, pnt) < t)
        gens = getts(TN, pnt+1) >= t ? (t - getts(TN, pnt)) : (getts(TN, pnt+1) - getts(TN, pnt))
        N = getns(TN, pnt)
        c += gens / 2N
        pnt += 1
    end
    if getts(TN, pnt) < t
        gens = t - getts(TN, pnt)
        N = getns(TN, pnt)
        c += gens / 2N
        pnt += 1
    end
    return exp(-c) / (2 * getns(TN, pnt-1))
end

"""
    extbps(t::Number, TN::Vector)

Calculate the the expected number of basepairs that still have to
reach coalescence at time `t` generations in the past. 

The demographic scenario is encoded in `TN`.

### Reference
"""
function extbps(t::Number, TN::Vector)
    pnt = 1
    c = 0.
    while (pnt < length(TN)÷2) && (getts(TN, pnt) < t)
        gens = getts(TN, pnt+1) >= t ? (t - getts(TN, pnt)) : (getts(TN, pnt+1) - getts(TN, pnt))
        N = getns(TN, pnt)
        c += gens / 2N
        pnt += 1
    end
    if getts(TN, pnt) < t
        gens = t - getts(TN, pnt)
        N = getns(TN, pnt)
        c += gens / (2 * N)
        pnt += 1
    end
    return round(TN[1]*exp(-c))
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
function lineages(t::Number, rho::Number, TN::Vector; k::Number = 0)
    L = TN[1]
    pnt = 1
    c = 0.
    while (pnt < length(TN)÷2) && (getts(TN, pnt) < t)
        gens = getts(TN, pnt+1) >= t ? (t - getts(TN, pnt)) : (getts(TN, pnt+1) - getts(TN, pnt))
        N = getns(TN, pnt)
        c += gens / (2 * N)
        pnt += 1
    end
    if getts(TN, pnt) < t
        gens = t - getts(TN, pnt)
        N = getns(TN, pnt)
        c += gens / (2 * N)
        pnt += 1
    end
    return 2 * L * rho * t * exp(-2 * rho * t * k - c) / (2 * getns(TN, pnt-1))
end

function cumulative_lineages(t, TN::Vector, rho::Float64; k::Number = 0)
    N = TN[end]
    pnt = 1
    c = 0.
    while (pnt < length(TN)÷2) && (getts(TN, pnt) < t)
        gens = getts(TN, pnt+1) >= t ? (t - getts(TN, pnt)) : (getts(TN, pnt+1) - getts(TN, pnt))
        N = getns(TN, pnt)
        c += gens / (2 * N)
        pnt += 1
    end
    if getts(TN, pnt) < t
        gens = t - getts(TN, pnt)
        N = getns(TN, pnt)
        c += gens / (2 * N)
        pnt += 1
    end
    first_der = ForwardDiff.derivative(s -> laplace_n(TN,s), 2rho*k) / (2 * TN[end]^2)
    ep = 1
    while ep < length(TN)÷2 && getts(TN, ep+1) <= t
        ep += 1
    end
    if isnothing(ep)
        TNp = TN[1:2]
    else
        remn_ep = sum(TN[end-2ep-1:2:end-1]) - t
        TNp = TN[1:end-2ep+2]
        TNp[end-1] = remn_ep
    end
    first_der_p = ForwardDiff.derivative(s -> laplace_n(TNp,s), 2rho*k) / (2 * TNp[end]^2)
    lap_p = laplace_n(TNp, 2rho*k) / (2 * TNp[end]^2)
    return round(2TN[1] * rho * (exp(-c-2rho*k*t)*(first_der_p - t*lap_p) - first_der))
end

end