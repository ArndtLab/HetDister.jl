module Spectra

using LinearAlgebra
using PreallocationTools

include("CoalescentBase.jl")
using .CoalescentBase

include("SMCpIntegrals.jl")
using .SMCpIntegrals

export
    firstorder, firstorderint,
	laplacekingman, laplacekingmanint,
	mldsmcp, mldsmcp!, IntegralArrays, getnpicard, TimeGrid, timegrid, ndt, npanels,
	extbps,
    lineages, cumulative_lineages, crediblehistory,
    sampleN, quantilesN

"""
	mldsmcp(rs, edges, mu, rho, TN; msub = 0, nfin = 0, ntail = 0, npicard = 0)

Compute the expected number of segments at representative lengths `rs`
that are midpoints of log bins defined by `edges`,
given the mutation rate `mu`, recombination rate `rho`, and
population size history `TN`.

A single forward sweep in `r` over the Volterra form of the SMC' recursion
resolves **all** orders, using `npicard` transition applies per bin
(`npicard = 0` selects it with `getnpicard(mu, rho)`).

The time integration runs on a `timegrid(length(TN) ÷ 2; msub, nfin, ntail)`:
sub-panels pinned to the epoch boundaries, `msub` Gauss-Legendre nodes each,
`nfin` per finite epoch and `ntail` in the tail. Any of the three passed as zero
takes the `TimeGrid` default.
"""
function mldsmcp(rs, edges, mu, rho, TN;
	msub::Int = 0, nfin::Int = 0, ntail::Int = 0, npicard::Int = 0
)
	K = length(TN) ÷ 2
	bag = IntegralArrays(timegrid(K; msub, nfin, ntail), length(rs), Val{length(TN)})
	mldsmcp!(bag, rs, edges, mu, rho, TN; npicard)
	return get_tmp(bag.ys, eltype(TN))
end

"""
	mldsmcp!(bag, rs, edges, mu, rho, TN; npicard = 0)

In-place `mldsmcp`, writing `bag.ys`.
"""
function mldsmcp!(bag::IntegralArrays,
    rs::AbstractVector{<:Real}, edges::AbstractVector{<:Real}, mu::Real, rho::Real,
    TN::AbstractVector{<:Real}; npicard::Int = 0
)
	fusedsweep!(bag, rs, edges, mu, rho, TN; npicard)
	return nothing
end

"""
	laplacekingman(r, mu, TN)

Compute the approximate number of segments of length `r` 
using the Laplace transform of the Kingman coalescent at frequency `2mu r`,
given mutation rate `mu` and population size history `TN`.
"""
function laplacekingman(r::Real, mu::Real, TN::AbstractVector{<:Real})
    return firstorder(r, mu, TN) * 2 * mu * TN[1]
end

function laplacekingmanint(r::Real, mu::Real, TN::AbstractVector{<:Real})
    return firstorderint(r, mu, TN) * 2 * mu * TN[1]
end

end
