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
	mldsmcp, mldsmcp!, IntegralArrays, getnpicard,
	extbps,
    lineages, cumulative_lineages, crediblehistory,
    sampleN, quantilesN

"""
	mldsmcp(rs, edges, mu, rho, TN; order = 10, ndt = 800, method = :fused, npicard = 0)

Compute the expected number of segments at representative lengths `rs`
that are midpoints of log bins defined by `edges`,
given the mutation rate `mu`, recombination rate `rho`, and
population size history `TN`.

`ndt` is the number of Legendre nodes for the time integration; the rule of
thumb is `ndt == length(rs)`, and `800` is where the discretisation error drops
below Poisson noise for a whole genome.

With `method = :fused` (the default) a single forward sweep in `r` resolves all
orders of the SMC' recursion, using `npicard` transition applies per bin
(`npicard = 0` selects it with `getnpicard(mu, rho)`). With `method = :order`
the Neumann series is truncated at `order` intermediate recombination events
plus one, which is slower but produces the per-order `bag.res` columns.
`order` only affects `method = :order`; it is ignored by the fused path.
"""
function mldsmcp(rs, edges, mu, rho, TN; order = 10, ndt = 800,
	method::Symbol = :fused, npicard::Int = 0
)
	bag = IntegralArrays(order, ndt, length(rs), Val{length(TN)})
	mldsmcp!(bag, 1:order, rs, edges, mu, rho, TN; method, npicard)
	return get_tmp(bag.ys, eltype(TN))
end

"""
	mldsmcp!(bag, range, rs, edges, mu, rho, TN; method = :order, npicard = 0)

In-place `mldsmcp`, writing `bag.ys`. `method` defaults to `:order` here so
that existing callers keep the order loop; `range` selects which orders are
summed and is ignored when `method = :fused`, which always resolves all of
them. On the fused path `bag.res` is filled with `NaN`, since per-order
diagnostics are not produced.
"""
function mldsmcp!(bag::IntegralArrays, range::AbstractRange{<:Int},
    rs::AbstractVector{<:Real}, edges::AbstractVector{<:Real}, mu::Real, rho::Real,
    TN::AbstractVector{<:Real}; method::Symbol = :order, npicard::Int = 0
)
	if method === :fused
		fusedsweep!(bag, rs, edges, mu, rho, TN; npicard)
		fill!(get_tmp(bag.res, eltype(TN)), NaN)
	elseif method === :order
		prordn!(bag, rs, edges, mu+rho, TN)
		mldsmcp!(bag, range, mu, rho, TN)
	else
		throw(ArgumentError("method must be :fused or :order, got :$method"))
	end
	return nothing
end

function mldsmcp!(bag::IntegralArrays, range::AbstractRange{<:Int}, 
	mu::Real, rho::Real, TN::AbstractVector{<:Real}
)
	mldsmcp!(get_tmp(bag.ys, eltype(TN)), get_tmp(bag.res, eltype(TN)), range, mu, rho, TN)
	return nothing
end

function mldsmcp!(m::AbstractVector{<:Real}, res::AbstractMatrix{<:Real},
	range::AbstractRange{<:Int}, mu::Real, rho::Real, TN::AbstractVector{<:Real}
)
    m .= 0
    for i in range
        m .= m .+ view(res,:,i) .* (2 * mu * TN[1] * (rho/(mu+rho))^(i-1) * (mu/(mu+rho)))
    end
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
