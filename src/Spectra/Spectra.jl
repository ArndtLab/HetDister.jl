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
	mldsmcp(rs, edges, mu, rho, TN; order = 10, mpanel = 0, mtail = 0, method = :fused, npicard = 0)

Compute the expected number of segments at representative lengths `rs`
that are midpoints of log bins defined by `edges`,
given the mutation rate `mu`, recombination rate `rho`, and
population size history `TN`.

The time integration runs on a `TimeGrid(length(TN) ÷ 2)` (panels pinned to
the epoch boundaries), with `mpanel` Gauss-Legendre nodes per epoch panel and
`mtail` Gauss-Legendre nodes under the algebraic map on the final
semi-infinite panel. When either is zero, the `TimeGrid` default is used.

With `method = :fused` (the default) a single forward sweep in `r` resolves all
orders of the SMC' recursion, using `npicard` transition applies per bin
(`npicard = 0` selects it with `getnpicard(mu, rho)`). With `method = :order`
the Neumann series is truncated at `order` intermediate recombination events
plus one, which is slower but produces the per-order `bag.res` columns.
`order` only affects `method = :order`; it is ignored by the fused path.
"""
function mldsmcp(rs, edges, mu, rho, TN; order = 10, mpanel = 0, mtail = 0,
	method::Symbol = :fused, npicard::Int = 0
)
	K = length(TN) ÷ 2
	g = iszero(mpanel) && iszero(mtail) ? TimeGrid(K) :
	    TimeGrid(K; m = iszero(mpanel) ? TimeGrid(1).m : mpanel,
	                mtail = iszero(mtail) ? TimeGrid(1).mtail : mtail)
	bag = IntegralArrays(order, g, length(rs), Val{length(TN)})
	mldsmcp!(bag, 1:order, rs, edges, mu, rho, TN; method, npicard)
	return get_tmp(bag.ys, eltype(TN))
end

"""
	mldsmcp!(bag, range, rs, edges, mu, rho, TN; method = :fused, npicard = 0)

In-place `mldsmcp`, writing `bag.ys`. `range` selects which orders are summed
and is ignored by the default `method = :fused`, which always resolves all of
them; it applies only to `method = :order`. On the fused path `bag.res` is
filled with `NaN`, since per-order diagnostics are not produced.
"""
function mldsmcp!(bag::IntegralArrays, range::AbstractRange{<:Int},
    rs::AbstractVector{<:Real}, edges::AbstractVector{<:Real}, mu::Real, rho::Real,
    TN::AbstractVector{<:Real}; method::Symbol = :fused, npicard::Int = 0
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
