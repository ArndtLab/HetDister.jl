module Spectra

using LinearAlgebra
using ForwardDiff
using PreallocationTools

include("CoalescentBase.jl")
using .CoalescentBase

include("SMCpIntegrals.jl")
using .SMCpIntegrals

export 
    hid, hid_integral, firstorder, firstorderint, 
	laplacekingman, laplacekingmanint, 
	mldsmcp, mldsmcp!, IntegralArrays

# Computing

function secondderivative(f, x)
    dfdx = x -> ForwardDiff.derivative(f, x)
    ForwardDiff.derivative(dfdx, x)
end

function hid(TN::Vector, mu::Float64, r::Number)
	# TN = [L, N0, T1, N1, T2, N2, ...]
	L = TN[1]
	N = TN[end]
	(2*mu^2*L)/(N^2) * secondderivative(s -> laplace_n(TN, s), 2*mu*r)
	# prefactor        pure bliss
end

function hid(L::Number, N::Vector, T::Vector, mu::Float64, r::Number)
	# T = [T1, T2, ...]
	# N = [Nstat, N1, N2, ...]
	Nend = (length(N) == 0 ? Nstat : N[end])
	(2*mu^2*L)/(Nend^2) * secondderivative(s -> laplace_n(N, T, s), 2*mu*r)
	# prefactor           pure bliss
end

function hid_integral(TN::Vector, mu::Float64, r::Number)
    # integral of hid
	# TN = [L, N0, T1, N1, T2, N2, ...]
	L = TN[1]
	N = TN[end]
	(mu*L)/(N^2) * ForwardDiff.derivative(s -> laplace_n(TN, s), 2*mu*r)
	# prefactor    pure bliss
end

function hid_integral(Nv::Vector, Tv::Vector, L::Number, mu::Float64, r::Number)
    # integral of hid
	N = Nv[end]
	(mu*L)/(N^2) * ForwardDiff.derivative(s -> laplace_n(Nv, Tv, s), 2*mu*r)
	# prefactor    pure bliss
end

"""
	mldsmcp(rs, edges, mu, rho, TN; order = 10, ndt = 800)

Compute the expected number of segments with length in each bin defined by `edges`,
given the midpoints `rs`, mutation rate `mu`, recombination rate `rho`, and
population size history `TN`.

The computation uses the SMC' higher order transition probabilities
with `order` maximum number of intermediate recombination events plus one,
and `ndt` Legendre nodes for the numerical integration.
"""
function mldsmcp(rs, edges, mu, rho, TN; order = 10, ndt = 800)
	bag = IntegralArrays(order, ndt, length(rs), Val{length(TN)})
	mldsmcp!(bag, 1:order, rs, edges, mu, rho, TN)
	return get_tmp(bag.ys, eltype(TN))
end

function mldsmcp!(bag::IntegralArrays, range::AbstractRange{<:Int},
    rs::AbstractVector{<:Real}, edges::AbstractVector{<:Real}, mu::Real, rho::Real,
    TN::AbstractVector{<:Real}
)
    prordn!(bag, rs, edges, mu+rho, TN)
	mldsmcp!(bag, range, mu, rho, TN)
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
