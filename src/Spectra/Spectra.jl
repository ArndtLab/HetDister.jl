module Spectra

using LinearAlgebra
using ForwardDiff

include("CoalescentBase.jl")
using .CoalescentBase

include("SMCpIntegrals.jl")
using .SMCpIntegrals

export 
    hid, hid_integral, firstorder, firstorderint, laplacekingman, mldsmcp, mldsmcp!, IntegralArrays

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

function mldsmcp!(ys::AbstractVector{<:Real}, range::AbstractRange{<:Int}, bag::IntegralArrays{<:Real},
    rs::Vector{<:Real}, edges::Vector{<:Real}, mu::Real, rho::Real,
    TN::AbstractVector{<:Real}
)
    prordn!(bag, rs, edges, mu+rho, TN)
    ys .= 0.
    for i in range
        ys .+= bag.res[:,i] * 2 * mu * TN[1] * (rho/(mu+rho))^(i-1) * (mu/(mu+rho))
    end
end

function laplacekingman(r::Real, mu::Real, TN::AbstractVector{<:Real})
    return firstorder(r, mu, TN) * 2 * mu * TN[1]
end

function laplacekingmanint(r::Real, mu::Real, TN::AbstractVector{<:Real})
    return firstorderint(r, mu, TN) * 2 * mu * TN[1]
end

end
