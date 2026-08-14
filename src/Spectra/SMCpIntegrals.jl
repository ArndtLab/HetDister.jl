module SMCpIntegrals

using FastGaussQuadrature
using LinearAlgebra
using Base.Threads
using PreallocationTools

using ..CoalescentBase

export IntegralArrays, prordn!,
    firstorder, firstorderint


function firstorder(r::Real, rate::Real, TN::AbstractVector{<:Real})
    s = 0.
    cum = 0.
    pnt = 1
    while pnt < length(TN)÷2
        pnt += 1
        t = getts(TN, pnt)
        aem = 1/2getns(TN, pnt-1)
        aep = 1/2getns(TN, pnt)
        cum += (t - getts(TN, pnt-1)) / 2getns(TN, pnt-1)
        s += (
            t^2*(aep/(aep+2rate*r) - aem/(aem+2rate*r)) 
            + 2t*(aep/(aep+2rate*r)^2 - aem/(aem+2rate*r)^2) 
            + 2*(aep/(aep+2rate*r)^3 - aem/(aem+2rate*r)^3)
        ) * exp(-2rate * r * t - cum)
    end
    s += 8 * getns(TN, 1)^2 / (1 + 4*getns(TN, 1) * rate * r)^3
    return s * 2 * rate
end

function firstorderint(r::Real, rate::Real, TN::AbstractVector{<:Real})
    s = 0.
    cum = 0.
    pnt = 1
    while pnt < length(TN)÷2
        pnt += 1
        t = getts(TN, pnt)
        aem = 1/2getns(TN, pnt-1)
        aep = 1/2getns(TN, pnt)
        cum += (t - getts(TN, pnt-1)) / 2getns(TN, pnt-1)
        s += ( 
            t*(aep/(aep+2rate*r) - aem/(aem+2rate*r)) 
            + (aep/(aep+2rate*r)^2 - aem/(aem+2rate*r)^2)
        ) * exp(-2rate * r * t - cum)
    end
    s += 2 * getns(TN, 1) / (1 + 4*getns(TN, 1) * rate * r)^2
    return - s
end

function pt(t::Real, TN::AbstractVector{<:Real})
    return exp(-cumcr(0, t, TN)/2) * t / (2 * Nt(t, TN)) # / <t> simplifies when multiplying by number of segments
end

function margrecomb(t::Real, TN::AbstractVector{<:Real})
    s = 0.
    pnt = 1
    while pnt < length(TN)÷2 && getts(TN, pnt+1) < t
        s += (getns(TN, pnt) - getns(TN, pnt+1)) * exp(-cumcr(getts(TN, pnt+1), t, TN))
        pnt += 1
    end
    return s
end

function ptt(ti::Real, tj::Real, TN::AbstractVector{<:Real}) # ti given tj
    if ti == tj
        return ti - margrecomb(ti, TN) - Nt(ti, TN) + Nt(0, TN) * exp(-cumcr(0, ti, TN)) #/ 2tj
    elseif ti < tj
        return 1 + margrecomb(ti, TN)/Nt(ti, TN) - Nt(0, TN) / Nt(ti, TN) * exp(-cumcr(0, ti, TN)) #/ 2tj
    else
        return exp(-cumcr(tj, ti, TN)/2) * (Nt(tj, TN) + margrecomb(tj, TN) - Nt(0, TN) * exp(-cumcr(0, tj, TN))) / Nt(ti, TN) #/ 2tj
    end
end

# The transition kernel ptt is diagonal + rank-one lower + rank-one upper.
# With G(t) = N(t) + R(t) - N(0)exp(-C(t)):
#   ptt(t|t') = G(t)/N(t)                                for t < t'
#             = t - G(t)                                 for t = t'
#             = [exp(-C(t)/2)/N(t)]*[exp(C(t')/2)G(t')]  for t > t'
# Clamping ptt to zero is equivalent to clamping G, because exp(-C/2)/N > 0.
# Note dgn uses the UNCLAMPED G; Phi and Gc use the clamped one.
# ts must be ascending.
function sepkernel!(Phi::AbstractVector{<:Real}, dgn::AbstractVector{<:Real},
    Gc::AbstractVector{<:Real}, Ninv::AbstractVector{<:Real}, dC::AbstractVector{<:Real},
    ts::AbstractVector{<:Real}, TN::AbstractVector{<:Real}
)
    n = length(ts)
    n0 = Nt(0, TN)
    cprev = zero(eltype(Phi))
    @inbounds for j in 1:n
        t = ts[j]
        c = cumcr(0, t, TN)
        nt = Nt(t, TN)
        g = nt + margrecomb(t, TN) - n0 * exp(-c)
        Gc[j] = max(g, zero(g))
        Phi[j] = Gc[j] / nt
        dgn[j] = max(t - g, zero(g))
        Ninv[j] = 1 / nt
        j > 1 && (dC[j-1] = c - cprev)
        cprev = c
    end
    dC[n] = zero(eltype(dC))
    return nothing
end

# temp[i,:] = M * jprt[:,i], with M the semiseparable transition operator.
# One backward pass for the upper triangle, one forward pass for the
# rescaled lower-triangle prefix sum plus the diagonal atom.
function transition!(temp::AbstractMatrix{<:Real}, jprt::AbstractMatrix{<:Real}, i::Int,
    Phi::AbstractVector{<:Real}, dgn::AbstractVector{<:Real}, Gc::AbstractVector{<:Real},
    Ninv::AbstractVector{<:Real}, dC::AbstractVector{<:Real}, om::AbstractVector{<:Real},
    ndt::Int
)
    T = eltype(temp)
    sfx = zero(T)
    @inbounds for j in ndt:-1:1
        temp[i,j] = Phi[j] * sfx
        sfx += jprt[j,i] * om[j]
    end
    st = zero(T)
    @inbounds for j in 1:ndt
        temp[i,j] += st * Ninv[j] + dgn[j] * jprt[j,i]
        st = exp(-dC[j]/2) * (st + Gc[j] * jprt[j,i] * om[j])
    end
    return nothing
end

function tolaguerre(z, TN::AbstractVector{<:Real})
    epoch = 1
    ce = 0
    ae = 1/2getns(TN, epoch)
    t = (z - ce)/ae
    while epoch < length(TN)÷2 && t > getts(TN, epoch+1)
        epoch += 1
        ce += (getts(TN, epoch) - getts(TN, epoch-1)) * ae
        ae = 1/2getns(TN, epoch)
        t = (z - ce + ae*getts(TN, epoch))/ae
    end
    return t, 1/ae
end

function tolegendre(z, TN::AbstractVector{<:Real})
    y = -1 - 2/(z-1)
    dy = 2/(z-1)^2
    t, dt = tolaguerre(y, TN)
    return t, dt * dy
end

struct IntegralArrays{T}
    order::Int
    n_dt::Int
    nrs::Int
    ys::DiffCache{Vector{T},Vector{T}}
    res::DiffCache{Matrix{T},Vector{T}}
    jprt::DiffCache{Matrix{T},Vector{T}}
    temp::DiffCache{Matrix{T},Vector{T}}
    zs::Vector{Float64}
    wt::Vector{Float64}
    ts::DiffCache{Vector{T},Vector{T}}
    dts::DiffCache{Vector{T},Vector{T}}
    qs::DiffCache{Vector{T},Vector{T}}
    om::DiffCache{Vector{T},Vector{T}}
    Phi::DiffCache{Vector{T},Vector{T}}
    dgn::DiffCache{Vector{T},Vector{T}}
    Gc::DiffCache{Vector{T},Vector{T}}
    Ninv::DiffCache{Vector{T},Vector{T}}
    dC::DiffCache{Vector{T},Vector{T}}
end

function IntegralArrays(order::Int, ndt::Int, nrs::Int, chunk, levels = 1)
    t, w = gausslegendre(ndt)
    dcvec() = DiffCache(zeros(Float64, ndt), chunk; levels)
    IntegralArrays(
        order, ndt, nrs,
        DiffCache(zeros(Float64, nrs), chunk; levels),
        DiffCache(zeros(Float64, nrs, order), chunk; levels),
        DiffCache(zeros(Float64, ndt, nrs), chunk; levels),
        DiffCache(zeros(Float64, nrs, ndt), chunk; levels),
        t,
        w,
        dcvec(), dcvec(), dcvec(), dcvec(), dcvec(),
        dcvec(), dcvec(), dcvec(), dcvec()
    )
end

function prordn!(bag::IntegralArrays,
    rs::AbstractVector{<:Real}, edges::AbstractVector{<:Real}, rate::Real,
    TN::AbstractVector{<:Real}
)
    T = eltype(TN)
    prordn!(
        get_tmp(bag.res, T),
        get_tmp(bag.jprt, T),
        get_tmp(bag.temp, T),
        bag.zs,
        bag.wt,
        get_tmp(bag.ts, T),
        get_tmp(bag.dts, T),
        get_tmp(bag.qs, T),
        get_tmp(bag.om, T),
        get_tmp(bag.Phi, T),
        get_tmp(bag.dgn, T),
        get_tmp(bag.Gc, T),
        get_tmp(bag.Ninv, T),
        get_tmp(bag.dC, T),
        rs, edges, rate, bag.order, bag.n_dt, bag.nrs, TN
    )
    return nothing
end

function prordn!(res::AbstractMatrix{<:Real}, jprt::AbstractMatrix{<:Real},
    temp::AbstractMatrix{<:Real},
    zs::AbstractVector{<:Real}, wt::AbstractVector{<:Real},
    ts::AbstractVector{<:Real}, dts::AbstractVector{<:Real}, qs::AbstractVector{<:Real},
    om::AbstractVector{<:Real}, Phi::AbstractVector{<:Real}, dgn::AbstractVector{<:Real},
    Gc::AbstractVector{<:Real}, Ninv::AbstractVector{<:Real}, dC::AbstractVector{<:Real},
    rs::AbstractVector{<:Real}, edges::AbstractVector{<:Real}, rate::Real,
    order::Int, n_dt::Int, nrs::Int,
    TN::AbstractVector{<:Real}
)
    @assert length(rs) == nrs

    res .= 0
    jprt .= 0
    temp .= 0

    @threads for j in 1:n_dt
        t, dt = tolegendre(zs[j], TN)
        ts[j] = t
        dts[j] = dt
        qs[j] = pt(t, TN)
        om[j] = wt[j] * dt
    end
    sepkernel!(Phi, dgn, Gc, Ninv, dC, ts, TN)
    @threads for i in 1:nrs
        @inbounds for j in 1:n_dt
            p = rate * exp(-2rate * rs[i] * ts[j])
            jprt[j,i] = p * qs[j]
        end
        res[i,1] = firstorder(rs[i], rate, TN)
    end
    for o in 1:order-1
        # transition t integral: the kernel is semiseparable, so each column
        # costs O(n_dt) instead of O(n_dt^2)
        @threads for i in 1:nrs
            transition!(temp, jprt, i, Phi, dgn, Gc, Ninv, dC, om, n_dt)
        end # I am modifying jprt in the end, so need to finish all temp first
        # the inner loop is variable in r, more efficient to multithread 
        # the time loop and separate the terminal t integral below (only
        # additional linear cost when single threaded)
        @threads for j in 1:n_dt
            # convolution r integral
            # the contribution of all previous (completed) bins k < i decays
            # multiplicatively with r, so instead of recomputing the full
            # sum over k = 1:i-1 at every i (O(nrs^2)), we carry it forward
            # in `acc`, rolling it from edges[i] to edges[i+1] at each step
            # (O(nrs)). `acc` holds the sum evaluated at the left edge of
            # bin i; it is then shifted to rs[i] to get jprt[j,i].
            acc = 0.
            @inbounds for i in 1:nrs
                w = edges[i+1] - edges[i]
                s = acc * exp(-2rate * (rs[i] - edges[i]) * ts[j])
                if w <= 1
                    s += temp[i,j] * (- expm1(-2rate * w * ts[j])) / 2ts[j]
                else
                    wi = rs[i] - edges[i]
                    s += temp[i,j] * (- expm1(-2rate * wi * ts[j])) / 2ts[j]
                end
                jprt[j,i] = s
                frac = (- expm1(-2rate * w * ts[j])) / 2ts[j]
                acc = exp(-2rate * w * ts[j]) * acc + temp[i,j] * frac
            end
        end
        @threads for i in 1:nrs
            s2 = 0.
            @inbounds for j in 1:n_dt
                # terminal t integral part
                # 2t factor from p(r|t) here does not simplify
                s2 += jprt[j,i] * 2 * ts[j] * om[j]
            end
            res[i,o+1] = s2
        end
    end
    return nothing
end

end