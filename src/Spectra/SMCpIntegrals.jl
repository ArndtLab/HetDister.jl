module SMCpIntegrals

using FastGaussQuadrature
using LinearAlgebra
using Base.Threads

include("CoalescentBase.jl")
using .CoalescentBase

export IntegralArrays, prordn!, firstorder, firstorderint

function Nt(t::Real, TN::AbstractVector{<:Real})
    pnt = 1
    while pnt < length(TN)÷2 && getts(TN, pnt+1) <= t
        pnt += 1
    end
    return getns(TN, pnt)
end

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
    res::Matrix{T}
    jprt::Matrix{T}
    temp::Matrix{T}
    qtt::Matrix{T}
    zs::Vector{T}
    wt::Vector{T}
    ts::Vector{T}
    dts::Vector{T}
end

function IntegralArrays(order::Int, ndt::Int, nrs::Int)
    t, w = gausslegendre(ndt)
    IntegralArrays{Float64}(
        order, ndt, nrs,
        Array{Float64}(undef, nrs, order),
        Array{Float64}(undef, ndt, nrs),
        Array{Float64}(undef, nrs, ndt),
        Array{Float64}(undef, ndt, ndt),
        t,
        w,
        Array{Float64}(undef, ndt),
        Array{Float64}(undef, ndt)
    )
end

function prordn!(bag::IntegralArrays{<:Real}, 
    rs::Vector{<:Real}, edges::Vector{<:Real}, rate::Real, 
    TN::AbstractVector{<:Real}
)
    prordn!(bag.res, bag.jprt, bag.temp, bag.qtt, bag.zs, bag.wt, bag.ts, bag.dts, rs, edges, rate, bag.order, bag.n_dt, bag.nrs, TN)
    return nothing
end

function prordn!(res::AbstractMatrix{<:Real}, jprt::AbstractMatrix{<:Real}, temp::AbstractMatrix{<:Real}, qtt::AbstractMatrix{<:Real},
    zs::AbstractVector{<:Real}, wt::AbstractVector{<:Real}, ts::AbstractVector{<:Real}, dts::AbstractVector{<:Real},
    rs::Vector{<:Real}, edges::Vector{<:Real}, rate::Real, order::Int, n_dt::Int, nrs::Int,
    TN::AbstractVector{<:Real}
)
    @assert length(rs) == nrs

    res .= 0
    jprt .= 0
    temp .= 0

    @threads for i in 1:nrs
        for j in 1:n_dt
            t, dt = tolegendre(zs[j], TN)
            ts[j] = t
            dts[j] = dt
            q = pt(t, TN)
            p = rate * exp(-2rate * rs[i] * t)
            jprt[j,i] = p * q
        end
        res[i,1] = firstorder(rs[i], rate, TN)
    end
    @threads for i in 1:n_dt
        for j in 1:n_dt
            w = i == j ? 1. : wt[j] * dts[j]
            p = max(ptt(ts[i], ts[j], TN), 0.)
            qtt[j,i] = p * w
        end
    end
    for o in 1:order-1
        @threads for i in 1:nrs
            # transition t integral
            for j in 1:n_dt
                s = 0.
                for k in 1:n_dt
                    s += jprt[k,i] * qtt[k,j]
                end
                temp[i,j] = s
            end
        end # I am modifying jprt in the end, so need to finish all temp first
        @threads for i in 1:nrs
            s2 = 0.
            for j in 1:n_dt
                # convolution r integral
                s = 0.
                for k in 1:i-1
                    w = edges[k+1] - edges[k]
                    s += temp[k,j] * exp(-2rate * (rs[i]-edges[k+1]) * ts[j]) * (- expm1(-2rate * w * ts[j])) / 2ts[j]
                end
                w = edges[i+1] - edges[i]
                if w <= 1
                    s += temp[i,j] * (- expm1(-2rate * w * ts[j])) / 2ts[j]
                else
                    w = rs[i] - edges[i]
                    s += temp[i,j] * (- expm1(-2rate * w * ts[j])) / 2ts[j]
                end
                jprt[j,i] = s
                # terminal t integral part
                # 2t factor from p(r|t) here does not simplify
                s2 += jprt[j,i] * 2 * ts[j] * wt[j] * dts[j]
            end
            res[i,o+1] = s2
        end
    end
    return nothing
end

end