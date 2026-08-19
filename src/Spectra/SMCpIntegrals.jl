module SMCpIntegrals

using FastGaussQuadrature
using LinearAlgebra
using Base.Threads
using PreallocationTools

using ..CoalescentBase

export IntegralArrays, prordn!, fusedsweep!, getnpicard,
    firstorder, firstorderint, TimeGrid, timenodes!, ndt


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

"""
    TimeGrid(K; m = 48, mtail = 48)

Reference nodes and weights for the panel-wise time quadrature of a `K`-epoch
history. Holds nothing that depends on `TN`: the panels are built from the epoch
times at evaluation time by [`timenodes!`](@ref).

Epochs `1 … K-1` each get an `m`-point Gauss-Legendre panel over `[T_k, T_{k+1}]`;
the final epoch `[T_K, ∞)` gets an `mtail`-point Gauss-Laguerre panel in the
coalescent variable `u = (t - T_K) / 2N_K`, where the coalescent factor is exactly
`exp(-C(T_K)/2) * exp(-u)`.

The Laguerre weights are stored **folded**, `w_i * exp(u_i)`, so they cancel the
`exp(-u)` that the integrand already carries through `pt`. They stay well
conditioned because the growth of `exp(u_i)` is offset by the decay of `w_i`.
"""
struct TimeGrid
    m::Int
    mtail::Int
    K::Int
    zleg::Vector{Float64}
    wleg::Vector{Float64}
    ulag::Vector{Float64}
    wlag::Vector{Float64}
end

function TimeGrid(K::Int; m::Int = 48, mtail::Int = 48)
    @assert K >= 1 "need at least one epoch"
    @assert m >= 2 "need at least 2 nodes per panel"
    @assert mtail >= 2 "need at least 2 nodes in the tail panel"
    z, w = gausslegendre(m)
    u, wl = gausslaguerre(mtail)
    return TimeGrid(m, mtail, K, z, w, u, wl .* exp.(u))
end

"""
    ndt(g::TimeGrid)

Total number of quadrature nodes: `(K-1)` finite panels of `m` plus the `mtail`
tail nodes.
"""
ndt(g::TimeGrid) = (g.K - 1) * g.m + g.mtail

"""
    timenodes!(ts, om, g::TimeGrid, TN)

Fill `ts` with the quadrature nodes and `om` with their weights for the history
`TN`. Panels are pinned to the epoch boundaries, so each node is an **affine**
function of the epoch parameters: nodes move smoothly with `TN` and a node can
never migrate from one epoch to another. `ts` comes out ascending, as
`sepkernel!` and `transition!` require.

The tail weights carry the folded `exp(u_i)` from [`TimeGrid`](@ref); it cancels
against the `exp(-C(t)/2)` the integrand supplies through `pt`.
"""
function timenodes!(ts::AbstractVector{<:Real}, om::AbstractVector{<:Real},
    g::TimeGrid, TN::AbstractVector{<:Real}
)
    K = length(TN) ÷ 2
    @assert K == g.K "grid built for $(g.K) epochs, got $K"
    @assert length(ts) == ndt(g) "ts has length $(length(ts)), expected $(ndt(g))"
    @assert length(om) == ndt(g) "om has length $(length(om)), expected $(ndt(g))"

    j = 0
    @inbounds for k in 1:K-1
        a = getts(TN, k)
        b = getts(TN, k + 1)
        c = (a + b) / 2
        h = (b - a) / 2
        for i in 1:g.m
            j += 1
            ts[j] = c + h * g.zleg[i]
            om[j] = g.wleg[i] * h
        end
    end
    TK = getts(TN, K)
    twoNK = 2 * getns(TN, K)
    @inbounds for i in 1:g.mtail
        j += 1
        ts[j] = TK + twoNK * g.ulag[i]
        om[j] = g.wlag[i] * twoNK
    end
    return nothing
end

struct IntegralArrays{T}
    order::Int
    n_dt::Int
    nrs::Int
    ys::DiffCache{Vector{T},Vector{T}}
    res::DiffCache{Matrix{T},Vector{T}}
    jprt::DiffCache{Matrix{T},Vector{T}}
    temp::DiffCache{Matrix{T},Vector{T}}
    grid::TimeGrid
    ts::DiffCache{Vector{T},Vector{T}}
    qs::DiffCache{Vector{T},Vector{T}}
    om::DiffCache{Vector{T},Vector{T}}
    Phi::DiffCache{Vector{T},Vector{T}}
    dgn::DiffCache{Vector{T},Vector{T}}
    Gc::DiffCache{Vector{T},Vector{T}}
    Ninv::DiffCache{Vector{T},Vector{T}}
    dC::DiffCache{Vector{T},Vector{T}}
    A::DiffCache{Vector{T},Vector{T}}
    Jf::DiffCache{Vector{T},Vector{T}}
    MJ::DiffCache{Vector{T},Vector{T}}
    J1::DiffCache{Vector{T},Vector{T}}
end

function IntegralArrays(order::Int, grid::TimeGrid, nrs::Int, chunk, levels = 1)
    n = ndt(grid)
    dcvec() = DiffCache(zeros(Float64, n), chunk; levels)
    IntegralArrays(
        order, n, nrs,
        DiffCache(zeros(Float64, nrs), chunk; levels),
        DiffCache(zeros(Float64, nrs, order), chunk; levels),
        DiffCache(zeros(Float64, n, nrs), chunk; levels),
        DiffCache(zeros(Float64, nrs, n), chunk; levels),
        grid,
        dcvec(), dcvec(), dcvec(), dcvec(),
        dcvec(), dcvec(), dcvec(), dcvec(),
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
        bag.grid,
        get_tmp(bag.ts, T),
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
    grid::TimeGrid,
    ts::AbstractVector{<:Real}, qs::AbstractVector{<:Real},
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

    timenodes!(ts, om, grid, TN)
    @threads for j in 1:n_dt
        qs[j] = pt(ts[j], TN)
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

"""
    getnpicard(mu, rho)

Number of Picard iterations (`M`-applies) per bin the fused sweep needs so that
its discretisation error stays below `1e-2` Poisson sigma at the production
binning (`nbins = ndt = 800`, whole-genome `L`).  Plays for `fusedsweep!` the
role `getorder` plays for the order loop.

This bound is only calibrated for `alpha = rho / (mu + rho) <= 0.8`, i.e.
`rho / mu <= 4`, the largest realistic recombination-to-mutation ratio. Above
that range the returned count is NOT sufficient to keep the error below `1e-2`
Poisson sigma; callers operating there must pass an explicit, larger
`npicard`. For context, the order-loop path is no better in that regime:
`getorder` (src/utils.jl) saturates at its cap of 50 there, so this is a
shared validity bound of the model code, not a regression introduced by the
fused path.

Calibrated in §7.3 of `notes/smcp-integrals-notes.tex` against an order-400
reference, over `hi` from `3e4` to `5e7`; the worst case at each branch is
`1.9e-3` (alpha 0.5), `2.2e-3` (alpha 0.667) and `8.5e-3` (alpha 0.8), the
worst case in the calibrated range. The branch thresholds were also verified
at their upper edges: alpha 0.55 (np = 2) gives `4.0e-3` and alpha 0.72
(np = 3) gives `6.2e-3`. Beyond alpha 0.8 the error grows quickly: alpha 0.90
with np = 4 gives `7.7e-2`, alpha 0.95 gives `5.0e-1`, and reaching `1e-2` at
alpha 0.90 needs `np ~ 10`.
"""
function getnpicard(mu::Real, rho::Real)
    alpha = rho / (mu + rho)
    alpha <= 0.55 && return 2
    alpha <= 0.72 && return 3
    return 4
end

# out = M * x. Vector form of the transition! above; same two passes, same
# clamp conventions. Used by the fused sweep, which has no nrs dimension.
function transition!(out::AbstractVector{<:Real}, x::AbstractVector{<:Real},
    Phi::AbstractVector{<:Real}, dgn::AbstractVector{<:Real}, Gc::AbstractVector{<:Real},
    Ninv::AbstractVector{<:Real}, dC::AbstractVector{<:Real}, om::AbstractVector{<:Real},
    ndt::Int
)
    T = eltype(out)
    sfx = zero(T)
    @inbounds for j in ndt:-1:1
        out[j] = Phi[j] * sfx
        sfx += x[j] * om[j]
    end
    st = zero(T)
    @inbounds for j in 1:ndt
        out[j] += st * Ninv[j] + dgn[j] * x[j]
        st = exp(-dC[j]/2) * (st + Gc[j] * x[j] * om[j])
    end
    return nothing
end

"""
    fusedsweep!(ys, ts, qs, om, Phi, dgn, Gc, Ninv, dC, A, Jf, MJ, J1,
                grid, rs, edges, mu, rho, npicard, n_dt, nrs, TN)

One forward sweep in `r` over the Volterra form of the SMC' recursion, writing
the expected number of segments at `rs` into `ys`.

Solves `J = J_1 + alpha * conv(M J)` by forward substitution instead of
truncating the Neumann series at `order`, so **all** orders are resolved. Each
bin is one exponential-Euler step (stiff `-2*rate*t` term exact, `expm1` against
small-argument cancellation, only non-negative terms added), closed by `npicard`
Picard iterations because `M J` on the bin depends on `J` itself. `MJ` carries
over from the previous bin as the seed; re-seeding it instead with a fresh
`M*J_1` costs one extra apply per bin and is worse than spending that apply on
another Picard iteration (§7.4 of the spec).

The convolution part is rebuilt from the final `MJ` in the terminal loop rather
than being reused from the last Picard iterate, which is free and cuts the
error by 1.3x to 19x depending on `alpha` (§7.4).

Unlike `prordn!` this is sequential in `r` and cannot be threaded over `nrs`.
The node setup is under 0.1% of the runtime, so it is not threaded either.
It does not write `res`: per-order diagnostics require `prordn!`.
"""
function fusedsweep!(ys::AbstractVector{<:Real},
    ts::AbstractVector{<:Real}, qs::AbstractVector{<:Real},
    om::AbstractVector{<:Real}, Phi::AbstractVector{<:Real}, dgn::AbstractVector{<:Real},
    Gc::AbstractVector{<:Real}, Ninv::AbstractVector{<:Real}, dC::AbstractVector{<:Real},
    A::AbstractVector{<:Real}, Jf::AbstractVector{<:Real},
    MJ::AbstractVector{<:Real}, J1::AbstractVector{<:Real},
    grid::TimeGrid,
    rs::AbstractVector{<:Real}, edges::AbstractVector{<:Real}, mu::Real, rho::Real,
    npicard::Int, n_dt::Int, nrs::Int,
    TN::AbstractVector{<:Real}
)
    @assert length(rs) == nrs
    @assert length(edges) == nrs + 1
    @assert npicard >= 1

    T = eltype(ys)
    rate = mu + rho
    alpha = rho / rate

    timenodes!(ts, om, grid, TN)
    for j in 1:n_dt
        qs[j] = pt(ts[j], TN)
    end
    sepkernel!(Phi, dgn, Gc, Ninv, dC, ts, TN)

    fill!(A, zero(T))
    fill!(MJ, zero(T))

    scale = 2 * mu * TN[1] * (mu / rate)
    @inbounds for i in 1:nrs
        w = edges[i+1] - edges[i]
        # Unit bins hold exactly the segments of length edges[i], so their
        # representative point is the lower edge and the self-bin contribution
        # spans the full width. Wider bins report a density at the geometric
        # midpoint and use the partial width. The carried accumulator always
        # advances by the full width. See §6 of the spec.
        wi = w <= 1 ? w : rs[i] - edges[i]
        for j in 1:n_dt
            J1[j] = rate * exp(-2rate * rs[i] * ts[j]) * qs[j]
        end
        for _ in 1:npicard
            for j in 1:n_dt
                t = ts[j]
                Jf[j] = J1[j] + A[j] * exp(-2rate * (rs[i] - edges[i]) * t) +
                        alpha * MJ[j] * (- expm1(-2rate * wi * t)) / 2t
            end
            transition!(MJ, Jf, Phi, dgn, Gc, Ninv, dC, om, n_dt)
        end
        s = zero(T)
        for j in 1:n_dt
            t = ts[j]
            # convolution part rebuilt from the MJ the last Picard apply
            # produced, one iterate fresher than the Jf above
            jc = A[j] * exp(-2rate * (rs[i] - edges[i]) * t) +
                 alpha * MJ[j] * (- expm1(-2rate * wi * t)) / 2t
            # terminal t integral of the convolution part; 2t from p(r|t)
            s += jc * 2 * t * om[j]
            # roll the accumulator from edges[i] to edges[i+1]
            A[j] = exp(-2rate * w * t) * A[j] +
                   alpha * MJ[j] * (- expm1(-2rate * w * t)) / 2t
        end
        # order 1 comes from the analytic firstorder, as in prordn!'s res[:,1]
        ys[i] = (firstorder(rs[i], rate, TN) + s) * scale
    end
    return nothing
end

"""
    fusedsweep!(bag::IntegralArrays, rs, edges, mu, rho, TN; npicard = 0)

Bag wrapper for the fused sweep. `npicard = 0` selects the iteration count with
`getnpicard(mu, rho)`; pass a positive value to override it. Writes `bag.ys`
and leaves `bag.res` untouched.
"""
function fusedsweep!(bag::IntegralArrays,
    rs::AbstractVector{<:Real}, edges::AbstractVector{<:Real}, mu::Real, rho::Real,
    TN::AbstractVector{<:Real}; npicard::Int = 0
)
    T = eltype(TN)
    np = npicard > 0 ? npicard : getnpicard(mu, rho)
    fusedsweep!(
        get_tmp(bag.ys, T),
        get_tmp(bag.ts, T),
        get_tmp(bag.qs, T),
        get_tmp(bag.om, T),
        get_tmp(bag.Phi, T),
        get_tmp(bag.dgn, T),
        get_tmp(bag.Gc, T),
        get_tmp(bag.Ninv, T),
        get_tmp(bag.dC, T),
        get_tmp(bag.A, T),
        get_tmp(bag.Jf, T),
        get_tmp(bag.MJ, T),
        get_tmp(bag.J1, T),
        bag.grid, rs, edges, mu, rho, np, bag.n_dt, bag.nrs, TN
    )
    return nothing
end

end