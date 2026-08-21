module SMCpIntegrals

using FastGaussQuadrature
using LinearAlgebra
using PreallocationTools

using ..CoalescentBase

export IntegralArrays, fusedsweep!, getnpicard,
    firstorder, firstorderint, TimeGrid, timegrid, timenodes!, ndt, npanels


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
#
# The two branches agree in value at t' = t but not in slope, so ptt(t, .) has a
# CORNER at t' = t whose location moves with the row. The exponential rescaling
# that keeps the lower branch stable is carried by EE/EB, built by timenodes!;
# see transition! below.
function sepkernel!(Phi::AbstractVector{<:Real}, dgn::AbstractVector{<:Real},
    Gc::AbstractVector{<:Real}, Ninv::AbstractVector{<:Real},
    ts::AbstractVector{<:Real}, TN::AbstractVector{<:Real}
)
    n = length(ts)
    n0 = Nt(0, TN)
    @inbounds for j in 1:n
        t = ts[j]
        c = cumcr(0, t, TN)
        nt = Nt(t, TN)
        g = nt + margrecomb(t, TN) - n0 * exp(-c)
        Gc[j] = max(g, zero(g))
        Phi[j] = Gc[j] / nt
        dgn[j] = max(t - g, zero(g))
        Ninv[j] = 1 / nt
    end
    return nothing
end

"""
    TimeGrid(K; msub = 8, nfin = 12, ntail = 16,
              fmin = 1e-6, umin = 1e-8, umax = 25.0, dumax = 1.0)
    timegrid(K; msub = 0, nfin = 0, ntail = 0)

Reference nodes, weights and partial-integral weights for the composite panel-wise
time quadrature of a `K`-epoch history. Holds nothing that depends on `TN`: the
panels are built from the epoch times at evaluation time by [`timenodes!`](@ref).

The time axis is tiled by `npanels(g)` **sub-panels**, each carrying `msub`
Gauss-Legendre nodes:

- epoch `k = 1 … K-1` is split into `nfin` equal sub-panels of `[T_k, T_{k+1}]`,
  with the nodes placed affinely in `t`;
- the final epoch `[T_K, ∞)` is worked in the coalescent variable
  `u = (t - T_K) / 2N_K`, where the coalescent factor is exactly
  `exp(-C(T_K)/2) * exp(-u)`. Its mesh is **geometric** from `umin` to 1 in
  `ntail` steps and then uniform in steps of `dumax` up to `umax`. The geometric
  part is what resolves the recombination decay `exp(-2·rate·r·t)`, whose scale
  in `u` is `1/(4·rate·r·N_K)` and reaches ~1e-5 at the largest `r`: a geometric
  mesh resolves an exponential whose scale is not known in advance. Inside each
  sub-panel the nodes are placed affinely in `u`.

Placing the nodes affinely inside every sub-panel is what makes
`om[j] = wleg[i] * (sub-panel half-width)` hold everywhere, so the single
`msub × msub` matrix `Lpw` serves them all.

`Lpw` holds the **partial integrals of the local Lagrange basis**, normalised by
the Gauss weight:

    Lpw[q,i] * wleg[i] = ∫_{-1}^{z_q} l_i(z) dz

so that, for a sub-panel with nodes `x[s0+i]` and weights `om[s0+i]`,

    ∫_{a_p}^{t_{s0+q}} x dt = Σ_i Lpw[q,i] * om[s0+i] * x[s0+i]

is exact for any `x` of degree `< msub`. This is what removes the moving corner of
the transition kernel from the quadrature: the corner always sits *at a node*, so
only the sub-panel containing the row is ever partial, and its partial integral
costs one length-`msub` dot product. See
`docs/superpowers/specs/2026-08-21-diagonal-corner-correction-design.md`.

Truncating the tail at `u = umax` costs `~umax·exp(-umax)` in relative terms
(3.5e-10 at the default 25). `umin` sets how far towards the origin the geometric
part reaches. `dumax` caps the coalescent width `du` of a tail sub-panel by
splitting any wider one: the lower branch rescales by `exp(du)` inside a
sub-panel, so a wide sub-panel amplifies whatever it is fed and makes the Picard
iteration diverge (measured: `du ≈ 19` diverges, `du ≈ 12` does not). All three
are grid constants applied once in the constructor, never a runtime test on `TN`.

The same bound applies to the finite epochs, where it is `dC_k/(2·nfin)` and
therefore TN-dependent: `nfin` must be large enough for the coalescent mass the
histories carry. A fitted history holds `dC_k/2 ≲ 2` per epoch, which `nfin = 4`
covers with room to spare, but an epoch far outside that (a population size near
the lower bound held across a very wide epoch) will produce `NaN`. That is the
same deliberate no-guard trade recorded for the overflow case below.
"""
struct TimeGrid
    msub::Int
    nfin::Int
    ntail::Int
    K::Int
    zleg::Vector{Float64}
    wleg::Vector{Float64}
    Lpw::Matrix{Float64}
    fedge::Vector{Float64}
    uedge::Vector{Float64}
end

# Lpw[q,i] * wleg[i] = int_{-1}^{z_q} l_i(z) dz, exactly.
#
# Gauss-Legendre orthogonality gives the Lagrange basis in closed form,
#   l_i(z) = sum_{k=0}^{msub-1} ((2k+1)/2) * wleg[i] * P_k(z_i) * P_k(z),
# (exact because l_i has degree msub-1 and the rule integrates degree 2·msub-1),
# and int_{-1}^{x} P_0 = x+1, int_{-1}^{x} P_k = (P_{k+1}(x) - P_{k-1}(x))/(2k+1).
# So no linear solve and no Vandermonde inverse is needed.
function partialweights(z::Vector{Float64}, w::Vector{Float64})
    m = length(z)
    Lpw = zeros(Float64, m, m)
    P  = zeros(Float64, m + 1)          # P_0 .. P_m at the current point
    Pz = zeros(Float64, m, m)           # Pz[k+1,i] = P_k(z_i)
    for i in 1:m
        legendre!(P, z[i], m)
        for k in 0:m-1
            Pz[k+1, i] = P[k+1]
        end
    end
    for q in 1:m
        legendre!(P, z[q], m)
        for k in 0:m-1
            Ik = k == 0 ? z[q] + 1 : (P[k+2] - P[k]) / (2k + 1)
            c = (2k + 1) / 2 * Ik
            for i in 1:m
                Lpw[q, i] += c * Pz[k+1, i]
            end
        end
    end
    return Lpw
end

# P[k+1] = P_k(x) for k = 0 .. n, by the three-term recurrence
function legendre!(P::Vector{Float64}, x::Float64, n::Int)
    P[1] = 1.0
    n >= 1 && (P[2] = x)
    for k in 1:n-1
        P[k+2] = ((2k + 1) * x * P[k+1] - k * P[k]) / (k + 1)
    end
    return P
end

"""
Default sub-panel counts, the single place they are set. `timegrid` substitutes
them for any argument passed as zero, which is how `FitOptions` and `mldsmcp`
spell "leave it at the default".
"""
const TIMEGRID_DEFAULTS = (msub = 8, nfin = 12, ntail = 16)

function TimeGrid(K::Int;
    msub::Int = TIMEGRID_DEFAULTS.msub,
    nfin::Int = TIMEGRID_DEFAULTS.nfin,
    ntail::Int = TIMEGRID_DEFAULTS.ntail,
    fmin::Float64 = 1.0e-6, umin::Float64 = 1.0e-8, umax::Float64 = 25.0,
    dumax::Float64 = 1.0
)
    @assert K >= 1 "need at least one epoch"
    @assert msub >= 2 "need at least 2 nodes per sub-panel"
    @assert nfin >= 1 "need at least 1 sub-panel per finite epoch"
    @assert ntail >= 1 "need at least 1 sub-panel in the tail"
    @assert 0 < fmin < 1 "fmin must lie in (0,1)"
    @assert 0 < umin < 1 "umin must lie in (0,1)"
    @assert umax > 1 "tail truncation must exceed 1"
    @assert dumax > 0 "dumax must be positive"
    z, w = gausslegendre(msub)
    return TimeGrid(msub, nfin, ntail, K, z, w, partialweights(z, w),
                    geomedges(nfin, fmin), tailedges(ntail, umin, umax, dumax))
end

# Sub-panel edges as fractions of a finite epoch's width, geometric towards the
# epoch's left endpoint. Same reasoning as the tail mesh: near t = 0 the
# integrand carries exp(-2*rate*r*t), whose scale is ~3 generations at the
# largest r, and the epoch width is not known in advance — a geometric mesh
# resolves a boundary layer of unknown scale, while the handful of panels above
# it cover the smooth bulk spectrally. The edges are fixed fractions, so every
# node stays exactly affine in TN.
function geomedges(nfin::Int, fmin::Float64)
    f = [0.0, fmin]
    ratio = (1 / fmin)^(1 / nfin)
    for q in 1:nfin
        push!(f, fmin * ratio^q)
    end
    f[end] = 1.0
    return f
end

# Tail sub-panel edges in u.
#
# Two regimes, because the tail integrand carries two decays with unrelated
# scales: the coalescent exp(-u), whose scale is 1, and the recombination
# exp(-4*rate*r*N_K*u), whose scale 1/(4*rate*r*N_K) reaches ~1e-5 at the largest
# r. A geometric mesh resolves an exponential whose scale is not known in advance
# — the panel straddling the scale is a few decay lengths wide and the ones below
# it are finer still — so the mesh is geometric from `umin` to 1 in `ntail` steps,
# then steps of at most `dumax` up to `umax`. The leading `[0, umin]` panel closes
# the domain at the origin.
#
# `dumax` is what keeps the correction stable: the lower branch rescales by
# exp(du) inside a sub-panel, so a wide one amplifies whatever it is fed.
function tailedges(ntail::Int, umin::Float64, umax::Float64, dumax::Float64)
    uedge = [0.0, umin]
    ratio = (1 / umin)^(1 / ntail)
    for q in 1:ntail
        push!(uedge, umin * ratio^q)
    end
    uedge[end] = 1.0
    while uedge[end] < umax
        push!(uedge, min(umax, uedge[end] + dumax))
    end
    # honour dumax in the geometric part too (it only binds if umin is large)
    out = [0.0]
    for q in 1:length(uedge)-1
        a, b = uedge[q], uedge[q+1]
        nsplit = max(1, ceil(Int, (b - a) / dumax))
        for i in 1:nsplit
            push!(out, a + (b - a) * i / nsplit)
        end
    end
    return out
end

"""
    npanels(g::TimeGrid)

Number of sub-panels tiling the time axis: `(K-1)` epochs of `nfin` each, plus
the tail mesh. The tail count is `ntail` before the `dumax` split and may be
larger after it; `length(g.uedge) - 1` is the count that matters.
"""
npanels(g::TimeGrid) = (g.K - 1) * (length(g.fedge) - 1) + length(g.uedge) - 1

function timegrid(K::Int; msub::Int = 0, nfin::Int = 0, ntail::Int = 0)
    TimeGrid(K;
        msub  = iszero(msub)  ? TIMEGRID_DEFAULTS.msub  : msub,
        nfin  = iszero(nfin)  ? TIMEGRID_DEFAULTS.nfin  : nfin,
        ntail = iszero(ntail) ? TIMEGRID_DEFAULTS.ntail : ntail,
    )
end

"""
    ndt(g::TimeGrid)

Total number of quadrature nodes, `npanels(g) * msub`.
"""
ndt(g::TimeGrid) = npanels(g) * g.msub

"""
    timenodes!(ts, om, EE, EB, g::TimeGrid, TN)

Fill `ts` with the quadrature nodes and `om` with their weights for the history
`TN`, plus the two exponential rescalings the transition apply needs:

- `EE[j] = exp((C(t_j) - C(a_p))/2)`, with `a_p` the left edge of `j`'s sub-panel
- `EB[p] = exp((C(b_p) - C(a_p))/2)`, the sub-panel's own coalescent width

Both are read straight off the affine maps — inside epoch `k`, `d(C/2)/dt` is
`1/(2N_k)`, and in the tail `d(C/2) = du` exactly — so no `cumcr` call is needed.

Sub-panels are pinned to the epoch boundaries, so each node is an **affine**
function of the epoch parameters: nodes move smoothly with `TN` and a node can
never migrate from one epoch to another. `ts` comes out ascending, as
`sepkernel!` and `transition!` require.

!!! note "No overflow guard"
    `EE` is bounded by `exp(dC_p/2)` over a sub-panel. In a finite epoch that is
    TN-dependent and the parameter box permits it to exceed 709, in which case
    this returns `Inf` and the sweep returns `NaN`. Reaching it needs a
    population size near the lower bound held across a very wide epoch, where the
    likelihood is already numerically zero. This is deliberate: the alternatives
    either dent the C¹ invariant or add machinery for a degenerate region.
"""
function timenodes!(ts::AbstractVector{<:Real}, om::AbstractVector{<:Real},
    EE::AbstractVector{<:Real}, EB::AbstractVector{<:Real},
    g::TimeGrid, TN::AbstractVector{<:Real}
)
    K = length(TN) ÷ 2
    @assert K == g.K "grid built for $(g.K) epochs, got $K"
    @assert length(ts) == ndt(g) "ts has length $(length(ts)), expected $(ndt(g))"
    @assert length(om) == ndt(g) "om has length $(length(om)), expected $(ndt(g))"
    @assert length(EE) == ndt(g) "EE has length $(length(EE)), expected $(ndt(g))"
    @assert length(EB) == npanels(g) "EB has length $(length(EB)), expected $(npanels(g))"

    j = 0
    p = 0
    @inbounds for k in 1:K-1
        a0 = getts(TN, k)
        b0 = getts(TN, k + 1)
        b0 > a0 || throw(ArgumentError(
            "timenodes!: epoch $k has non-positive width: T_$k = $a0, T_$(k+1) = $b0"
        ))
        Nk = getns(TN, k)
        Nk > 0 || throw(ArgumentError(
            "timenodes!: population size N_$k must be strictly positive, got $Nk"
        ))
        W = b0 - a0
        for q in 1:length(g.fedge)-1
            p += 1
            a = a0 + W * g.fedge[q]
            h = W * (g.fedge[q+1] - g.fedge[q]) / 2   # sub-panel half-width in t
            c = a + h
            EB[p] = exp(2 * h / (2 * Nk))
            for i in 1:g.msub
                j += 1
                t = c + h * g.zleg[i]
                ts[j] = t
                om[j] = g.wleg[i] * h
                EE[j] = exp((t - a) / (2 * Nk))
            end
        end
    end
    TK = getts(TN, K)
    NK = getns(TN, K)
    NK > 0 || throw(ArgumentError(
        "timenodes!: tail population size N_$K must be strictly positive, got $NK"
    ))
    twoNK = 2 * NK
    @inbounds for q in 1:length(g.uedge)-1
        p += 1
        ua = g.uedge[q]
        hu = (g.uedge[q+1] - ua) / 2        # sub-panel half-width in u
        cu = ua + hu
        EB[p] = exp(2 * hu)
        for i in 1:g.msub
            j += 1
            u = cu + hu * g.zleg[i]
            ts[j] = TK + twoNK * u
            om[j] = g.wleg[i] * hu * twoNK
            EE[j] = exp(u - ua)
        end
    end
    return nothing
end

struct IntegralArrays{T}
    n_dt::Int
    nrs::Int
    ys::DiffCache{Vector{T},Vector{T}}
    grid::TimeGrid
    ts::DiffCache{Vector{T},Vector{T}}
    qs::DiffCache{Vector{T},Vector{T}}
    om::DiffCache{Vector{T},Vector{T}}
    EE::DiffCache{Vector{T},Vector{T}}
    EB::DiffCache{Vector{T},Vector{T}}
    Phi::DiffCache{Vector{T},Vector{T}}
    dgn::DiffCache{Vector{T},Vector{T}}
    Gc::DiffCache{Vector{T},Vector{T}}
    Ninv::DiffCache{Vector{T},Vector{T}}
    A::DiffCache{Vector{T},Vector{T}}
    Jf::DiffCache{Vector{T},Vector{T}}
    MJ::DiffCache{Vector{T},Vector{T}}
    J1::DiffCache{Vector{T},Vector{T}}
end

function IntegralArrays(grid::TimeGrid, nrs::Int, chunk, levels = 1)
    n = ndt(grid)
    dcvec(len = n) = DiffCache(zeros(Float64, len), chunk; levels)
    IntegralArrays(
        n, nrs,
        DiffCache(zeros(Float64, nrs), chunk; levels),
        grid,
        dcvec(), dcvec(), dcvec(), dcvec(), dcvec(npanels(grid)),
        dcvec(), dcvec(), dcvec(), dcvec(),
        dcvec(), dcvec(), dcvec(), dcvec()
    )
end

"""
    getnpicard(mu, rho)

Number of Picard iterations (`M`-applies) per bin the fused sweep needs so that
its discretisation error stays below `1e-2` Poisson sigma at the production
binning (`nbins = 800`, whole-genome `L`).

This bound is only calibrated for `alpha = rho / (mu + rho) <= 0.8`, i.e.
`rho / mu <= 4`, the largest realistic recombination-to-mutation ratio. Above
that range the returned count is NOT sufficient to keep the error below `1e-2`
Poisson sigma; callers operating there must pass an explicit, larger
`npicard`. This is a validity bound of the model code, not of the fused path
in particular: the order loop saturated its own cap in the same regime.

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

# out = M * x, with M the semiseparable transition operator.
#
# The kernel is C0 but has a CORNER at t' = t whose location moves with the row,
# so splitting one global Gauss rule at the row index — using whole-panel weights
# on what is really a partial interval — converges only as O(1/n). Because the
# corner always sits AT A NODE, the fix is local: whole sub-panels below and above
# the row are integrated exactly by their own Gauss rules, and only the sub-panel
# containing the row is partial. Its partial integral is the integral of the local
# interpolant, one length-msub dot product against `Lpw` (see TimeGrid), so the
# apply stays O(n * msub).
#
# Upper branch, backward over sub-panels: sfx is the exact int_{b_p}^{tmax} x dt'.
# Lower branch, forward: st is int_0^{a_p} exp(-(C(a_p)-C(t'))/2) G x dt',
# referenced to the current sub-panel's LEFT edge, which is what keeps the
# rescaling bounded; EE/EB carry the reference changes. The diagonal atom
# dgn[j]*x[j] is exact and untouched.
function transition!(out::AbstractVector{<:Real}, x::AbstractVector{<:Real},
    Phi::AbstractVector{<:Real}, dgn::AbstractVector{<:Real}, Gc::AbstractVector{<:Real},
    Ninv::AbstractVector{<:Real}, EE::AbstractVector{<:Real}, EB::AbstractVector{<:Real},
    om::AbstractVector{<:Real}, g::TimeGrid
)
    T = eltype(out)
    m = g.msub
    npan = npanels(g)
    Lpw = g.Lpw

    sfx = zero(T)
    @inbounds for p in npan:-1:1
        s0 = (p - 1) * m
        Sp = zero(T)
        for i in 1:m
            Sp += x[s0+i] * om[s0+i]
        end
        for q in 1:m
            low = zero(T)
            for i in 1:m
                low += Lpw[q,i] * om[s0+i] * x[s0+i]
            end
            out[s0+q] = Phi[s0+q] * (sfx + Sp - low)
        end
        sfx += Sp
    end

    st = zero(T)
    @inbounds for p in 1:npan
        s0 = (p - 1) * m
        for q in 1:m
            low = zero(T)
            for i in 1:m
                low += Lpw[q,i] * om[s0+i] * EE[s0+i] * Gc[s0+i] * x[s0+i]
            end
            out[s0+q] += (st + low) * Ninv[s0+q] / EE[s0+q] + dgn[s0+q] * x[s0+q]
        end
        Sp = zero(T)
        for i in 1:m
            Sp += om[s0+i] * EE[s0+i] * Gc[s0+i] * x[s0+i]
        end
        st = (st + Sp) / EB[p]
    end
    return nothing
end

"""
    fusedsweep!(ys, ts, qs, om, EE, EB, Phi, dgn, Gc, Ninv, A, Jf, MJ, J1,
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

The sweep is sequential in `r` by construction and is not threaded.
"""
function fusedsweep!(ys::AbstractVector{<:Real},
    ts::AbstractVector{<:Real}, qs::AbstractVector{<:Real},
    om::AbstractVector{<:Real}, EE::AbstractVector{<:Real}, EB::AbstractVector{<:Real},
    Phi::AbstractVector{<:Real}, dgn::AbstractVector{<:Real},
    Gc::AbstractVector{<:Real}, Ninv::AbstractVector{<:Real},
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

    timenodes!(ts, om, EE, EB, grid, TN)
    for j in 1:n_dt
        qs[j] = pt(ts[j], TN)
    end
    sepkernel!(Phi, dgn, Gc, Ninv, ts, TN)

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
            transition!(MJ, Jf, Phi, dgn, Gc, Ninv, EE, EB, om, grid)
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
        # order 1 comes from the analytic firstorder
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
        get_tmp(bag.EE, T),
        get_tmp(bag.EB, T),
        get_tmp(bag.Phi, T),
        get_tmp(bag.dgn, T),
        get_tmp(bag.Gc, T),
        get_tmp(bag.Ninv, T),
        get_tmp(bag.A, T),
        get_tmp(bag.Jf, T),
        get_tmp(bag.MJ, T),
        get_tmp(bag.J1, T),
        bag.grid, rs, edges, mu, rho, np, bag.n_dt, bag.nrs, TN
    )
    return nothing
end

end