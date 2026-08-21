# Fused Volterra Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `order`-truncated Neumann series in `mldsmcp` with a single forward sweep in `r` that resolves all orders exactly, cutting cost from `O(order·nrs·ndt)` to `O(npicard·nrs·ndt)` with `npicard` = 2–4 against `order` = 8–43.

**Architecture:** The order-summed generating function `J(r,t) = Σ_o α^(o-1) J_o(r,t)` obeys a Volterra equation of the second kind with a causal kernel, so forward substitution in `r` generates the whole Neumann series implicitly in one pass. Each bin step is the same exponential-Euler step the current `acc` recursion already performs — stiff diagonal treated exactly, `expm1` against small-argument cancellation, only non-negative quantities added — made implicit by the fact that `(MJ)_i` depends on `J_i`, and closed by `npicard` Picard iterations per bin. The semiseparable `M`-apply from the previous plan is reused verbatim. Bin conventions (`w <= 1`) and public signatures are unchanged; the existing `prordn!` order loop is kept as the per-order diagnostic path, the test reference, and the opt-out.

**Tech Stack:** Julia 1.12, ForwardDiff, PreallocationTools (`DiffCache`), Base.Threads, Test, HistogramBinnings.

**Spec:** `../../../../notes/smcp-integrals-notes.tex` (in the parent `minus3-simulation` directory). §3 derives the ODE, §5 derives the fused step and corrects the Picard diagnosis, §6 fixes the `w <= 1` reporting convention, §7.3 is the calibration this plan's acceptance thresholds come from, §8 gives the go/no-go criterion. The measurement script behind §7.3 is `../../../../notes/fused_prototype.jl`.

**Predecessor:** `2026-08-14-semiseparable-transition.md` (complete, merged as `7aa1dd9`–`4d445de`). This plan implements the "fused order loop" that plan explicitly deferred.

## Global Constraints

- Branch: `performance` (already checked out, at `4d445de`).
- **This is an approximation, not an algebraic identity.** Unlike the predecessor plan, agreement with the order loop is *not* to machine precision. The acceptance criterion is stated in noise units: at the production binning (`nbins = ndt = 800`, three-epoch history, `L = TN[1] = 3e9` bp), `max |z| < 1e-2` where `z_i = (m_i - m_i^ref)/sqrt(m_i^ref)` over bins with `m_i^ref > 10`. Never assert `≈` against the order loop at tight tolerance.
- **Do not modify `prordn!`, `transition!(::AbstractMatrix, ...)`, `sepkernel!`, `ptt`, `pt`, `firstorder` or `tolegendre`.** They are the reference the new code is tested against, and `prordn!` remains the `method = :order` path.
- Must stay ForwardDiff-compatible: every new buffer is a `DiffCache`, all new arithmetic is `+ - * / exp expm1` only. Use `zero(T)`/`fill!(x, zero(T))` — never a literal `0.0` — for any accumulator whose element type follows `TN`.
- `mu`, `rho`, `rs` and `edges` are always `Float64`; only `TN` carries duals. `alpha = rho/(mu+rho)` and `rate = mu+rho` are therefore plain `Float64`.
- `ts` nodes are ascending and must not be reordered.
- `IntegralArrays`' public constructor signature stays `IntegralArrays(order, ndt, nrs, chunk, levels = 1)`.
- **Default-flip policy.** `mldsmcp` (the convenience entry, used only by tests) defaults to `method = :fused`. `mldsmcp!` (the mutating entry, used by `mle_optimization.jl`, `corrections.jl` and `IBSpector.jl`) defaults to `method = :order`, so no existing caller changes behaviour in this branch. Flipping that second default is a one-line follow-up, gated on comparing an end-to-end fit — out of scope here (see "Out of Scope").
- All existing testsets must stay green.

## Background: the scheme being implemented

Write `λ = mu + rho`, `α = rho/λ`, `J_1(r,t) = λ e^{-2λrt} q(t)`, and `M` for the transition operator applied by `transition!`. The order-summed `J = Σ_o α^{o-1} J_o` satisfies

```
J(r,t) = J_1(r,t) + α ∫_0^r dr' e^{-2λ(r-r')t} (MJ)(r',t)
```

Discretised with one exponential-Euler step per reporting bin `i = 1..nrs`, carrying `A` = the convolution part evaluated at the **left edge** `edges[i]`:

```
w   = edges[i+1] - edges[i]
wi  = w <= 1 ? w : rs[i] - edges[i]          # reported vs carried width, see spec §6
Jc  = A * exp(-2λ(rs[i]-edges[i])t) + α·U·(-expm1(-2λ·wi·t))/(2t)    # reported
J   = J_1(rs[i],t) + Jc
U   = M J                                                            # implicit
A'  = exp(-2λ·w·t)·A          + α·U·(-expm1(-2λ·w ·t))/(2t)          # carried
```

`U` is unknown at the top of the bin, so the three middle lines are iterated
`npicard` times, seeded with `U` left over from bin `i-1` (`U = 0` for `i = 1`).
The observable is

```
ys[i] = (firstorder(rs[i], λ, TN) + Σ_j Jc[j]·2·ts[j]·om[j]) · 2·mu·TN[1]·(mu/λ)
```

— note the order-1 term uses the **analytic** `firstorder`, exactly as the order
loop's `res[:,1]` does, while `Jc` carries only the convolution part. The `2 mu
TN[1] (mu/λ)` factor is what `mldsmcp!` applies to `res[:,1]`; the `α^{o-1}`
factors it applies to higher orders are already inside `J`.

**Why `npicard` is small (spec §5).** Picard contracts by ≈1/3 per extra
`M`-apply at every binning tested. The floor it converges to is the
exponential-Euler step error, first order in `Δr`, which at `nbins = 800` is
already ~100× below Poisson noise. Neither term needs sub-stepping.

## File Structure

- `src/Spectra/SMCpIntegrals.jl` — append `getnpicard`, a vector method of `transition!`, and both `fusedsweep!` methods after the existing `prordn!`; add five length-`n_dt` `DiffCache` fields to `IntegralArrays`. Single responsibility unchanged: the SMC' integrals. Ends at ~450 lines, in line with `src/utils.jl` (472).
- `src/Spectra/Spectra.jl` — add the `method`/`npicard` keywords to `mldsmcp` and `mldsmcp!`, and export `getnpicard`.
- `test/smcp_fused.jl` — **new**. Owns all testing for this change: Picard convergence against the order loop, the noise-unit calibration that is the real acceptance gate, robustness at large `hi`, and ForwardDiff. Self-contained.
- `test/runtests.jl` — one `include` line.

---

### Task 1: `getnpicard` and the fused sweep core

**Files:**
- Modify: `src/Spectra/SMCpIntegrals.jl` (append after `prordn!`, i.e. after line 287, before the module's closing `end`)
- Create: `test/smcp_fused.jl`
- Modify: `test/runtests.jl` (add include)

**Interfaces:**
- Consumes: `sepkernel!`, `transition!`, `pt`, `firstorder`, `tolegendre`, `gausslegendre`, `prordn!`, `IntegralArrays` (all module-local); `Nt`, `cumcr` from `..CoalescentBase`
- Produces:
  - `getnpicard(mu::Real, rho::Real) -> Int` — returns 2, 3 or 4
  - `transition!(out::AbstractVector, x::AbstractVector, Phi, dgn, Gc, Ninv, dC, om, ndt) -> nothing` — vector method, `out = M*x`
  - `fusedsweep!(ys, ts, dts, qs, om, Phi, dgn, Gc, Ninv, dC, A, Jc, Jf, MJ, J1, zs, wt, rs, edges, mu, rho, npicard, n_dt, nrs, TN) -> nothing` — writes `nrs` expected counts into `ys`

- [ ] **Step 1: Write the failing test**

Create `test/smcp_fused.jl`:

```julia
using IBSpector
using IBSpector.Spectra
using IBSpector.Spectra.PreallocationTools
using IBSpector.Spectra.SMCpIntegrals: getnpicard, fusedsweep!, transition!, sepkernel!
using HistogramBinnings
using StatsBase
using Test

const SMCp = IBSpector.Spectra.SMCpIntegrals

# Production binning: log-spaced edges pushed up to distinct integers, then the
# geometric midpoint for wide bins and the lower edge for unit bins.
function prodgrid(nbins, hi)
    ev = IBSpector.CustomEdgeVector(lo = 1, hi = hi, nbins = nbins)
    collect(Float64, ev), collect(Float64, midpoints(ev))
end

# Order-loop reference, summed over orders with alpha and scaled exactly as
# mldsmcp! scales it. `order` must be large enough to be converged.
function orderref(rs, edges, mu, rho, ndt, TN, order)
    rate = mu + rho
    alpha = rho / rate
    bag = IntegralArrays(order, ndt, length(rs), Val{length(TN)})
    SMCp.prordn!(bag, rs, edges, rate, TN)
    res = get_tmp(bag.res, eltype(TN))
    scale = 2 * mu * TN[1] * (mu / rate)
    [sum(res[i, o] * alpha^(o - 1) for o in 1:order) * scale for i in eachindex(rs)]
end

# Raw fusedsweep! with freshly allocated Float64 buffers.
function rawfused(rs, edges, mu, rho, ndt, TN, npicard)
    nrs = length(rs)
    zs, wt = SMCp.gausslegendre(ndt)
    v() = zeros(Float64, ndt)
    ys = zeros(Float64, nrs)
    fusedsweep!(ys, v(), v(), v(), v(), v(), v(), v(), v(), v(),
                v(), v(), v(), v(), v(),
                zs, wt, rs, edges, mu, rho, npicard, ndt, nrs, TN)
    ys
end

@testset "getnpicard" begin
    mu = 1.25e-8
    @test getnpicard(mu, 0.25mu) == 2     # alpha = 0.20
    @test getnpicard(mu, 1.0mu)  == 2     # alpha = 0.50
    @test getnpicard(mu, 2.0mu)  == 3     # alpha = 0.667
    @test getnpicard(mu, 4.0mu)  == 4     # alpha = 0.80
    @test getnpicard(mu, 0.0)    == 2     # alpha = 0, degenerate but legal
end

@testset "transition! vector method == matrix method" begin
    TN = [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0]
    ndt = 120
    zs, wt = SMCp.gausslegendre(ndt)
    ts = zeros(ndt); om = zeros(ndt)
    for j in 1:ndt
        t, dt = SMCp.tolegendre(zs[j], TN)
        ts[j] = t
        om[j] = wt[j] * dt
    end
    Phi = zeros(ndt); dgn = zeros(ndt); Gc = zeros(ndt)
    Ninv = zeros(ndt); dC = zeros(ndt)
    sepkernel!(Phi, dgn, Gc, Ninv, dC, ts, TN)

    x = abs.(randn(ndt)) .* 1e-6
    out = zeros(ndt)
    transition!(out, x, Phi, dgn, Gc, Ninv, dC, om, ndt)

    jprt = reshape(copy(x), ndt, 1)
    temp = zeros(1, ndt)
    transition!(temp, jprt, 1, Phi, dgn, Gc, Ninv, dC, om, ndt)
    @test out ≈ vec(temp[1, :])
end

@testset "fused sweep converges to the order loop under Picard" begin
    TN = [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0]
    mu = 1.25e-8
    ndt = 200
    edges, rs = prodgrid(200, 30_000)

    for ratio in (1.0, 4.0)
        rho = mu * ratio
        ref = orderref(rs, edges, mu, rho, ndt, TN, 200)
        errs = [maximum(abs.(rawfused(rs, edges, mu, rho, ndt, TN, np) .- ref) ./ abs.(ref))
                for np in 1:6]
        @test all(isfinite, errs)
        # Picard contracts by about 1/3 at first, then tails off as the error
        # approaches the exponential-Euler step floor, which is nonzero and is
        # what remains after Picard has converged. So the successive ratios
        # rise toward 1 and only the first one is near 1/3. Measured on this
        # grid: alpha 0.5 -> 0.26 0.36 0.40 0.41 0.42, alpha 0.8 -> 0.33 0.52
        # 0.60 0.65 0.68. Do not tighten these to 0.5 across the board.
        for k in 1:5
            @test errs[k+1] < errs[k]
        end
        @test errs[2] < 0.5 * errs[1]
        @test errs[6] < 0.1 * errs[1]
    end
end

@testset "fused sweep is positive and finite" begin
    TN = [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0]
    mu = 1.25e-8
    rho = 4mu
    edges, rs = prodgrid(200, 30_000)
    for np in 1:4
        ys = rawfused(rs, edges, mu, rho, 200, TN, np)
        @test all(isfinite, ys)
        @test all(ys .> 0)
    end
end
```

Add to `test/runtests.jl` immediately after the existing `include("smcp_semiseparable.jl")`:

```julia
include("smcp_fused.jl")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL — `UndefVarError: getnpicard not defined in IBSpector.Spectra.SMCpIntegrals`

- [ ] **Step 3: Write minimal implementation**

In `src/Spectra/SMCpIntegrals.jl`, extend the export line at the top from

```julia
export IntegralArrays, prordn!,
    firstorder, firstorderint
```

to

```julia
export IntegralArrays, prordn!, fusedsweep!, getnpicard,
    firstorder, firstorderint
```

then append, after the closing `end` of the second `prordn!` (line 287) and
before the module's final `end`:

```julia
"""
    getnpicard(mu, rho)

Number of Picard iterations (`M`-applies) per bin the fused sweep needs so that
its discretisation error stays below `1e-2` Poisson sigma at the production
binning (`nbins = ndt = 800`, whole-genome `L`).  Plays for `fusedsweep!` the
role `getorder` plays for the order loop.

Calibrated in §7.3 of `notes/smcp-integrals-notes.tex` against an order-400
reference, over `hi` from `3e4` to `5e7`; the worst case at each branch is
`1.9e-3` (alpha 0.5), `2.2e-3` (alpha 0.667) and `8.5e-3` (alpha 0.8).
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
    fusedsweep!(ys, ts, dts, qs, om, Phi, dgn, Gc, Ninv, dC, A, Jc, Jf, MJ, J1,
                zs, wt, rs, edges, mu, rho, npicard, n_dt, nrs, TN)

One forward sweep in `r` over the Volterra form of the SMC' recursion, writing
the expected number of segments at `rs` into `ys`.

Solves `J = J_1 + alpha * conv(M J)` by forward substitution instead of
truncating the Neumann series at `order`, so **all** orders are resolved. Each
bin is one exponential-Euler step (stiff `-2*rate*t` term exact, `expm1` against
small-argument cancellation, only non-negative terms added), closed by `npicard`
Picard iterations because `M J` on the bin depends on `J` itself. `MJ` carries
over from the previous bin as the seed.

Unlike `prordn!` this is sequential in `r` and cannot be threaded over `nrs`.
It does not write `res`: per-order diagnostics require `prordn!`.
"""
function fusedsweep!(ys::AbstractVector{<:Real},
    ts::AbstractVector{<:Real}, dts::AbstractVector{<:Real}, qs::AbstractVector{<:Real},
    om::AbstractVector{<:Real}, Phi::AbstractVector{<:Real}, dgn::AbstractVector{<:Real},
    Gc::AbstractVector{<:Real}, Ninv::AbstractVector{<:Real}, dC::AbstractVector{<:Real},
    A::AbstractVector{<:Real}, Jc::AbstractVector{<:Real}, Jf::AbstractVector{<:Real},
    MJ::AbstractVector{<:Real}, J1::AbstractVector{<:Real},
    zs::AbstractVector{<:Real}, wt::AbstractVector{<:Real},
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

    @threads for j in 1:n_dt
        t, dt = tolegendre(zs[j], TN)
        ts[j] = t
        dts[j] = dt
        qs[j] = pt(t, TN)
        om[j] = wt[j] * dt
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
                Jc[j] = A[j] * exp(-2rate * (rs[i] - edges[i]) * t) +
                        alpha * MJ[j] * (- expm1(-2rate * wi * t)) / 2t
                Jf[j] = J1[j] + Jc[j]
            end
            transition!(MJ, Jf, Phi, dgn, Gc, Ninv, dC, om, n_dt)
        end
        s = zero(T)
        for j in 1:n_dt
            t = ts[j]
            # terminal t integral of the convolution part; 2t from p(r|t)
            s += Jc[j] * 2 * t * om[j]
            # roll the accumulator from edges[i] to edges[i+1]
            A[j] = exp(-2rate * w * t) * A[j] +
                   alpha * MJ[j] * (- expm1(-2rate * w * t)) / 2t
        end
        # order 1 comes from the analytic firstorder, as in prordn!'s res[:,1]
        ys[i] = (firstorder(rs[i], rate, TN) + s) * scale
    end
    return nothing
end
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS — the four new testsets plus every pre-existing one.

- [ ] **Step 5: Commit**

```bash
git add src/Spectra/SMCpIntegrals.jl test/smcp_fused.jl test/runtests.jl
git commit -m "add the fused Volterra sweep for the SMC' order sum

Forward substitution in r over the Volterra form of the recursion resolves
every order in one pass instead of truncating at \`order\`. Each bin is the
same exponential-Euler step the acc recursion already performs, closed by
npicard Picard iterations because M J is implicit on the bin. Not yet wired
into mldsmcp."
```

---

### Task 2: Buffers in `IntegralArrays` and the bag wrapper

**Files:**
- Modify: `src/Spectra/SMCpIntegrals.jl` (the `IntegralArrays` struct at lines 151-170, its constructor at 172-186, and append one `fusedsweep!` method)
- Modify: `test/smcp_fused.jl` (append a testset)

**Interfaces:**
- Consumes: `fusedsweep!` and `getnpicard` from Task 1
- Produces:
  - `IntegralArrays` with five new fields appended after `dC`: `A, Jc, Jf, MJ, J1`, each a `DiffCache{Vector{T},Vector{T}}` of length `n_dt`. Public constructor signature unchanged.
  - `fusedsweep!(bag::IntegralArrays, rs, edges, mu, rho, TN; npicard = 0) -> nothing` — `npicard = 0` means "choose with `getnpicard`". Writes `bag.ys`.

- [ ] **Step 1: Write the failing test**

Append to `test/smcp_fused.jl`:

```julia
@testset "bag wrapper matches the raw fusedsweep!" begin
    TN = [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0]
    mu = 1.25e-8
    ndt = 200
    edges, rs = prodgrid(200, 30_000)

    for ratio in (1.0, 4.0)
        rho = mu * ratio
        np = getnpicard(mu, rho)

        bag = IntegralArrays(10, ndt, length(rs), Val{length(TN)})
        fusedsweep!(bag, rs, edges, mu, rho, TN)
        auto = copy(get_tmp(bag.ys, eltype(TN)))
        @test auto ≈ rawfused(rs, edges, mu, rho, ndt, TN, np)

        # an explicit npicard overrides the rule
        fusedsweep!(bag, rs, edges, mu, rho, TN; npicard = 6)
        @test get_tmp(bag.ys, eltype(TN)) ≈ rawfused(rs, edges, mu, rho, ndt, TN, 6)

        # calling twice with the same arguments must give the same answer:
        # A and MJ have to be reset, not carried between calls
        fusedsweep!(bag, rs, edges, mu, rho, TN)
        @test get_tmp(bag.ys, eltype(TN)) ≈ auto
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL — `MethodError: no method matching fusedsweep!(::IntegralArrays{Float64}, ...)`

- [ ] **Step 3: Write minimal implementation**

In `src/Spectra/SMCpIntegrals.jl`, add five fields to the end of the struct so it reads:

```julia
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
    A::DiffCache{Vector{T},Vector{T}}
    Jc::DiffCache{Vector{T},Vector{T}}
    Jf::DiffCache{Vector{T},Vector{T}}
    MJ::DiffCache{Vector{T},Vector{T}}
    J1::DiffCache{Vector{T},Vector{T}}
end
```

and five more `dcvec()` calls to the constructor so it reads:

```julia
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
        dcvec(), dcvec(), dcvec(), dcvec(),
        dcvec(), dcvec(), dcvec(), dcvec(), dcvec()
    )
end
```

Then append, after the `fusedsweep!` from Task 1:

```julia
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
        get_tmp(bag.dts, T),
        get_tmp(bag.qs, T),
        get_tmp(bag.om, T),
        get_tmp(bag.Phi, T),
        get_tmp(bag.dgn, T),
        get_tmp(bag.Gc, T),
        get_tmp(bag.Ninv, T),
        get_tmp(bag.dC, T),
        get_tmp(bag.A, T),
        get_tmp(bag.Jc, T),
        get_tmp(bag.Jf, T),
        get_tmp(bag.MJ, T),
        get_tmp(bag.J1, T),
        bag.zs, bag.wt, rs, edges, mu, rho, np, bag.n_dt, bag.nrs, TN
    )
    return nothing
end
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS — all testsets, including `smcp_semiseparable.jl`, which constructs `IntegralArrays` positionally through the public constructor and is unaffected by the added fields.

- [ ] **Step 5: Commit**

```bash
git add src/Spectra/SMCpIntegrals.jl test/smcp_fused.jl
git commit -m "give IntegralArrays the fused-sweep buffers and a bag wrapper

Five extra length-ndt DiffCache vectors; the public constructor signature is
unchanged. npicard defaults to getnpicard(mu, rho)."
```

---

### Task 3: Wire into `mldsmcp` / `mldsmcp!`

**Files:**
- Modify: `src/Spectra/Spectra.jl:20-52` (docstring, `mldsmcp`, the four-argument `mldsmcp!`, and the export list at 12-18)
- Modify: `test/smcp_fused.jl` (append a testset)

**Interfaces:**
- Consumes: `fusedsweep!(bag, ...)` and `getnpicard` from Task 2
- Produces:
  - `mldsmcp(rs, edges, mu, rho, TN; order = 10, ndt = 800, method = :fused, npicard = 0)`
  - `mldsmcp!(bag, range, rs, edges, mu, rho, TN; method = :order, npicard = 0)`
  - `getnpicard` re-exported from `Spectra`

Note the asymmetric defaults: they are deliberate, see Global Constraints.

- [ ] **Step 1: Write the failing test**

Append to `test/smcp_fused.jl`:

```julia
@testset "mldsmcp method keyword" begin
    TN = [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0]
    mu = 1.25e-8
    rho = 4mu
    ndt = 200
    edges, rs = prodgrid(200, 30_000)

    # :fused agrees with the direct bag call
    got = mldsmcp(rs, edges, mu, rho, TN; ndt = ndt, method = :fused)
    bag = IntegralArrays(10, ndt, length(rs), Val{length(TN)})
    fusedsweep!(bag, rs, edges, mu, rho, TN)
    @test got ≈ get_tmp(bag.ys, eltype(TN))

    # :order reproduces the pre-change behaviour bit for bit
    order = 12
    want = mldsmcp(rs, edges, mu, rho, TN; order = order, ndt = ndt, method = :order)
    bag2 = IntegralArrays(order, ndt, length(rs), Val{length(TN)})
    SMCp.prordn!(bag2, rs, edges, mu + rho, TN)
    mldsmcp!(bag2, 1:order, mu, rho, TN)
    @test want == get_tmp(bag2.ys, eltype(TN))

    # the mutating entry still defaults to the order loop
    bag3 = IntegralArrays(order, ndt, length(rs), Val{length(TN)})
    mldsmcp!(bag3, 1:order, rs, edges, mu, rho, TN)
    @test get_tmp(bag3.ys, eltype(TN)) == want

    # res is poisoned on the fused path so stale per-order reads are loud
    bag4 = IntegralArrays(order, ndt, length(rs), Val{length(TN)})
    mldsmcp!(bag4, 1:order, rs, edges, mu, rho, TN; method = :fused)
    @test all(isnan, get_tmp(bag4.res, eltype(TN)))

    @test_throws ArgumentError mldsmcp(rs, edges, mu, rho, TN; ndt = ndt, method = :bogus)
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL — `MethodError: no method matching mldsmcp(...; ndt, method)` (unsupported keyword `method`)

- [ ] **Step 3: Write minimal implementation**

In `src/Spectra/Spectra.jl`, add `getnpicard` to the export list:

```julia
export
    firstorder, firstorderint,
	laplacekingman, laplacekingmanint,
	mldsmcp, mldsmcp!, IntegralArrays, getnpicard,
	extbps,
    lineages, cumulative_lineages, crediblehistory,
    sampleN, quantilesN
```

Replace the `mldsmcp` docstring and function (lines 20-36) with:

```julia
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
"""
function mldsmcp(rs, edges, mu, rho, TN; order = 10, ndt = 800,
	method::Symbol = :fused, npicard::Int = 0
)
	bag = IntegralArrays(order, ndt, length(rs), Val{length(TN)})
	mldsmcp!(bag, 1:order, rs, edges, mu, rho, TN; method, npicard)
	return get_tmp(bag.ys, eltype(TN))
end
```

Replace the four-argument `mldsmcp!` (lines 38-45) with:

```julia
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
```

Leave the two remaining `mldsmcp!` methods (the ones that only consume `bag.res`) exactly as they are.

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS. In particular `test/spectra.jl`'s "mld smcp runs" testset now exercises the fused path (it calls `mldsmcp` without keywords and only asserts positivity), and every `mldsmcp!` caller in `src/` still runs the order loop.

- [ ] **Step 5: Commit**

```bash
git add src/Spectra/Spectra.jl test/smcp_fused.jl
git commit -m "add the method keyword to mldsmcp and mldsmcp!

mldsmcp defaults to :fused, mldsmcp! to :order, so no caller in src/ changes
behaviour yet. On the fused path bag.res is NaN-filled: per-order diagnostics
require method = :order."
```

---

### Task 4: The accuracy gate — error in units of Poisson noise

**Files:**
- Modify: `test/smcp_fused.jl` (append a testset)

**Interfaces:**
- Consumes: everything from Tasks 1-3. Produces nothing new.

This is the task that decides whether the change is acceptable. Max-relative
error is deliberately *not* the metric: it is always attained in the extreme
tail, where the expected count is a fraction of a segment. The metric is the
deviation in Poisson standard deviations over the bins `adapt_histogram` would
keep. Thresholds come from §7.3 of the spec.

- [ ] **Step 1: Write the failing test**

Append to `test/smcp_fused.jl`:

```julia
# max |z| over bins that would survive adapt_histogram's tail threshold
function maxz(rs, edges, mu, rho, ndt, TN; reford = 200, tailthr = 10)
    ref = orderref(rs, edges, mu, rho, ndt, TN, reford)
    got = mldsmcp(rs, edges, mu, rho, TN; ndt = ndt, method = :fused)
    keep = findall(ref .> tailthr)
    @assert length(keep) > 100
    maximum(abs.(got[keep] .- ref[keep]) ./ sqrt.(ref[keep]))
end

@testset "fused error is far below Poisson noise at the production binning" begin
    TN = [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0]
    mu = 1.25e-8
    nbins = ndt = 800

    # (rho/mu, hi, npicard chosen by the rule, measured max |z| in §7.3)
    cases = [
        (1.0, 30_000,     2, 5.4e-4),
        (1.0, 10_000_000, 2, 1.9e-3),
        (2.0, 10_000_000, 3, 2.2e-3),
        (4.0, 30_000,     4, 1.5e-3),
        (4.0, 10_000_000, 4, 6.4e-3),
    ]
    for (ratio, hi, np, measured) in cases
        rho = mu * ratio
        @test getnpicard(mu, rho) == np
        edges, rs = prodgrid(nbins, hi)
        z = maxz(rs, edges, mu, rho, ndt, TN)
        @test z < 1e-2                # the design target
        @test z < 3 * measured        # guards against silent regression
    end
end

@testset "fused sweep survives an unadapted hi" begin
    # 5e7 is adapt_histogram's default hi before adaptation: bins up to
    # 1.1e6 bp wide, far wider than anything real data produces.
    TN = [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0]
    mu = 1.25e-8
    rho = 4mu
    edges, rs = prodgrid(800, 50_000_000)
    ys = mldsmcp(rs, edges, mu, rho, TN; ndt = 800, method = :fused)
    @test all(isfinite, ys)
    @test all(ys .> 0)
    @test maxz(rs, edges, mu, rho, 800, TN) < 1e-2
end

@testset "fused error does not depend on the demography" begin
    mu = 1.25e-8
    rho = 4mu
    edges, rs = prodgrid(800, 10_000_000)
    for TN in ([3.0e9, 10000.0],
               [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0],
               [3.0e9, 20000.0, 60000.0, 8000.0, 8000.0, 16000.0, 1600.0, 2000.0, 400.0, 10000.0])
        @test maxz(rs, edges, mu, rho, 800, TN) < 1e-2
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. -t auto -e 'using Pkg; Pkg.test()'`
Expected: PASS if Tasks 1-3 are correct — this testset has no new implementation
behind it, it is the acceptance measurement. Run it *before* trusting the
implementation. If any `z < 1e-2` assertion fails, the bug is in `fusedsweep!`
(most likely the `wi` vs `w` distinction, or `MJ` not being seeded/reset), not
in the thresholds. **Do not loosen the thresholds to make this pass.**

Runtime note: nine order-200 references at `nrs = ndt = 800` dominate; expect
~15 s threaded, ~60 s single-threaded. This is the slowest testset in the suite
and that is acceptable for the acceptance gate.

- [ ] **Step 3: No implementation**

This task adds no source changes. If Step 2 failed, fix `fusedsweep!` in
`src/Spectra/SMCpIntegrals.jl` and re-run.

- [ ] **Step 4: Run the full suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS, all testsets.

- [ ] **Step 5: Commit**

```bash
git add test/smcp_fused.jl
git commit -m "gate the fused sweep on error measured in Poisson noise

max |z| < 1e-2 at nbins = ndt = 800 for alpha up to 0.8 and hi up to 5e7,
across three demographies, with npicard from getnpicard. Thresholds from
section 7.3 of the notes."
```

---

### Task 5: ForwardDiff compatibility and measured speedup

**Files:**
- Modify: `test/smcp_fused.jl` (add `using ForwardDiff` to the import block at the top, append a testset)

**Interfaces:**
- Consumes: everything from Tasks 1-4. Produces nothing new.

`ForwardDiff` is already in `test/Project.toml` (added by the predecessor plan),
so no dependency change is needed.

- [ ] **Step 1: Write the failing test**

Add `using ForwardDiff` to the **top-level** import block of
`test/smcp_fused.jl` (a bare `using` is only legal at top level, never inside a
`@testset`), then append:

```julia
@testset "fused sweep is ForwardDiff-differentiable" begin
    mu = 1.25e-8
    rho = 4mu
    ndt = 60
    edges = collect(1.0:1.0:40.0)
    rs = collect(1.0:1.0:39.0)
    TN0 = [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0]

    function total(TN)
        bag = IntegralArrays(4, ndt, length(rs), Val{length(TN)})
        fusedsweep!(bag, rs, edges, mu, rho, TN)
        sum(get_tmp(bag.ys, eltype(TN)))
    end

    g = ForwardDiff.gradient(total, TN0)
    @test length(g) == length(TN0)
    @test all(isfinite, g)
    @test any(!iszero, g)

    # central differences on the population-size entries (indices 2, 4, 6)
    for k in (2, 4, 6)
        h = 1e-3 * TN0[k]
        tp = copy(TN0); tp[k] += h
        tm = copy(TN0); tm[k] -= h
        fd = (total(tp) - total(tm)) / 2h
        @test isapprox(g[k], fd; rtol = 1e-4, atol = 1e-8 * abs(g[k]))
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS if the `zero(T)`/`fill!(x, zero(T))` discipline of Task 1 was
followed. A failure here is a real AD bug — the usual cause is a literal `0.0`
accumulator, or `eltype(ys)` resolving to `Float64` when `TN` carries duals —
and must be fixed in `src/Spectra/SMCpIntegrals.jl` before continuing.

- [ ] **Step 3: Record the measured speedup**

No source changes. Run this and paste the numbers into the commit message:

```bash
julia --project=. -t 1 -e '
using IBSpector, HistogramBinnings, StatsBase
using IBSpector.Spectra, IBSpector.Spectra.PreallocationTools
const S = IBSpector.Spectra.SMCpIntegrals
TN = [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0]
mu = 1.25e-8
for ratio in (1.0, 4.0)
    rho = mu * ratio
    ord = IBSpector.getorder(2e-5, mu, rho)
    np = getnpicard(mu, rho)
    ev = IBSpector.CustomEdgeVector(lo = 1, hi = 30_000, nbins = 800)
    edges = collect(Float64, ev); rs = collect(Float64, midpoints(ev))
    bag = IntegralArrays(ord, 800, length(rs), Val{length(TN)})
    mldsmcp!(bag, 1:ord, rs, edges, mu, rho, TN)
    to = @elapsed for _ in 1:3; mldsmcp!(bag, 1:ord, rs, edges, mu, rho, TN); end
    S.fusedsweep!(bag, rs, edges, mu, rho, TN)
    tf = @elapsed for _ in 1:3; S.fusedsweep!(bag, rs, edges, mu, rho, TN); end
    println("rho/mu=", ratio, "  order=", ord, ": ", round(1e3to/3, digits=1), " ms   ",
            "fused np=", np, ": ", round(1e3tf/3, digits=1), " ms   ",
            round(to/tf, digits=1), "x")
end'
```

Expected, from §7.3 (single-threaded, `nrs = ndt = 800`, three-epoch): at
`rho/mu = 4` the order loop takes ~1590 ms and the fused sweep ~100 ms, ~16×.
At `rho/mu = 1` the gap is smaller because `getorder` returns 16 rather than 43.
What must hold is that the fused time is proportional to `npicard` and
independent of `order`.

- [ ] **Step 4: Run the full suite once more**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS, all testsets.

- [ ] **Step 5: Commit**

```bash
git add test/smcp_fused.jl
git commit -m "test ForwardDiff compatibility of the fused sweep

Gradient checked against central differences on the population-size
parameters. Benchmark at nrs = ndt = 800, single-threaded: <paste measured
numbers>."
```

---

## Out of Scope (deliberately)

- **Flipping `mldsmcp!`'s default to `:fused`.** The one-line change is
  `method::Symbol = :order` → `:fused` in `src/Spectra/Spectra.jl`. It should
  follow a comparison of an end-to-end fit (`fit_model_epochs!` on simulated
  segments) under both methods, which is a separate piece of work.
- **Migrating `mle_optimization.jl`, `corrections.jl`, `sequential_fit.jl`.**
  They keep calling `mldsmcp!` with its default, so they keep the order loop.
- **Reclaiming the `res`, `jprt` and `temp` buffers.** The fused path does not
  use them, but `IntegralArrays` still allocates them because `method = :order`
  does. Making them lazy is a follow-up worth roughly `nrs*ndt*2 + nrs*order`
  Float64 per bag.
- **A better Picard seed.** `npicard = 1` carries a 1% relative error in the
  first bin purely because the seed there is `MJ = 0` (bin 2 onwards is at
  2e-4). Seeding `MJ = M J_1(rs[i])` instead of the previous bin's value may buy
  an iteration everywhere. Untested — spec §9, item 4.
- **Sub-stepping wide bins.** Not needed at the production binning; spec §5.
- **Threading the fused sweep.** It is sequential in `r` by construction. The
  per-bin body is ~10^4 flops, too small to thread over `n_dt`. Spend threads on
  the parameter sweep instead.
- **Hand-written adjoint gradients.** Spec §11, dropped at the user's request in
  the predecessor plan.

## Self-Review Notes

- Spec coverage: §3 + §5 (the fused step, the Picard closure, the corrected
  contraction rate) → Task 1. §6 (the `w <= 1` reporting convention) → Task 1's
  `wi` computation, commented in place. §7.3 (the calibration and its
  thresholds) → `getnpicard` in Task 1 and the gate in Task 4. §8 (the go/no-go
  criterion and the α-driven rule) → `getnpicard` in Task 1, timings in Task 5.
  §9 items 1, 2, 4, 5 → addressed in Task 3 (`res` NaN-fill keeps `prordn!` as
  the diagnostic path), Task 5 (ForwardDiff), and Out of Scope (seed, threading)
  respectively. §9 item 3 (the `npicard` rule) → Task 1.
- Type consistency: the buffer names `A, Jc, Jf, MJ, J1` are identical in the
  `fusedsweep!` definition, in the `IntegralArrays` struct, in the constructor's
  positional order, and in the bag wrapper's `get_tmp` calls. `getnpicard(mu,
  rho)` takes rates, not `alpha`, everywhere it appears. `method` is a `Symbol`
  compared with `===` in both branches.
- The `IntegralArrays` field order in Task 2's struct matches the constructor's
  positional argument order exactly: 4 scalars/matrices, `zs`, `wt`, then 9 +
  5 = 14 `dcvec()` in the order `ts, dts, qs, om, Phi, dgn, Gc, Ninv, dC, A, Jc,
  Jf, MJ, J1`.
- Task 4 is the only task whose Step 2 is expected to pass rather than fail;
  this is stated explicitly there, along with the instruction not to loosen the
  thresholds.
