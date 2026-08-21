# Panel-wise Time Quadrature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the global time-quadrature map with panels pinned to the epoch boundaries, so the SMC' likelihood becomes C¹ in the parameters and LBFGS stops failing its line search.

**Architecture:** Each epoch `[T_k, T_{k+1}]` gets its own fixed-size Gauss-Legendre panel, mapped affinely from the epoch endpoints; the final epoch `[T_K, ∞)` gets a Gauss-Laguerre panel in the coalescent variable `u = (t−T_K)/2N_K` with weights folded (`w_i·e^{u_i}`). Because every node is an affine function of the epoch parameters, nodes move smoothly with `TN` and can never migrate between epochs — which is what removes the kink. Everything downstream (`sepkernel!`, `transition!`, the semiseparable structure, the r-sweep) is untouched.

**Tech Stack:** Julia 1.12, FastGaussQuadrature (`gausslegendre`, `gausslaguerre`), ForwardDiff, PreallocationTools, Test.

**Spec:** `docs/superpowers/specs/2026-08-19-panel-time-quadrature-design.md`

## Global Constraints

- `m` and `mtail` may depend on μ, ρ and the r-grid — all fixed during a fit — but **never on `TN`**. A node count that varies with `TN` reintroduces parameter-dependent discreteness, trading a kink for a jump.
- Node positions and weights must be **exactly affine in `TN`**. This is testable: along any straight line in `TN` space, every `ts[j]` must be exactly linear in the line parameter, to roundoff.
- `ts` must stay **ascending**; `sepkernel!` and `transition!` depend on it for their prefix/suffix recursions.
- All new numerical code must remain **ForwardDiff-safe** (no `Float64` type annotations on values derived from `TN`, no branches on dual values).
- The branch is `performance`. Do not commit unless the user asks.

**Correction to the spec:** the spec says three `IntegralArrays` construction sites need `K` threaded in. There are **five** in `src/`: `Spectra/Spectra.jl:42`, `IBSpector.jl:66`, `mle_optimization.jl:163`, `mle_optimization.jl:297`, `corrections.jl:147`, `corrections.jl:250` (six lines, five call contexts). Task 5 covers all of them.

---

### Task 1: `TimeGrid` — the TN-independent reference data

**Files:**
- Modify: `src/Spectra/SMCpIntegrals.jl` (add after the `transition!` definitions, around line 150; extend the `export` list at line 10)
- Test: `test/time_quadrature.jl` (create), `test/runtests.jl` (add include)

**Interfaces:**
- Consumes: nothing.
- Produces: `TimeGrid(K::Int; m::Int = 48, mtail::Int = 48)` returning a `TimeGrid` with fields `m::Int`, `mtail::Int`, `K::Int`, `zleg::Vector{Float64}`, `wleg::Vector{Float64}`, `ulag::Vector{Float64}`, `wlag::Vector{Float64}`; and `ndt(g::TimeGrid)::Int`.

- [ ] **Step 1: Write the failing test**

Create `test/time_quadrature.jl`:

```julia
using IBSpector
using IBSpector.Spectra
using Test

const SMCp = IBSpector.Spectra.SMCpIntegrals
using IBSpector.Spectra.SMCpIntegrals: TimeGrid, ndt, timenodes!

@testset "TimeGrid" begin
    g = TimeGrid(5; m = 48, mtail = 32)
    @test g.K == 5 && g.m == 48 && g.mtail == 32
    @test ndt(g) == 4 * 48 + 32
    # a one-epoch history has no finite panels, only the tail
    @test ndt(TimeGrid(1; m = 48, mtail = 32)) == 32

    # Gauss-Legendre weights on (-1,1) sum to the interval length
    @test sum(g.wleg) ≈ 2 rtol = 1e-12
    @test length(g.zleg) == 48 && all(-1 .< g.zleg .< 1)

    # The Laguerre weights are stored FOLDED (w_i * exp(u_i)), so recovering
    # int_0^inf e^{-u} u^p du = p! requires multiplying the integrand by e^{-u}
    # exactly as `pt` does inside the sweep.
    for p in 0:5
        @test sum(g.wlag .* g.ulag .^ p .* exp.(-g.ulag)) ≈ factorial(p) rtol = 1e-9
    end
    # folded weights must be well conditioned, not e^{+u}-huge
    @test maximum(g.wlag) < 1e3
    @test minimum(g.wlag) > 1e-3
end
```

Add to `test/runtests.jl` after line 16:

```julia
include("time_quadrature.jl")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'` (or `julia --project=. test/time_quadrature.jl`)
Expected: FAIL — `UndefVarError: TimeGrid not defined in IBSpector.Spectra.SMCpIntegrals`

- [ ] **Step 3: Write minimal implementation**

In `src/Spectra/SMCpIntegrals.jl`, extend the export list at line 10:

```julia
export IntegralArrays, prordn!, fusedsweep!, getnpicard,
    firstorder, firstorderint, TimeGrid, timenodes!, ndt
```

Insert before `struct IntegralArrays` (line 151):

```julia
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS — the `TimeGrid` testset green, all pre-existing testsets still green (nothing else changed yet).

- [ ] **Step 5: Commit**

```bash
git add src/Spectra/SMCpIntegrals.jl test/time_quadrature.jl test/runtests.jl
git commit -m "add TimeGrid: TN-independent nodes for panel-wise time quadrature"
```

---

### Task 2: `timenodes!` — building the panels from `TN`

**Files:**
- Modify: `src/Spectra/SMCpIntegrals.jl` (add directly after `ndt`)
- Test: `test/time_quadrature.jl`

**Interfaces:**
- Consumes: `TimeGrid`, `ndt` from Task 1; `getts`, `getns` from `CoalescentBase` (already in scope via `using ..CoalescentBase`).
- Produces: `timenodes!(ts::AbstractVector{<:Real}, om::AbstractVector{<:Real}, g::TimeGrid, TN::AbstractVector{<:Real})::Nothing`, filling `ts` with nodes and `om` with weights, both of length `ndt(g)`.

- [ ] **Step 1: Write the failing test**

Append to `test/time_quadrature.jl`:

```julia
# the real 5-epoch history where the production fit stalls; N3 = 9.99e7 sits at
# the upper bound, giving the worst adjacent-N ratio the box permits (~14300)
const TNSTALL = [3.003e9, 12388.8, 28302.1, 6975.85, 6214.37,
                 9.99002e7, 3066.44, 2754.27, 215.101, 21782.5]

@testset "timenodes! tiles the epochs" begin
    TN = TNSTALL
    K = length(TN) ÷ 2
    g = TimeGrid(K; m = 16, mtail = 16)
    n = ndt(g)
    ts = zeros(n); om = zeros(n)
    timenodes!(ts, om, g, TN)

    @test issorted(ts)
    @test all(ts .> 0)
    @test all(isfinite, ts) && all(isfinite, om)

    # every finite panel's nodes lie strictly inside its own epoch
    for k in 1:K-1
        lo = Spectra.getts(TN, k); hi = Spectra.getts(TN, k+1)
        blk = ts[(k-1)*g.m+1 : k*g.m]
        @test all(lo .< blk .< hi)
        # its weights sum to the epoch width
        @test sum(om[(k-1)*g.m+1 : k*g.m]) ≈ hi - lo rtol = 1e-12
    end
    # tail nodes are past the last epoch time
    @test all(ts[(K-1)*g.m+1 : end] .> Spectra.getts(TN, K))
end

@testset "timenodes! integrates the tail exactly" begin
    # single epoch: the whole domain is the Laguerre panel, and
    # int_0^inf exp(-t/2N) dt = 2N  is reproduced by the folded weights
    N = 12345.0
    TN = [3.0e9, N]
    g = TimeGrid(1; m = 8, mtail = 32)
    n = ndt(g)
    ts = zeros(n); om = zeros(n)
    timenodes!(ts, om, g, TN)
    @test sum(om .* exp.(-ts ./ (2N))) ≈ 2N rtol = 1e-10
    # and a polynomial-times-exponential moment
    @test sum(om .* ts .* exp.(-ts ./ (2N))) ≈ (2N)^2 rtol = 1e-10
end

@testset "timenodes! is exactly affine in TN" begin
    # THE invariant that the old tolegendre/tolaguerre map violated: there, a node
    # crossing an epoch boundary kept t continuous but changed dt/dz by the ratio
    # of adjacent N (14300x at this TN), producing a kink in the likelihood.
    # Here every node is affine in the epoch parameters, so along any straight
    # line in TN space each ts[j] must be exactly linear, to roundoff.
    TN0 = TNSTALL
    K = length(TN0) ÷ 2
    g = TimeGrid(K; m = 48, mtail = 48)
    n = ndt(g)
    for idx in (3, 5, 7, 9, 2, 10)          # durations and sizes alike
        d = zeros(length(TN0)); d[idx] = TN0[idx]   # relative perturbation
        as = range(-1e-3, 1e-3, length = 51)
        T = zeros(length(as), n)
        ts = zeros(n); om = zeros(n)
        for (q, a) in enumerate(as)
            timenodes!(ts, om, g, TN0 .+ a .* d)
            T[q, :] .= ts
        end
        for j in 1:n
            lo, hi = T[1, j], T[end, j]
            for q in 1:length(as)
                lin = lo + (hi - lo) * (q - 1) / (length(as) - 1)
                @test T[q, j] ≈ lin atol = 1e-9 * max(abs(lo), abs(hi)) + 1e-12
            end
        end
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL — `UndefVarError: timenodes! not defined`

- [ ] **Step 3: Write minimal implementation**

In `src/Spectra/SMCpIntegrals.jl`, directly after `ndt(g::TimeGrid)`:

```julia
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS — all three new testsets green.

If "exactly affine" fails, the cause is a non-affine expression in the map: check that nothing recomputes `getts` from a node value and that no `min`/`max`/comparison on `TN` crept in.

- [ ] **Step 5: Commit**

```bash
git add src/Spectra/SMCpIntegrals.jl test/time_quadrature.jl
git commit -m "add timenodes!: epoch-pinned panels, exactly affine in TN"
```

---

### Task 3: Wire the panels into `prordn!` and `fusedsweep!`

**Files:**
- Modify: `src/Spectra/SMCpIntegrals.jl` — delete `tolaguerre` (130-143) and `tolegendre` (144-149); `IntegralArrays` struct (151-173) and constructor (175-190); `prordn!` bag method (192-214) and kernel (216-236); `fusedsweep!` kernel (369-396) and bag method (445-472)
- Modify: `test/smcp_semiseparable.jl:3,26`, `test/smcp_fused.jl:36-41,58`
- Test: `test/time_quadrature.jl`

**Interfaces:**
- Consumes: `TimeGrid`, `ndt`, `timenodes!` from Tasks 1-2.
- Produces: `IntegralArrays(order::Int, grid::TimeGrid, nrs::Int, chunk, levels = 1)`; field `grid::TimeGrid` replaces `zs`/`wt`; `n_dt` stays a field, set to `ndt(grid)`. Kernel signatures replace the `zs, wt` argument pair with a single `grid::TimeGrid`.

- [ ] **Step 1: Write the failing test**

Append to `test/time_quadrature.jl`:

```julia
@testset "sweep runs on the panel grid and matches the order loop" begin
    using IBSpector.Spectra.PreallocationTools
    mu, rho = 1.0e-8, 2.0e-8
    TN = TNSTALL
    K = length(TN) ÷ 2
    ev = IBSpector.CustomEdgeVector(lo = 1, hi = 30_000, nbins = 120)
    edges = collect(Float64, ev); rs = collect(Float64, IBSpector.midpoints(ev))
    grid = TimeGrid(K; m = 64, mtail = 64)

    bagf = IntegralArrays(60, grid, length(rs), Val{length(TN)})
    bago = IntegralArrays(60, grid, length(rs), Val{length(TN)})
    @test bagf.n_dt == ndt(grid)

    SMCp.fusedsweep!(bagf, rs, edges, mu, rho, TN; npicard = 6)
    yf = copy(get_tmp(bagf.ys, Float64))
    SMCp.prordn!(bago, rs, edges, mu + rho, TN)
    Spectra.mldsmcp!(bago, 1:60, mu, rho, TN)
    yo = copy(get_tmp(bago.ys, Float64))

    @test all(isfinite, yf) && all(yf .> 0)
    # converged Picard vs converged order loop: same integral, same grid
    @test maximum(abs.(yf .- yo) ./ yo) < 1e-6
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL — `MethodError: no method matching IntegralArrays(::Int64, ::TimeGrid, ::Int64, ::Type{Val{10}})`

- [ ] **Step 3: Write minimal implementation**

**3a.** Delete lines 130-149 of `src/Spectra/SMCpIntegrals.jl` — the whole `tolaguerre` and `tolegendre` definitions.

**3b.** In `struct IntegralArrays`, replace the two lines

```julia
    zs::Vector{Float64}
    wt::Vector{Float64}
```

with

```julia
    grid::TimeGrid
```

**3c.** Replace the constructor (starting line 175) with:

```julia
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
```

Note the field order must match the struct: `order, n_dt, nrs, ys, res, jprt, temp, grid, ts, qs, om, Phi, dgn, Gc, Ninv, dC, A, Jf, MJ, J1`.

**3d.** In the `prordn!` bag method, replace the two arguments

```julia
        bag.zs,
        bag.wt,
```

with

```julia
        bag.grid,
```

**3e.** In the `prordn!` kernel signature, replace

```julia
    zs::AbstractVector{<:Real}, wt::AbstractVector{<:Real},
```

with

```julia
    grid::TimeGrid,
```

and replace the node-setup block

```julia
    @threads for j in 1:n_dt
        t, dt = tolegendre(zs[j], TN)
        ts[j] = t
        qs[j] = pt(t, TN)
        om[j] = wt[j] * dt
    end
```

with

```julia
    timenodes!(ts, om, grid, TN)
    @threads for j in 1:n_dt
        qs[j] = pt(ts[j], TN)
    end
```

**3f.** In the `fusedsweep!` kernel signature, replace

```julia
    zs::AbstractVector{<:Real}, wt::AbstractVector{<:Real},
```

with

```julia
    grid::TimeGrid,
```

and replace the node-setup block

```julia
    for j in 1:n_dt
        t, dt = tolegendre(zs[j], TN)
        ts[j] = t
        qs[j] = pt(t, TN)
        om[j] = wt[j] * dt
    end
```

with

```julia
    timenodes!(ts, om, grid, TN)
    for j in 1:n_dt
        qs[j] = pt(ts[j], TN)
    end
```

**3g.** In the `fusedsweep!` bag method, replace `bag.zs, bag.wt,` with `bag.grid,`.

**3h.** Fix the two tests that reached for the deleted functions.

`test/smcp_semiseparable.jl:3` — drop `tolegendre` from the import list:

```julia
using IBSpector.Spectra.SMCpIntegrals: sepkernel!, transition!, ptt, TimeGrid, ndt, timenodes!
```

`test/smcp_semiseparable.jl` around line 26, replace the per-node loop

```julia
        ts[j], dts[j] = tolegendre(zs[j], TN)
```

and its surrounding `for j` with a single call. The block becomes:

```julia
        g = TimeGrid(length(TN) ÷ 2; m = 32, mtail = 32)
        n = ndt(g)
        ts = zeros(n); dts = zeros(n)
        timenodes!(ts, dts, g, TN)
```

(`dts` was the weight vector; `timenodes!` fills it directly, so any later `wt[j] *` scaling must be dropped — read the surrounding lines and remove the now double-counted weight factor.)

`test/smcp_fused.jl` — `rawfused` at lines 36-41 builds `zs, wt` by hand. Replace with:

```julia
function rawfused(rs, edges, mu, rho, grid, TN, npicard)
    nrs = length(rs)
    n = SMCp.ndt(grid)
    v() = zeros(Float64, n)
    ys = zeros(Float64, nrs)
    fusedsweep!(ys, v(), v(), v(), v(), v(), v(), v(), v(),
                v(), v(), v(), v(),
                grid, rs, edges, mu, rho, npicard, n, nrs, TN)
    ys
end
```

and at line 58 replace the `SMCp.tolegendre(zs[j], TN)` loop the same way as in `smcp_semiseparable.jl`. Update every `rawfused(..., ndt, ...)` call to pass a `TimeGrid`.

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS for the new testset. Pre-existing testsets in `spectra.jl`, `smcp_semiseparable.jl`, `smcp_fused.jl` that assert *numerical values against the old grid* will shift — accuracy tolerances stated in Poisson-σ terms should still hold; hard-coded reference numbers tied to `ndt = 800` must be regenerated and the regeneration noted in the commit message.

- [ ] **Step 5: Commit**

```bash
git add src/Spectra/SMCpIntegrals.jl test/
git commit -m "use epoch-pinned panels in prordn! and fusedsweep!; drop tolegendre/tolaguerre"
```

---

### Task 3B: Convert the tail from Gauss-Laguerre to the algebraic map

**Added 2026-08-20** after Task 4 returned BLOCKED. Gauss-Laguerre was measured and
rejected: its nodes are spaced by the coalescent rate and miss the recombination scale
near T_K (relative error 1.00 at r=3e6 for K=1), and its folded weights overflow to
`Inf`/`NaN` above `mtail ≈ 200`, which is below the size the accuracy target needs.
See the spec's "Tail rule (amended 2026-08-20)" section.

**Files:**
- Modify: `src/Spectra/SMCpIntegrals.jl` — `TimeGrid` struct + constructor, `timenodes!` tail branch
- Test: `test/time_quadrature.jl`

**Interfaces:**
- Consumes: `TimeGrid`, `ndt`, `timenodes!` as they stand after Task 3.
- Produces: `TimeGrid` fields `ulag`/`wlag` replaced by `utail`/`wtail`; defaults become
  `m = 64`, `mtail = 384`. `ndt` and `timenodes!`'s signature are unchanged.

- [ ] **Step 1: Update the tail tests**

In `test/time_quadrature.jl`, replace the folded-Laguerre assertions in the `TimeGrid`
testset with the algebraic-map equivalents:

```julia
    # tail rule: u = (1+z)/(1-z) with weight w*2/(1-z)^2, so it integrates
    # int_0^inf e^{-u} u^p du = p! directly (no exponential folding)
    for p in 0:3
        @test sum(g.wtail .* g.utail .^ p .* exp.(-g.utail)) ≈ factorial(p) rtol = 1e-9
    end
    @test all(isfinite, g.wtail) && all(g.wtail .> 0)
    @test issorted(g.utail) && g.utail[1] > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `julia --project=. test/time_quadrature.jl`
Expected: FAIL — `type TimeGrid has no field wtail`

- [ ] **Step 3: Implement**

In `TimeGrid`, replace the two Laguerre field declarations with:

```julia
    utail::Vector{Float64}
    wtail::Vector{Float64}
```

and the constructor body's Laguerre lines with:

```julia
function TimeGrid(K::Int; m::Int = 64, mtail::Int = 384)
    @assert K >= 1 "need at least one epoch"
    @assert m >= 2 "need at least 2 nodes per panel"
    @assert mtail >= 2 "need at least 2 nodes in the tail panel"
    z, w = gausslegendre(m)
    zt, wt = gausslegendre(mtail)
    u  = (1 .+ zt) ./ (1 .- zt)          # algebraic map (-1,1) -> [0,inf)
    du = 2 ./ (1 .- zt) .^ 2             # du/dz
    return TimeGrid(m, mtail, K, z, w, u, wt .* du)
end
```

In `timenodes!`, replace the tail loop's two assignment lines with:

```julia
        ts[j] = TK + twoNK * g.utail[i]
        om[j] = g.wtail[i] * twoNK
```

The loop shape is otherwise unchanged — `utail`/`wtail` are precomputed constants, so
`ts` stays affine in `(T_K, N_K)` and `om` affine in `N_K`.

Remove `gausslaguerre` from the imports if nothing else uses it.

- [ ] **Step 4: Verify**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS, and in particular the "timenodes! is exactly affine in TN" testset must
still pass unchanged — the algebraic tail is exactly affine (measured deviation 2.2e-16).
If that testset fails, the tail branch is wrong; do not weaken it.

Note the "timenodes! integrates the tail exactly" testset asserts
`sum(om .* exp.(-ts./(2N))) ≈ 2N` for K=1; that identity holds for the algebraic rule
too and must keep passing.

- [ ] **Step 5: Commit**

```bash
git add src/Spectra/SMCpIntegrals.jl test/time_quadrature.jl
git commit -m "replace the Gauss-Laguerre tail with the algebraic map"
```

---

### Task 4: Calibrate `m` and `mtail` against `firstorder`

**Files:**
- Test: `test/time_quadrature.jl`

**Interfaces:**
- Consumes: `TimeGrid`, `timenodes!`, `firstorder`, `pt` from earlier tasks.
- Produces: the chosen default constants for `TimeGrid`'s `m` and `mtail` keywords (edit them in Task 1's constructor if calibration says the 48/48 guess is wrong), plus a permanent regression test.

`firstorder(r, rate, TN)` is the closed form of the order-1 terminal integral, so it scores the quadrature against **truth** at any K and any TN, with no reference implementation. It exercises the hard part — the `exp(-2·rate·r·t)` decay, whose rate spans ~3 decades across the r grid.

It does *not* exercise the transition operator, so it is necessary but not sufficient; the Task 3 test (converged Picard vs converged order loop) covers internal consistency at the transition level.

- [ ] **Step 1: Write the failing test**

Append to `test/time_quadrature.jl`:

```julia
# quadrature estimate of the order-1 terminal integral, whose exact value is
# firstorder(r, rate, TN)
function firstorder_quad(r, mu, rho, grid, TN)
    rate = mu + rho
    n = ndt(grid)
    ts = zeros(n); om = zeros(n)
    timenodes!(ts, om, grid, TN)
    s = 0.0
    for j in 1:n
        s += rate * exp(-2rate * r * ts[j]) * SMCp.pt(ts[j], TN) * 2 * ts[j] * om[j]
    end
    s
end

@testset "quadrature matches the analytic firstorder" begin
    mu, rho = 1.0e-8, 2.0e-8
    rate = mu + rho
    ev = IBSpector.CustomEdgeVector(lo = 1, hi = 3_000_000, nbins = 200)
    rs = collect(Float64, IBSpector.midpoints(ev))

    histories = Dict(
        "stationary"          => [3.0e9, 20000.0],
        "real 5-epoch"        => TNSTALL,
        "N at upper bound"    => [3.0e9, 12000.0, 5000.0, 1.0e8, 3000.0, 7000.0],
        "T at lower floor"    => [3.0e9, 12000.0, 10.0, 3000.0, 10.0, 20000.0],
        "near-empty epoch"    => [3.0e9, 15000.0, 6000.0, 9.9e7, 4000.0, 8000.0],
    )
    for (name, TN) in histories
        K = length(TN) ÷ 2
        g = TimeGrid(K)                       # the shipped defaults
        err = maximum(abs(firstorder_quad(r, mu, rho, g, TN) -
                          firstorder(r, rate, TN)) / firstorder(r, rate, TN)
                      for r in rs)
        @test err < 1e-6
    end
end

@testset "firstorder error converges geometrically in m" begin
    mu, rho = 1.0e-8, 2.0e-8
    rate = mu + rho
    TN = TNSTALL
    K = length(TN) ÷ 2
    rs = [1.0, 100.0, 10_000.0, 1_000_000.0]
    # hold the tail fixed and generous so this measures the FINITE panels only
    errs = map((8, 16, 32)) do m
        g = TimeGrid(K; m = m, mtail = 768)
        maximum(abs(firstorder_quad(r, mu, rho, g, TN) -
                    firstorder(r, rate, TN)) / firstorder(r, rate, TN) for r in rs)
    end
    # Each doubling of m must gain at least an order of magnitude UNTIL the error
    # reaches the double-precision floor, after which no further gain is possible.
    # Measured for this history: 4.8e-2, 1.7e-5, 6.8e-12 (floor). Asserting a
    # ratio without the floor guard is unsatisfiable — that was a defect in the
    # original plan, found during execution on 2026-08-20.
    const FLOOR = 1e-10
    @test errs[2] < errs[1] / 10 || errs[2] < FLOOR
    @test errs[3] < errs[2] / 10 || errs[3] < FLOOR
end
```

- [ ] **Step 2: Run test to verify it fails (or reveals the right constants)**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: either PASS (48/48 is adequate) or FAIL on a specific history with a reported `err`.

- [ ] **Step 3: Choose the constants**

Run this sweep and read off the smallest `(m, mtail)` meeting `err < 1e-6` on every history:

```bash
julia --project=. -e '
using IBSpector, IBSpector.Spectra, Printf
include("test/time_quadrature.jl")
mu, rho = 1.0e-8, 2.0e-8; rate = mu+rho
ev = IBSpector.CustomEdgeVector(lo=1, hi=3_000_000, nbins=200)
rs = collect(Float64, IBSpector.midpoints(ev))
for m in (16,32,48,64,96), mt in (96,192,384,768)
    worst = 0.0
    for TN in ([3.0e9,20000.0], TNSTALL, [3.0e9,12000.0,5000.0,1.0e8,3000.0,7000.0])
        g = TimeGrid(length(TN)÷2; m=m, mtail=mt)
        e = maximum(abs(firstorder_quad(r,mu,rho,g,TN)-firstorder(r,rate,TN))/firstorder(r,rate,TN) for r in rs)
        worst = max(worst, e)
    end
    @printf("m=%-4d mtail=%-4d worst rel err=%.3e\n", m, mt, worst)
end'
```

If the `m = 64`, `mtail = 384` defaults set in Task 3B do not clear 1e-6, edit the `TimeGrid` keyword defaults in `src/Spectra/SMCpIntegrals.jl` to the smallest pair that does. Controller measurements on 2026-08-20 say 64/384 should clear it on every history in the list; `mtail` is set by the K=1 case, where the tail is the whole domain, and K=5 alone would need only ~144.

If **no** pair up to `m = 96`, `mtail = 768` clears it, stop and report rather than inflating further. Do not loosen the 1e-6 bound.

**Expected to fail, and not your problem:** a *stationary* history with N ≳ 1e6 cannot resolve the recombination scale at the largest r under any affordable node count — 7.2e-1 at N=1e6, 1.00 at N=1e8. The original map gives the same numbers at matched cost, so it is pre-existing, not a regression. The calibration history list deliberately uses `stationary N=2e4`, which does clear.

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS — both calibration testsets green with the chosen defaults.

- [ ] **Step 5: Commit**

```bash
git add src/Spectra/SMCpIntegrals.jl test/time_quadrature.jl
git commit -m "calibrate panel node counts against the analytic firstorder"
```

---

### Task 5: Thread `mpanel`/`mtail` through `FitOptions` and the call sites

**Files:**
- Modify: `src/utils.jl:233` (field), `:268` (docstring), `:280` (kwarg), `:293-297` (default), `:332` (constructor arg)
- Modify: `src/Spectra/Spectra.jl:21,28-29,39,42`
- Modify: `src/IBSpector.jl:57,60,66`
- Modify: `src/mle_optimization.jl:163,297`
- Modify: `src/corrections.jl:147,250`
- Test: `test/runtests.jl:30-58` (the FitOptions testset), `test/time_quadrature.jl`

**Interfaces:**
- Consumes: `TimeGrid` from Task 1, the `IntegralArrays(order, grid, nrs, chunk, levels)` signature from Task 3.
- Produces: `FitOptions` fields `mpanel::Int`, `mtail::Int` replacing `ndt::Int`; keyword arguments `mpanel::Int = 0`, `mtail::Int = 0` on the `FitOptions` constructor (0 meaning "use the `TimeGrid` defaults").

`ndt` is **renamed**, not redefined, because its meaning changes from total nodes to nodes per panel: a caller passing `ndt = 800` would silently get 800 nodes per epoch.

- [ ] **Step 1: Write the failing test**

Append to `test/time_quadrature.jl`:

```julia
@testset "FitOptions carries per-panel node counts" begin
    fop = FitOptions(3.0e9, 100_000, 1.0e-8, 2.0e-8; nepochs = 5)
    @test fop.mpanel > 0 && fop.mtail > 0
    @test !hasproperty(fop, :ndt)   # renamed, so a stale `ndt = 800` cannot pass silently
    g = TimeGrid(fop.nepochs; m = fop.mpanel, mtail = fop.mtail)
    @test ndt(g) == (fop.nepochs - 1) * fop.mpanel + fop.mtail
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL — `type FitOptions has no field mpanel`

- [ ] **Step 3: Write minimal implementation**

**5a.** `src/utils.jl:233` — replace `ndt::Int` with:

```julia
    mpanel::Int
    mtail::Int
```

**5b.** `src/utils.jl:268` — replace the `ndt` docstring line with:

```julia
- `mpanel::Int=0`: Gauss-Legendre nodes per epoch panel in the time quadrature.
  When zero, the `TimeGrid` default is used.
- `mtail::Int=0`: Gauss-Laguerre nodes on the final semi-infinite panel.
  When zero, the `TimeGrid` default is used.
```

**5c.** `src/utils.jl:280` — replace `ndt::Int = 0,` with:

```julia
    mpanel::Int = 0,
    mtail::Int = 0,
```

**5d.** `src/utils.jl:293-297` — replace the `nhet`-dependent block

```julia
    if iszero(ndt)
        if nhet > 1e7
            ndt = 1600
        else
            ndt = 800
        end
    end
```

with (the panel scheme's accuracy no longer depends on total sample size, so the `nhet` branch goes away):

```julia
    dflt = TimeGrid(1)
    iszero(mpanel) && (mpanel = dflt.m)
    iszero(mtail)  && (mtail  = dflt.mtail)
```

**5e.** `src/utils.jl:332` — replace `ndt,` with `mpanel,` and `mtail,` in the positional constructor call, keeping the struct field order.

**5f.** `src/Spectra/Spectra.jl` — change the `mldsmcp` signature at line 39 and its call at 42:

```julia
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
```

Update the docstring at lines 21 and 28-29 to describe `mpanel`/`mtail` instead of `ndt`.

**5g.** `src/IBSpector.jl:57,60,66` — same substitution in `compute_residuals`; `K = length(TN) ÷ 2`:

```julia
        bag = IntegralArrays(order, TimeGrid(length(TN) ÷ 2; m = mpanel, mtail = mtail),
                             length(rs), Val{length(TN)})
```

with `mpanel = 0, mtail = 0` added to the keyword list and resolved to the `TimeGrid` defaults as above.

**5h.** `src/mle_optimization.jl:163` and `:297` — both read `Val{length(options.init)}`, so `K = options.nepochs`:

```julia
    dc = IntegralArrays(options.order,
                        TimeGrid(options.nepochs; m = options.mpanel, mtail = options.mtail),
                        length(rs), Val{length(options.init)}, 3)
```

**5i.** `src/corrections.jl:147` — `K = epochs` there:

```julia
    bag = IntegralArrays(fop.order, TimeGrid(epochs; m = fop.mpanel, mtail = fop.mtail),
                         length(rs_th), Val{2epochs})
```

**5j.** `src/corrections.jl:250` — `K = length(fit.para) ÷ 2`:

```julia
    bag = IntegralArrays(fop.order,
                         TimeGrid(length(fit.para) ÷ 2; m = fop.mpanel, mtail = fop.mtail),
                         length(rs), Val{length(fit.para)}, 3)
```

**5k.** `src/corrections.jl:58` uses `nbins::Int = fop.ndt` and `th_discr::Int = fop.ndt` as *histogram* bin counts — unrelated to the time quadrature, and they must not silently shrink to a per-panel count. Replace both defaults with the literal `800` and note in the docstring that they are histogram discretisation, not quadrature nodes.

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS. Grep for stragglers first:

```bash
grep -rn "\.ndt\|ndt =\|ndt=" src/ test/
```

Expected: no hits referring to the time quadrature.

- [ ] **Step 5: Commit**

```bash
git add src/ test/
git commit -m "rename FitOptions.ndt to mpanel/mtail and thread the TimeGrid through all call sites"
```

---

### Task 6: Acceptance — smoothness and the real fit

**Files:**
- Test: `test/time_quadrature.jl`
- Uses (not committed): `/project/minus3-simulation-data/temp-results/{fop,segments.csv}` on `secondchoice`

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: no new API. This task's deliverable is evidence.

- [ ] **Step 1: Write the failing test**

Append to `test/time_quadrature.jl`:

```julia
@testset "likelihood is smooth along a line" begin
    using IBSpector.Spectra.PreallocationTools
    using ForwardDiff, LinearAlgebra, Random, Distributions, Statistics
    mu, rho = 1.0e-8, 2.0e-8
    TN = TNSTALL
    K = length(TN) ÷ 2
    ev = IBSpector.CustomEdgeVector(lo = 1, hi = 3_000_000, nbins = 200)
    edges = collect(Float64, ev); rs = collect(Float64, IBSpector.midpoints(ev))

    # Poisson counts from the model itself, so the surface has a real optimum
    Random.seed!(20260819)
    grid = TimeGrid(K)
    bag = IntegralArrays(60, grid, length(rs), Val{K * 2}, 3)
    SMCp.fusedsweep!(bag, rs, edges, mu, rho, TN)
    w0 = get_tmp(bag.ys, Float64) .* diff(edges)
    counts = [rand(Poisson(max(w, 0.0))) for w in w0]

    function f(v)
        SMCp.fusedsweep!(bag, rs, edges, mu, rho, v)
        w = get_tmp(bag.ys, eltype(v)) .* diff(edges)
        s = zero(eltype(v))
        for i in eachindex(counts)
            (!(w[i] > 0) || isnan(w[i])) && continue
            s += counts[i] * log(w[i]) - w[i]
        end
        s
    end

    Random.seed!(3)
    for trial in 1:3
        d = normalize(randn(length(TN)) .* TN)
        # detrended residual must sit at the float floor at EVERY window size;
        # the old global map climbed like h (1e-8 at h=3e-7 -> 0.42 at h=1e-5)
        for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
            n = 41
            ss = collect(range(-h, h, length = n))
            fs = [f(TN .+ a .* d) for a in ss]
            A = hcat(ones(n), ss, ss .^ 2, ss .^ 3)
            resid = maximum(abs, fs .- A * (A \ fs))
            @test resid < 1e-6 * max(1.0, maximum(abs, fs))
        end
    end
end
```

- [ ] **Step 2: Run test to verify it fails on the OLD code**

This is the regression guard. Confirm it would have caught the bug by checking out the pre-Task-3 tree and running it:

```bash
git stash && git log --oneline -8
```

Expected on old code: FAIL at `h = 1e-5` and `h = 1e-4`. Restore with `git stash pop`.

- [ ] **Step 3: Add the other two acceptance checks from the spec's table**

Append to the same testset, reusing `f` and `d` from Step 1:

```julia
    # (a) derivative-jump ratio: was 5.85e5 with the old map, must be O(1)
    n = 241; hw = 1e-4
    as = collect(range(-hw, hw, length = n))
    ps = [f(TN .+ a .* d) for a in as]
    slope = [(ps[i+1] - ps[i]) / (as[i+1] - as[i]) for i in 1:n-1]
    d2 = [abs(slope[i+1] - slope[i]) for i in 1:n-2]
    @test maximum(d2) < 50 * median(d2)

    # (b) central FD must agree with AD at EVERY h, not only below 1e-7
    g = ForwardDiff.gradient(f, TN)
    gd = dot(g, d)
    for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
        fd = (f(TN .+ h .* d) - f(TN .- h .* d)) / (2h)
        @test abs(fd - gd) < 1e-3 * abs(gd)
    end
```

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS at every window size and every `h`.

- [ ] **Step 4: End-to-end on the stored production case**

The 13.9 GB `fitresult` is not needed and must not be deserialized — the login node has 15 GB of RAM. Copy only `fop` (1 KB) and `segments.csv` (4 MB).

```bash
scp secondchoice:/project/minus3-simulation-data/temp-results/{fop,segments.csv} /tmp/
julia --project=. -e '
using Serialization, IBSpector, HistogramBinnings, StatsBase, Printf
fop = deserialize("/tmp/fop")
segs = [parse(Int, l) for l in Iterators.drop(eachline("/tmp/segments.csv"), 1)]
h = adapt_histogram(segs)
IBSpector.setnaive!(fop, false)
fop.opt = (; maxiters = 30000, maxtime = 1800, g_tol = 5e-8)
r = IBSpector.fit_model_epochs!(fop, h; stats = false)
println("converged = ", r.converged, "   lp = ", r.lp)
println(r.opt.optim_result.original)'
```

Expected: `Status: success` with `|g(x)| <= 5.0e-08`, against the baseline `Status: failure (line search failed)` at `|g| = 1.91e+03`.

Note the deserialized `fop` predates Task 5 and carries an `ndt` field. If deserialization errors on the renamed struct, rebuild the options by hand from the recorded values: `nepochs = 5, mu = 1e-8, rho = 2e-8, Ltot = 3e9, order = 25, locut = 1`, `init = [2.999320961530274e9, 19270.62422663158, 35781.92532637016, 6316.9862758435565, 5492.1311850599905, 9.99e7, 3697.0683405275686, 3300.8938105169723, 985.1399028974942, 28879.231074189112]`.

- [ ] **Step 5: MLE stability**

Refit with `mpanel` and `mtail` doubled. Every parameter must move <1%, against the 17% shift in T₂ measured between the old `ndt = 800` and `ndt = 3200`.

- [ ] **Step 6: Commit**

```bash
git add test/time_quadrature.jl
git commit -m "add smoothness acceptance test for the panel quadrature"
```

---

## Self-Review

**Spec coverage.** Panels and affine maps → Tasks 2-3. Folded Laguerre tail → Tasks 1-2. Node budget and the TN-independence constraint → Global Constraints + Task 4. `TimeGrid`/`timenodes!`/deletions/unchanged pieces → Tasks 1-3. API ripple → Task 5 (with the three-vs-five call-site correction recorded above). Validation §1 analytic anchor → Task 4; §2 smoothness → Task 6; §3 node-map unit test → Task 2; §4 epoch structures → Task 4 histories; §5 end-to-end → Task 6; §6 regression → Task 3 step 3h and Task 5 step 4.

**Known gap carried from the spec.** Nothing here proves the panel scheme beats the old one at fixed cost for the *transition-level* integrals — `firstorder` cannot see them, and the Task 3 Picard-vs-order check only shows internal consistency. The MLE-stability check in Task 6 step 5 is the real evidence, and it is comparative rather than absolute.

**Type consistency.** `TimeGrid`, `ndt(g)`, `timenodes!(ts, om, g, TN)`, `IntegralArrays(order, grid, nrs, chunk, levels)`, `fop.mpanel`, `fop.mtail` are used identically in every task.
