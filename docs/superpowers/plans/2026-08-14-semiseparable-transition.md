# Semiseparable Transition Operator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `O(n_dt²)` transition step in `prordn!` with an exactly equivalent `O(n_dt)` recurrence, cutting the dominant cost of the SMC' order-n integrals by 8–33× with no change in results.

**Architecture:** The local-time transition kernel `ptt` is exactly diagonal-plus-rank-one-lower-plus-rank-one-upper (semiseparable), because `cumcr` is additive in time. The matrix product `temp[i,j] = Σ_k jprt[k,i]*qtt[k,j]` therefore collapses to one backward suffix sum plus one forward prefix sum per `r` bin. The order loop, the `r` recurrence, the bin conventions and all public signatures are unchanged. The `n_dt × n_dt` `qtt` buffer disappears entirely.

**Tech Stack:** Julia 1.12, ForwardDiff, PreallocationTools (`DiffCache`), Base.Threads, Test.

**Spec:** `../../../smcp-integrals-notes.tex` (in the parent `minus3-simulation` directory; §2 derives the decomposition, §7.1 gives the measured speedup and the machine-precision equivalence check)

## Global Constraints

- This is an **exact algebraic restructuring**. Any observable change in `mldsmcp` output is a bug, not a tradeoff. Acceptance is agreement with the current dense path to `~1e-14` relative.
- All public signatures unchanged: `prordn!(bag, rs, edges, rate, TN)`, `IntegralArrays(order, ndt, nrs, chunk, levels=1)`, `mldsmcp`, `mldsmcp!`.
- Must stay ForwardDiff-compatible: every new buffer is a `DiffCache`, all new arithmetic is `+ - * / exp` only. No `eigen`, no BLAS, no in-place LAPACK.
- `t` nodes are ascending (`gausslegendre` nodes ascend and `tolegendre` is monotone increasing). The recurrences depend on this. Do not reorder `ts`.
- Do not delete `ptt` — after this change it is used only by `test/spectra.jl` ("aux functions SMCpIntegrals"), which is fine. It is also the reference the new code is tested against.
- Do not attempt the "fused order loop" idea from §5 of the spec. It is unresolved and explicitly out of scope.
- Branch: `performance` (already created from `main` at `e54edad`).

## Background: the decomposition being implemented

Writing `C(t) = cumcr(0,t,TN)`, `N(t) = Nt(t,TN)`, `R(t) = margrecomb(t,TN)` and

```
G(t) = N(t) + R(t) - N(0)*exp(-C(t))
```

the three branches of `ptt(t | t')` become:

```
ptt(t|t') =  G(t)/N(t)                                   t < t'    (no t' dependence)
             t - G(t)                                    t = t'    (the atom)
             [exp(-C(t)/2)/N(t)] * [exp(C(t')/2)*G(t')]   t > t'    (rank one)
```

With `ts` ascending, `x = jprt[:,i]`, `om[j] = wt[j]*dts[j]`, and the current code's
`max(ptt(...), 0)` clamp, the product `temp[i,j] = Σ_k x_k * ptt(t_j|t_k) * w_k` is:

```
temp[i,j] = Phi[j] * Σ_{k>j} x_k*om_k          # upper triangle
          + Stil[j] * Ninv[j]                  # lower triangle, rescaled prefix sum
          + dgn[j] * x_j                       # diagonal atom, weight 1 (not om)
```

where `Stil[j] = exp(-C_j/2) * Σ_{k<j} exp(C_k/2)*Gc_k*x_k*om_k` obeys the overflow-free recurrence

```
Stil[1]   = 0
Stil[j+1] = exp(-dC[j]/2) * (Stil[j] + Gc[j]*x_j*om_j),    dC[j] = C_{j+1} - C_j >= 0
```

**Clamp equivalence (subtle, get this right).** The current code clamps the assembled
`ptt` value. Because `exp(-C/2)/N > 0` strictly, clamping the lower-triangle product is
identical to clamping `G` at build time. So:

- `Gc[j]  = max(G_j, 0)`  — used for BOTH the upper triangle and the prefix sum
- `Phi[j] = max(G_j, 0) / N_j`  — equals `max(G_j/N_j, 0)` since `N > 0`
- `dgn[j] = max(t_j - G_j, 0)`  — uses the **unclamped** `G_j`

Using clamped `G` in `dgn` would be wrong.

## File Structure

- `src/Spectra/SMCpIntegrals.jl` — add `sepkernel!` and `transition!`; rewrite the transition
  step of `prordn!`; drop the `qtt` field from `IntegralArrays` and add six length-`n_dt`
  `DiffCache` vectors. Single responsibility unchanged: the SMC' integrals.
- `test/smcp_semiseparable.jl` — **new**. Owns all equivalence testing for this change:
  unit-level (`transition!` vs a dense reference), end-to-end (`mldsmcp` vs a dense
  reference), and ForwardDiff compatibility. Self-contained — it defines its own dense
  reference rather than depending on golden files.
- `test/runtests.jl` — one `include` line.

---

### Task 1: Semiseparable kernel builder and apply

**Files:**
- Modify: `src/Spectra/SMCpIntegrals.jl` (add two functions after `ptt`, around line 75)
- Create: `test/smcp_semiseparable.jl`
- Modify: `test/runtests.jl` (add include)

**Interfaces:**
- Consumes: `Nt`, `cumcr` (from `..CoalescentBase`), `margrecomb`, `ptt`, `tolegendre` (module-local)
- Produces:
  - `sepkernel!(Phi, dgn, Gc, Ninv, dC, ts, TN) -> nothing` — fills five length-`n` vectors from ascending nodes `ts`
  - `transition!(temp, jprt, i, Phi, dgn, Gc, Ninv, dC, om, ndt) -> nothing` — writes row `i` of `temp`

- [ ] **Step 1: Write the failing test**

Create `test/smcp_semiseparable.jl`:

```julia
using IBSpector.Spectra
using IBSpector.Spectra.SMCpIntegrals: sepkernel!, transition!, ptt, tolegendre
using IBSpector.Spectra.CoalescentBase: Nt, cumcr
using Test

const SMCp = IBSpector.Spectra.SMCpIntegrals

# Dense reference: builds qtt exactly as the pre-change prordn! did, then
# temp[i,j] = sum_k jprt[k,i] * qtt[k,j].
function dense_transition(jprt, i, ts, om, TN)
    ndt = length(ts)
    qtt = zeros(ndt, ndt)
    for a in 1:ndt, b in 1:ndt
        w = a == b ? 1.0 : om[b]
        qtt[b, a] = max(ptt(ts[a], ts[b], TN), 0.0) * w
    end
    [sum(jprt[k, i] * qtt[k, j] for k in 1:ndt) for j in 1:ndt]
end

function nodes(ndt, TN)
    zs, wt = SMCp.gausslegendre(ndt)
    ts = zeros(ndt); dts = zeros(ndt)
    for j in 1:ndt
        ts[j], dts[j] = tolegendre(zs[j], TN)
    end
    ts, wt .* dts
end

@testset "semiseparable transition == dense transition" begin
    TNs = [
        [3.0e9, 10000.0],
        [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0],
        [3.0e9, 20000.0, 60000.0, 8000.0, 8000.0, 16000.0, 1600.0, 2000.0, 400.0, 10000.0],
    ]
    for TN in TNs
        ndt = 120
        ts, om = nodes(ndt, TN)
        @test issorted(ts)

        Phi = zeros(ndt); dgn = zeros(ndt); Gc = zeros(ndt)
        Ninv = zeros(ndt); dC = zeros(ndt)
        sepkernel!(Phi, dgn, Gc, Ninv, dC, ts, TN)

        @test all(isfinite, Phi)
        @test all(isfinite, Gc)
        @test all(dC .>= 0)          # C is monotone increasing
        @test all(Phi .>= 0)
        @test all(dgn .>= 0)

        nrs = 4
        jprt = abs.(randn(ndt, nrs)) .* 1e-6
        temp = zeros(nrs, ndt)
        for i in 1:nrs
            transition!(temp, jprt, i, Phi, dgn, Gc, Ninv, dC, om, ndt)
            ref = dense_transition(jprt, i, ts, om, TN)
            @test temp[i, :] ≈ ref rtol = 1e-10
        end
    end
end
```

Add to `test/runtests.jl` immediately after the existing `include("spectra.jl")`:

```julia
include("smcp_semiseparable.jl")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL — `UndefVarError: sepkernel! not defined in IBSpector.Spectra.SMCpIntegrals`

- [ ] **Step 3: Write minimal implementation**

In `src/Spectra/SMCpIntegrals.jl`, insert after the `ptt` function (after line 75, before `tolaguerre`):

```julia
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS — "semiseparable transition == dense transition"

- [ ] **Step 5: Commit**

```bash
git add src/Spectra/SMCpIntegrals.jl test/smcp_semiseparable.jl test/runtests.jl
git commit -m "add semiseparable form of the SMC' transition kernel

ptt is exactly diagonal + rank-one lower + rank-one upper because cumcr
is additive, so the transition matvec collapses to a suffix sum plus a
rescaled prefix sum. Not yet wired into prordn!."
```

---

### Task 2: Wire into `prordn!` and drop the `qtt` buffer

**Files:**
- Modify: `src/Spectra/SMCpIntegrals.jl:98-237` (`IntegralArrays` struct, its constructor, both `prordn!` methods)
- Modify: `test/smcp_semiseparable.jl` (append a second testset)

**Interfaces:**
- Consumes: `sepkernel!`, `transition!` from Task 1
- Produces: `IntegralArrays` with fields `order, n_dt, nrs, ys, res, jprt, temp, zs, wt, ts, dts, qs, om, Phi, dgn, Gc, Ninv, dC` — the `qtt` field is **removed**; six length-`n_dt` `DiffCache` vectors are added. Public constructor signature unchanged.

**Note on an existing argument-order slip:** the bag wrapper currently passes `ts_, qs_, dts_` while the inner method declares `ts, dts, qs`, so the last two are swapped. It is harmless today (both are same-typed scratch filled inside the function and never read by callers), but this task adds six more vectors to that list, so the order is straightened here deliberately.

- [ ] **Step 1: Write the failing test**

Append to `test/smcp_semiseparable.jl`:

```julia
# Full pre-change prordn!, kept verbatim as the regression reference.
function dense_prordn(rs, edges, rate, order, ndt, TN)
    nrs = length(rs)
    ts, om = nodes(ndt, TN)
    qs = [SMCp.pt(t, TN) for t in ts]
    qtt = zeros(ndt, ndt)
    for a in 1:ndt, b in 1:ndt
        w = a == b ? 1.0 : om[b]
        qtt[b, a] = max(ptt(ts[a], ts[b], TN), 0.0) * w
    end
    res = zeros(nrs, order)
    jprt = zeros(ndt, nrs)
    temp = zeros(nrs, ndt)
    for i in 1:nrs, j in 1:ndt
        jprt[j, i] = rate * exp(-2rate * rs[i] * ts[j]) * qs[j]
    end
    for i in 1:nrs
        res[i, 1] = SMCp.firstorder(rs[i], rate, TN)
    end
    for o in 1:order-1
        for i in 1:nrs, j in 1:ndt
            temp[i, j] = sum(jprt[k, i] * qtt[k, j] for k in 1:ndt)
        end
        for j in 1:ndt
            acc = 0.0
            for i in 1:nrs
                w = edges[i+1] - edges[i]
                s = acc * exp(-2rate * (rs[i] - edges[i]) * ts[j])
                wi = w <= 1 ? w : rs[i] - edges[i]
                s += temp[i, j] * (-expm1(-2rate * wi * ts[j])) / 2ts[j]
                jprt[j, i] = s
                frac = (-expm1(-2rate * w * ts[j])) / 2ts[j]
                acc = exp(-2rate * w * ts[j]) * acc + temp[i, j] * frac
            end
        end
        for i in 1:nrs
            res[i, o+1] = sum(jprt[j, i] * 2 * ts[j] * om[j] for j in 1:ndt)
        end
    end
    res
end

@testset "prordn! matches the dense reference" begin
    TNs = [
        [3.0e9, 10000.0],
        [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0],
        [3.0e9, 20000.0, 60000.0, 8000.0, 8000.0, 16000.0, 1600.0, 2000.0, 400.0, 10000.0],
    ]
    edges = vcat(collect(1.0:1.0:60.0),
                 exp.(range(log(61.0), log(2.0e4), length = 25)))
    rs = [(edges[i+1] - edges[i]) <= 1 ? edges[i] : sqrt(edges[i] * edges[i+1])
          for i in 1:length(edges)-1]
    order, ndt = 6, 120
    for TN in TNs, (mu, rho) in ((1.25e-8, 1.0e-8), (1.0e-8, 8.0e-8))
        rate = mu + rho
        bag = IntegralArrays(order, ndt, length(rs), Val{length(TN)})
        SMCp.prordn!(bag, rs, edges, rate, TN)
        got = get_tmp(bag.res, eltype(TN))
        want = dense_prordn(rs, edges, rate, order, ndt, TN)
        @test size(got) == size(want)
        @test got ≈ want rtol = 1e-12
    end
end
```

The file already has `using IBSpector.Spectra`; add `using IBSpector.Spectra.PreallocationTools` to the imports at the top of `test/smcp_semiseparable.jl` so `get_tmp` resolves.

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL — the new testset errors or mismatches, because `prordn!` still uses the dense `qtt` path and `IntegralArrays` has no `Phi` field yet. (It may also pass trivially at this point since both sides are the dense algorithm — that is acceptable; the test becomes meaningful in Step 4. Confirm it at least runs without error before proceeding.)

- [ ] **Step 3: Write minimal implementation**

Replace `src/Spectra/SMCpIntegrals.jl` lines 98-237 (from `struct IntegralArrays` to the end of the second `prordn!`, leaving the final `end` of the module) with:

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS — both new testsets, plus all pre-existing testsets (`mld smcp runs`, `Test core functionality`, etc.) still green.

- [ ] **Step 5: Commit**

```bash
git add src/Spectra/SMCpIntegrals.jl test/smcp_semiseparable.jl
git commit -m "use the semiseparable transition in prordn!

Drops the transition step from O(nrs*ndt^2) to O(nrs*ndt) and removes the
ndt x ndt qtt buffer entirely. Exact restructuring: agrees with the dense
path to ~1e-14. Also straightens the ts/dts/qs argument order, which was
transposed between the two prordn! methods."
```

---

### Task 3: ForwardDiff compatibility and measured speedup

**Files:**
- Modify: `test/smcp_semiseparable.jl` (append a third testset)

**Interfaces:**
- Consumes: everything from Task 2. Produces nothing new.

- [ ] **Step 1: Write the failing test**

Add `using ForwardDiff` to the import block at the **top** of
`test/smcp_semiseparable.jl` (a bare `using` is only legal at top level, never inside a
`@testset` block), then append:

```julia
@testset "prordn! is ForwardDiff-differentiable" begin
    edges = collect(1.0:1.0:40.0)
    rs = collect(1.0:1.0:39.0)
    order, ndt = 4, 60
    TN0 = [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0]

    function total(TN)
        bag = IntegralArrays(order, ndt, length(rs), Val{length(TN)})
        SMCp.prordn!(bag, rs, edges, 2.25e-8, TN)
        sum(get_tmp(bag.res, eltype(TN)))
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

Add `ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"` to the `[deps]` of `test/Project.toml`.

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL initially with a `ForwardDiff` load error until `test/Project.toml` is updated. After that it should pass — if it does not, the failure is a real AD bug in the new code and must be fixed before continuing.

- [ ] **Step 3: Record the measured speedup**

No source changes. Run this benchmark and paste the numbers into the commit message:

```bash
julia --project=. -t auto -e '
using IBSpector.Spectra, IBSpector.Spectra.PreallocationTools
const S = IBSpector.Spectra.SMCpIntegrals
TN = [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0]
edges = vcat(collect(1.0:1.0:100.0), exp.(range(log(101.0), log(3.0e4), length=60)))
rs = [(edges[i+1]-edges[i]) <= 1 ? edges[i] : sqrt(edges[i]*edges[i+1]) for i in 1:length(edges)-1]
for ndt in (400, 800, 1600)
    bag = IntegralArrays(14, ndt, length(rs), Val{length(TN)})
    S.prordn!(bag, rs, edges, 2.25e-8, TN)
    t = @elapsed for _ in 1:3; S.prordn!(bag, rs, edges, 2.25e-8, TN); end
    println("ndt=$ndt  ", round(1e3t/3, digits=1), " ms  (", Threads.nthreads(), " threads)")
end'
```

Expected, from the spec's single-threaded measurements at `order=14` (`n_rs=159`): the
dense path took 477 / 1932 / 8775 ms at `ndt` = 400 / 800 / 1600; the semiseparable path
took 57.4 / 148.7 / 264.8 ms — 8× / 13× / 33×. Threaded numbers will be lower for both;
what must hold is that cost now grows roughly linearly in `ndt` rather than quadratically.

- [ ] **Step 4: Run the full suite once more**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS, all testsets.

- [ ] **Step 5: Commit**

```bash
git add test/smcp_semiseparable.jl test/Project.toml
git commit -m "test ForwardDiff compatibility of the semiseparable prordn!

Gradient checked against central differences on the population-size
parameters. Benchmark at order=14, nrs=159: <paste measured numbers>."
```

---

## Out of Scope (deliberately)

- **The fused order loop** (spec §5). The `order` loop stays exactly as it is, so
  `res[:,o]` per-order diagnostics keep working unchanged and `getorder` keeps its
  current meaning. The fusion is blocked on an unresolved wide-bin / large-α
  discretisation problem documented in the spec.
- **Hand-written adjoint gradients** (spec §9). Dropped at the user's request.
- **Exponential-trapezoid `r` stepping** (spec §8). Measured convergence order is already
  1.89, so there is nothing to gain.
- **Removing `ptt`.** Still used by `test/spectra.jl` and by the new tests as the
  reference implementation.

## Self-Review Notes

- Spec coverage: §2 (Finding 1) → Tasks 1–2. §7.1 (measured equivalence and speedup) →
  Tasks 2–3. §§5, 8, 9 explicitly out of scope above.
- Type consistency: `sepkernel!` and `transition!` use the same argument order and names
  (`Phi, dgn, Gc, Ninv, dC`) in the definition, in `prordn!`, and in both tests. The
  `IntegralArrays` field order matches the constructor's positional arguments.
- The clamp-equivalence rule (`Gc` clamped, `dgn` unclamped) is stated in the Background,
  repeated as a code comment in Task 1 Step 3, and exercised by the Task 1 test across
  three demographies including a non-monotone one.
