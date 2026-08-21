# Panel-wise time quadrature

Date: 2026-08-19 · Branch: `performance` · Status: design approved, not yet implemented

## Problem

MLE against the SMC' likelihood (`mldsmcp!`, either `:fused` or `:order`) terminates
with `Status: failure (line search failed)` and a large residual gradient, while the
naive closed-form likelihood converges normally on the same data.

Reproduced end-to-end with the stored production case
(`/project/minus3-simulation-data/temp-results/{fop,segments.csv}`: K = 5 epochs,
μ = 1e-8, ρ = 2e-8, α = 2/3, `ndt` = 800, `locut` = 1, 887 609 segments, 200 bins):

```
naive :  Status: success                       |g| = 4.36e-08 <= 5e-08   1022 it
smcp  :  Status: failure (line search failed)  |g| = 1.91e+03            739 it
```

The fit stops with |g| = 1.9e3 — not near any stationary point.

## Root cause

**The likelihood is C⁰ but not C¹ in the parameters.** It has kink surfaces, and
LBFGS bisects onto one and cannot satisfy the Wolfe curvature condition there.

`tolaguerre` selects a quadrature node's epoch by searching on the node's own value
(`while t > getts(TN, epoch+1)`). The map is built so `t` is continuous through that
switch, but its Jacobian is `dt/dz = 2N(epoch)`, so the *slope* changes by the ratio
of adjacent population sizes. At the stalled point, 8.6e-7 away along the
steepest-descent direction:

```
node 328:  N 9.99002e+07 -> 6975.85     ratio 14300
           t 9493.6393  -> 9495.9129

phi' : 4.214151e+05 -> 7.266074e+04     jump = -3.49e+05  = 131 x |phi'(0)|
median |slope change| elsewhere = 0.596          ratio = 5.85e+05
```

The severity is set by N₃ = 9.99e7 sitting at its upper bound — the largest Jacobian
ratio the box permits.

### Evidence that it is a kink

| test | result | excludes |
|---|---|---|
| first differences, 241 pts over ±1e-4 | no outlier above 20× median | a jump; φ is continuous |
| BigFloat(160) vs Float64 | identical to every printed digit | roundoff |
| detrended amplitude vs window | ∝ h, onset at \|a\| ≈ 3e-7 | oscillation |
| central FD vs AD | 84× off at h=1e-5, agrees at h=1e-7 | non-differentiability *at* x₀ |

### Ruled out

Not the fusion — `:order` fails at least as often (4/4 vs 2/4 on synthetic
replicates). Not `npicard` (1,2,3,4,8 identical), not the `max(·,0)` clamps (G > 0
everywhere measured; removing them changes nothing), not roundoff, not `DiffCache`
(survives raw freshly-allocated buffers, bit-identical), not grid monotonicity
(`dC ≥ 0` throughout), not a Jacobian/`Nt` epoch mismatch (zero mismatched nodes).

It requires the transition operator: `order range=1:1` — analytic `firstorder`, no
nodes — is clean; trouble starts at `1:2`.

`ndt` = 800 is not special in itself. It is the value that happens to place a node
1e-6 from an epoch boundary at this parameter point; every other count tried
(400…1600) keeps its closest node ≥7e-5 away and is clean there. Nothing protects
any node count elsewhere in parameter space, so changing `ndt` is not a fix.

### The same defect powers the current adaptivity

The global map allocates nodes by coalescent mass, which is good design:

| epoch | width (gen) | Δy = ΔC/2 | nodes now |
|---|---|---|---|
| 1 | 215 | 0.0049 | 36 |
| 2 | 3066 | 0.557 | 291 |
| 3 | 6214 | 3.1e-05 | **0** |
| 4 | 28302 | 2.029 | 190 |
| 5 (tail) | ∞ | — | 283 |

But that per-epoch count is an integer depending on TN. Each 0 → 1 → 2 transition
*is* a node crossing a boundary — the kink event. TN-adaptive cross-epoch allocation
and a C¹ objective are the same knob. **This design trades the adaptivity away.**

## Design

### Panels

With K = `length(TN)÷2`, T_k = `getts(TN,k)` (T₁ = 0), N_k = `getns(TN,k)`:

- finite panels `[T_k, T_{k+1}]`, k = 1 … K−1 — Gauss-Legendre, `m` nodes each
- tail `[T_K, ∞)` where N ≡ N_K — Gauss-Legendre composed with an algebraic map,
  `mtail` nodes (**amended 2026-08-20**, see "Tail rule" below; Gauss-Laguerre was
  tried first and rejected)

Finite panel, nodes `z_i` / weights `w_i` fixed and TN-independent:

```
t_i  = (T_k + T_{k+1})/2 + (T_{k+1} - T_k)/2 * z_i
om_i = w_i * (T_{k+1} - T_k)/2
```

Tail, in the coalescent variable u = (t − T_K)/(2N_K):

```
u_i  = (1 + z_i)/(1 - z_i)          # algebraic map (-1,1) -> [0,inf), z_i Gauss-Legendre
w~_i = w_i * 2/(1 - z_i)^2          # weight including du/dz
t_i  = T_K + 2*N_K*u_i
om_i = w~_i * 2*N_K
```

`u_i` and `w~_i` depend only on the fixed Legendre node `z_i`, so they are precomputed
in `TimeGrid` and the tail branch has the same shape as the finite panels: `t` affine in
`(T_K, N_K)`, `om` affine in `N_K`.

Both maps are **affine in the epoch parameters**, hence C^∞ in TN. A node cannot
change epoch because its epoch is the panel it was constructed in, fixed before `t`
is computed. The value-dependent search is gone, and with it the kink.

Within an epoch, y = C(t)/2 is affine in t, so Gauss-Legendre placed in y and in t
give the *identical* node set. The coalescent substitution earns its place only in
the tail — which is why it is kept exactly there.

### Tail rule (amended 2026-08-20)

Gauss-Laguerre was the first choice — it is the natural rule for an `e^{-u}` weight —
and it was implemented and then **rejected on measurement**, for two independent
reasons:

1. **Wrong node placement.** Laguerre nodes are spaced by the coalescent rate 1/(2N_K).
   The integrand also carries `exp(-2·rate·r·t)`, whose scale at the largest r is
   ~5.6 generations. With K=1 and N=2e4 the first Laguerre node sits at t = 1192 — 215×
   beyond where the integrand still has mass. Relative error against the analytic
   `firstorder`: 7.4e-15 at r=1, 0.64 at r=1e5, **1.00** at r=3e6 (quadrature 3.8e-96
   against an exact 5.1e-10).
2. **Weight overflow.** The folded weights `w_i·exp(u_i)` are fine at `mtail = 48`
   (measured range 0.114 … 15.4) but Laguerre nodes reach u ≈ 4·mtail, so at the
   `mtail ≳ 200` the accuracy target actually requires, `exp(u)` overflows and the
   weights become `Inf`/`NaN`.

The algebraic map has neither problem. It is dense at **both** ends of [0,∞) — that
two-ended density is exactly the property the original global map had and the reason it
worked. Its weights are large but finite (2.5e-5 … 2.6e5 at mtail=384) and it reproduces
`∫e^{-u}du` and `∫u·e^{-u}du` to 10 digits.

Critically, it is **exactly affine in TN**: measured deviation from linearity 2.2e-16.
The original map's kink came from the epoch *search*, and the tail contains a single
epoch, so there is no branch to take. The exact-affine invariant survives intact.

### Evidence: three rules at matched total node count

Worst relative error against the analytic `firstorder` across the r grid (hi = 3e6).
GLOBAL-ALG is the original production map in its global form; PANEL-LAG is panels +
Gauss-Laguerre tail; PANEL-ALG is panels + algebraic tail.

```
history             K   nodes  GLOBAL-ALG   PANEL-LAG    PANEL-ALG
stationary N=2e4    1    400   3.003e-13    NaN          9.171e-15
real 5-epoch        5    400   4.477e-03    5.018e-13    4.983e-13
N at upper bound    3    400   6.631e-03    NaN          1.745e-10
T at lower floor    3    400   2.154e-03    NaN          4.091e-15
near-empty epoch    3    400   6.675e-04    NaN          1.792e-07
```

PANEL-ALG wins on every history, by four to ten orders where K > 1. The PANEL-LAG
`NaN`s are the weight overflow above.

Geometric convergence in the **finite** panels is confirmed (tail held fixed): the
real 5-epoch history gives 4.8e-2, 1.7e-5, 6.8e-12, floor for m = 8, 16, 32, 64. So
plain affine Legendre is right for the finite panels; the algebraic map earns its place
only in the tail, where the domain is semi-infinite.

### Node budget

`m = 64`, `mtail = 384`. `mtail` is set by the K=1 case, where the tail is the whole
domain; K=5 alone would need only ~144.

Total = (K−1)·64 + 384: **384 / 512 / 640 / 832** for K = 1 / 3 / 5 / 8, against 800
today — comparable cost, four to ten orders better accuracy.

The cost of dropping adaptivity is that a near-empty epoch (epoch 3 above, 1.2e-5 of
the coalescent mass) receives its full `m` nodes for nothing. Affordable.

### Known limitation (pre-existing, not introduced here)

A **stationary** (K=1) history with N ≳ 1e6 cannot resolve the recombination scale at
the largest r: relative error 7.2e-1 at N=1e6 and 1.00 at N=1e8. The original map gives
the same numbers at matched cost, so this is a property of the existing code, not a
regression. It matters because the sequential fit starts at K=1.

Conversely a multi-epoch history whose *tail* epoch has N=1e8 is fine under PANEL-ALG
(4.3e-12) and fails outright under GLOBAL-ALG (1.00), because the finite panels cover
the small-t region regardless of the tail's N.

**Hard constraint:** `m` and `mtail` may depend on μ, ρ and the r-grid — all fixed
during a fit — but never on TN. A rule keyed on epoch width would reintroduce
parameter-dependent discreteness, trading a kink for a jump. Same discipline as
`getnpicard` and `getorder`.

### Open: shipped defaults under-resolve the transition integrals (2026-08-20)

Scored in Poisson sigma against each scheme's own high-resolution limit,
real 5-epoch history, 800 bins:

    OLD global map, ndt=800 (production):  max 0.42 sigma,   0/800 bins > 1 sigma
    NEW panels, m=64 mtail=384 (ndt=640):  max 2.55 sigma, 127/800 bins > 1 sigma

Convergence of the new scheme: (m=128,mt=256) 1.22 sigma; (192,256) 0.76; (256,384) 0.46.
K=1 depends on mtail alone: 384 -> 2.54 sigma, 768 -> 1.20, 1536 -> 0.52.

Matching the old map's fidelity needs roughly m=256, mtail=1536 — about 4x the current
defaults and about 4x the runtime (a real fit goes from ~81 s to ~5 min).

Node counts were calibrated against `firstorder`, the analytic ORDER-1 terminal
integral, where the ranking is the opposite (panels 5e-13 vs the old map 4.5e-3);
`firstorder` structurally cannot see the transition-operator integrals, which dominate.
Recalibration should target a transition-level self-convergence criterion (~0.2 sigma
against a high-resolution reference), not `firstorder`. The defaults were deliberately
NOT changed here because it is a cost/accuracy decision for the user.

**Resolved (2026-08-21).** The 4x is not a node-budget problem at all: the transition
kernel has a corner at `t' = t` that moves with the row, and every TN-independent node
map is O(1/N) against it. See `../2026-08-20-time-quadrature-node-budget-spike.md` for
the measurements and `2026-08-21-diagonal-corner-correction-design.md` for the fix —
composite sub-panels plus a per-row partial integral of the local interpolant, which
converges at the smallest setting tried and inverts the 4x penalty.

## Code structure

New, in `SMCpIntegrals.jl`:

```julia
struct TimeGrid
    m::Int; mtail::Int; K::Int
    zleg::Vector{Float64};  wleg::Vector{Float64}   # Gauss-Legendre on (-1,1), finite panels
    utail::Vector{Float64}; wtail::Vector{Float64}  # u=(1+z)/(1-z), weight incl. du/dz
end
ndt(g) = (g.K - 1) * g.m + g.mtail
```

`timenodes!(ts, om, g, TN)` — the whole node setup, ~20 lines, no branch on a node's
value. Built once per fit from K; nothing in `TimeGrid` depends on TN.

**Deleted:** `tolaguerre`, `tolegendre`.

**Changed:** in `prordn!` and `fusedsweep!` the setup block collapses to `timenodes!`
plus the `qs[j] = pt(ts[j], TN)` loop. `IntegralArrays` carries the grid instead of
`zs`/`wt`; `n_dt` becomes derived.

**Unchanged:** `sepkernel!`, `transition!`, the semiseparable prefix/suffix structure,
the r-sweep, the Picard loop, `firstorder`. Concatenated panels are ascending and
C(t) is continuous across joins, so every downstream invariant holds. The fix is
confined to node construction.

**API ripple:** `FitOptions.ndt` changes meaning from total to per-panel, so rename it
`mpanel` and add `mtail` rather than silently redefining it — a caller passing
`ndt = 800` would otherwise get 800 nodes per epoch. Three construction sites need K
threaded in: `mle_optimization.jl` (two, K = `options.nepochs`) and `IBSpector.jl:66`
(K = `length(TN)÷2`).

## Validation

### 1. Analytic anchor

`firstorder(r, rate, TN)` is the closed form of the order-1 terminal integral, so the
quadrature is scored against truth at any K and any TN, with no reference
implementation:

```
err(r) = | sum_j rate*exp(-2*rate*r*t_j)*q_j*2*t_j*om_j - firstorder(r, rate, TN) | / firstorder(...)
```

Target ≤1e-6 relative across the r grid — well below the fusion's own ~1e-3 σ, against
today's 0.13 σ (ρ/μ=1) to 0.44 σ (ρ/μ=4).

Limit: this exercises only the terminal integral with a smooth integrand. It does
**not** exercise the transition operator, which is where the kink lived. Necessary,
not sufficient. Use self-convergence in m (m, 2m, 4m) for the transition level.

### 2. Smoothness — the acceptance test

| check | today (ndt=800, real data) | required |
|---|---|---|
| detrended residual vs window h | ∝ h; 1e-8 → 0.42 as h: 3e-7 → 1e-5 | flat at the ~1e-8 float floor for all h |
| max slope change / median | 5.85e5 | O(1) |
| central FD vs AD | 84× off at h=1e-5 | agrees at every h above roundoff |

Across several directions, several TN, **and a sweep of m** — the pathology only
surfaced because node counts were swept.

### 3. Node-map unit test

For random TN and direction d, every node's `t_j(TN + a·d)` must have bounded second
differences in a. Today node 328's slope jumps by 14 300×. Cheap, no likelihood
needed, tests the invariant that actually broke.

### 4. Epoch structures

K = 1…8 as the sequential fit sweeps, plus adversarial cases: an N pinned at the 1e8
bound adjacent to a normal N (the 14 300 ratio), epochs at the T_low = 10 floor, and
epochs holding near-zero coalescent mass.

### 5. End-to-end

`fit_model_epochs!` with `naive=false` on the stored case must reach
`Status: success`, |g| ≤ 5e-8. Plus MLE stability: parameters move <1% between m and
2m, against the 17% shift measured in T₂ between ndt = 800 and 3200.

### 6. Regression

`spectra.jl`, `smcp_semiseparable.jl`, `smcp_fused.jl` hard-code node counts and need
the rename threaded through. ForwardDiff-compatibility tests apply unchanged —
`timenodes!` is pure arithmetic.

### Known gap

Nothing here shows the panel scheme beats the current one at fixed cost for the
*transition-level* integrals. `firstorder` cannot see them and self-convergence only
shows internal consistency. The MLE-stability check in (5) is the real evidence.

## Parked / not doing

- **Graded sub-panels within an epoch** — contingency only, if calibration shows a
  fixed `m` cannot hold the exp(−2·rate·r·t) decay in the first panel (β ≈ 41 at
  r ≈ 3e6, which m = 48 should handle). Any sub-panel count must be TN-independent.
- **The r' ∈ [0,1) slab** — unaffected. It enters every bin through the smooth `A`
  accumulator; a bias, never a source of non-smoothness. Stays open on its own terms.
- **The `max(·,0)` clamps** — G > 0 and t − G > 0 at every point measured (stalled
  point, naive optimum, 6 random TN); removing them changes nothing. Guards, not
  active constraints. No panel split at a G root is needed.
- **L as a free parameter** — pinned at its upper bound in both paths; worth fixing
  rather than inferring, but unrelated to this.
- **Swapping the line search** — rejected. BackTracking reports success in 8/8 SMC'
  runs while sitting at |g| up to 1.7e3, i.e. at the same non-stationary points
  HagerZhang correctly refuses; it does not test the curvature condition. It would
  hand back converged-looking fits that are not at a stationary point.
