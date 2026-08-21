# Diagonal-corner correction for the SMC' transition quadrature

Date: 2026-08-21 · Branch: `performance` · Status: design approved, implementation in progress

Follows on from `2026-08-19-panel-time-quadrature-design.md` (section "Open: shipped
defaults under-resolve the transition integrals") and from the measurements in
`../2026-08-20-time-quadrature-node-budget-spike.md`.

## Problem

PANEL-ALG removed the C¹ kink that stalled the MLE, but it needs ~4x the nodes of the
old global map to reach the same accuracy, and matching the old production fidelity
costs ~4x the runtime.

The spike showed this is not a property of the node map. Every TN-independent map
tested — panels, uniform composites, graded meshes, rescaled Laguerre, frozen
GLOBAL-ALG — converges at exactly **O(1/N)** in the total node count, differing only
in a constant that varies by less than 2x.

The cause is the transition kernel. With `G(t) = N(t) + R(t) − N(0)e^{−C(t)}`,

```
K(t,t') = G(t)/N(t)                                for t' > t
        = [e^{−C(t)/2}/N(t)] * [e^{C(t')/2} G(t')] for t' < t
        + delta(t−t') * (t − G(t))                 the exact atom
```

Both branches agree in value at `t' = t`, so `K` is C⁰; but the upper branch is constant
in `t'` and the lower one is not, so the slope jumps. `K(t, ·)` has a **corner at
`t' = t`, whose location moves with the row**. No fixed shared node set resolves it.

`transition!` currently integrates each row by splitting the *global* Gauss sum at the
row index and using whole-panel weights on what is really a partial interval. That is
the O(1/N) rule. Splitting the quadrature at the corner instead takes one row integral
from `5.4e-4` to `3.2e-15` at m = 32 — eleven orders, at 1/32 of the nodes.

## Design

The corner always sits **at a node**. With composite sub-panels of `msub` nodes, whole
sub-panels below and above the row are integrated exactly by their own Gauss rules, and
only the single sub-panel containing the row is partial. That partial piece is the
integral of the local interpolant over `[a_p, t_j]` — a fixed `msub × msub` matrix
applied per row, i.e. **O(n · msub)**, not O(n²).

### Grid

```julia
struct TimeGrid
    msub::Int; nfin::Int; ntail::Int; K::Int
    zleg::Vector{Float64}; wleg::Vector{Float64}   # msub-point Gauss-Legendre on (-1,1)
    Lp::Matrix{Float64}                            # msub x msub, Lp[q,i] = ∫_{-1}^{z_q} l_i
    uedge::Vector{Float64}                         # ntail+1 tail sub-panel edges in u
    ulast::Vector{Float64}; wlast::Vector{Float64} # algebraic map for [uedge[end], ∞)
    ncorr::Int                                     # leading tail sub-panels that get corrected
end
TimeGrid(K; msub = 8, nfin = 4, ntail = 32, dumax = 40.0)
npanels(g) = (g.K - 1) * g.nfin + g.ntail
ndt(g)     = npanels(g) * g.msub
```

- **Finite epoch `k`**: `nfin` equal sub-panels over `[T_k, T_{k+1}]`, `msub`
  Gauss-Legendre nodes each, affine in `t`. Every node stays exactly affine in `TN`,
  and no node can migrate between epochs — the PANEL-ALG invariant is untouched.
- **Tail**: sub-panel edges `uedge[q] = (1+z_q)/(1−z_q)` taken on a *uniform z grid*
  over `(−1,1)`. That keeps the two-ended density in `u` which the tail rule depends on
  (dense near `u = 0`, where the recombination scale lives at large `r`). Inside each
  finite tail sub-panel the nodes are placed **affinely in `u`**, not through the
  algebraic map. Affine-in-`u` is what makes `om[j] = wleg[i] * hw[p]` hold, so one `Lp`
  matrix serves every sub-panel. The final sub-panel `[uedge[ntail], ∞)` keeps the
  algebraic map (`ulast`/`wlast`) and is never corrected.
- `ncorr` counts the leading tail sub-panels whose `du` is below `dumax`. Tail nodes
  live in `u` and are **TN-independent**, so `ncorr` is a grid constant fixed in the
  constructor — never a runtime test on a value. The C^∞ invariant survives.

### The partial-integral matrix

Gauss-Legendre orthogonality gives the Lagrange basis in closed form,
`l_i(z) = Σ_{k=0}^{msub−1} ((2k+1)/2) w_i P_k(z_i) P_k(z)`, hence

```
Lp[q,i] = Σ_k ((2k+1)/2) * w_i * P_k(z_i) * I_k(z_q)
I_0(x) = x + 1        I_k(x) = (P_{k+1}(x) − P_{k−1}(x)) / (2k+1)
```

Exact, no linear solve, O(msub³) once at construction. `Lp` is a pure constant: it
carries no `TN` dependence, so it is AD-transparent.

### Node setup

`timenodes!(ts, om, EE, EB, hw, g, TN)` gains three outputs, all read off the affine
maps directly (no `cumcr` call — inside epoch `k`, `d(C/2)/dt = 1/(2N_k)`; in the tail
`d(C/2) = du` exactly):

- `EE[j] = exp((C(t_j) − C(a_p))/2)`, with `a_p` the left edge of `j`'s sub-panel
- `EB[p] = exp((C(b_p) − C(a_p))/2)`
- `hw[p]`, the sub-panel half-width in the integration variable

`sepkernel!` correspondingly drops `dC`, which `EE`/`EB` supersede.

### The two corrected passes

Upper branch (backward over sub-panels); `sfx` is the exact `∫_{b_p}^∞ x dt'`:

```
sfx = 0
for p = npan:-1:1
    s0 = (p-1)*msub
    Sp = Σ_i x[s0+i]*om[s0+i]
    for q = 1:msub
        low = hw[p] * Σ_i Lp[q,i]*x[s0+i]
        out[s0+q] = Phi[s0+q] * (sfx + Sp - low)
    end
    sfx += Sp
end
```

Lower branch (forward); `st` is `∫_0^{a_p} e^{−(C(a_p)−C(t'))/2} G x dt'`, referenced to
the current sub-panel's **left** edge:

```
for p = 1:npan
    s0 = (p-1)*msub
    for q = 1:msub
        low = hw[p] * Σ_i Lp[q,i]*EE[s0+i]*Gc[s0+i]*x[s0+i]
        out[s0+q] += (st + low) * Ninv[s0+q] / EE[s0+q] + dgn[s0+q]*x[s0+q]
    end
    st = (st + Σ_i om[s0+i]*EE[s0+i]*Gc[s0+i]*x[s0+i]) / EB[p]
end
```

Two length-`msub` dot products per row. Uncorrected sub-panels (tail `q > ncorr`, and
the semi-infinite one) fall back to the shipped node-to-node recursion inside the
sub-panel; the handoff of `st` across that boundary is re-referenced explicitly, because
the shipped recursion carries `st` relative to the running node rather than to `a_p`.
Which sub-panels are corrected is read off `g.ncorr` and the panel index only.

The delta atom `dgn[j] * x[j]` is exact and unchanged.

### Overflow: no guard (deliberate)

The lower branch needs `exp((C_i − C(a_p))/2)`, bounded by `exp(dC_p/2)`. Inside a
finite epoch `dC_p` is TN-dependent, and the parameter box (`Nlow = 10`, `Tupp = 1e7`)
permits `dC_p/2 > 709`. **No guard is applied.** Such a history returns `NaN` where the
shipped code merely underflows the coalescent factor to 0. Reaching it requires a
population size near the lower bound held across a very wide epoch, where the
likelihood is already numerically zero; the rejected alternatives (a `min` clamp, or a
C^∞ bump blend) both buy robustness in a degenerate region at the cost of extra
machinery, and one of them dents the C¹ invariant this branch exists to protect.

### API strip

`prordn!` and the matrix-form `transition!` are deleted from `src/`. They own the only
large allocations left (`jprt` and `temp`, each `n_dt × nrs` `DiffCache`) and the whole
`order`/`res` apparatus. An unthreaded reference `prordn_ref!` moves to `test/`, built
column by column on the `src` vector `transition!`, so it reuses the code under test
rather than duplicating it. `mldsmcp!` loses `method = :order`; `FitOptions` loses
`order` and swaps `mpanel`/`mtail` for `msub`/`nfin`/`ntail`; `getorder` goes;
`Base.Threads` leaves `SMCpIntegrals.jl` entirely.

Comparing whole fits between the fused and order paths is explicitly **not** a goal —
only the objective function's analytic properties are being improved.

## Expected result

From the prototype in the spike document, `TNFIT` at 800 bins, max Poisson σ against a
common reference:

| nodes | plain | corrected |
|---:|---:|---:|
| 384 | 1.808 | 0.000016 |
| 768 | 0.917 | 0.000000 |
| 3072 | 0.239 | 0.000000 |

Converged at the smallest setting tried. The correction costs ~2x per node at
`msub = 8` but needs ~7x fewer nodes, so against the setting the 2026-08-19 spec says is
actually required (`m = 256, mtail = 1536`) it is ~3x faster **and** more accurate. The
4x node penalty does not merely disappear; it inverts.

## Validation

1. `Lp` reproduces `∫_{−1}^{z_q} z^d dz` exactly for `d = 0 … msub−1`, and
   `Σ_i Lp[q,i] == z_q + 1`.
2. `EE`/`EB` match `exp(cumcr(...)/2)` recomputed from `CoalescentBase`; `ts` strictly
   ascending; `ndt` consistent; the existing "exactly affine in `TN`" testset extended.
3. `transition!` against a dense O(n²) reference assembled from `ptt` with exact
   split-at-node quadrature: ≤1e-10 relative at `msub = 8, nfin = 16`, against the
   shipped rule's 2.4e-2.
4. Self-convergence of `fusedsweep!` in `(msub, nfin, ntail)` flat, not O(1/N).
5. Zero allocations for a warmed `fusedsweep!` at `Float64` and at `ForwardDiff.Dual`.
6. The C¹ acceptance suite unchanged and still green: `timenodes!` exactly affine in
   `TN`, non-monotone `TN` rejected, "likelihood is smooth along a line".
7. End to end: the stored production case reaches `Status: success`, `|g| ≤ 5e-8`, at
   lower wall time than `m = 256, mtail = 1536`.

## Calibration (measured 2026-08-21)

`th_discr` (the theory-side binning, `src/corrections.jl`) and
`(msub, nfin, ntail)` were calibrated against total histogram counts, μ, ρ and
the r-grid — never against `TN`, the same discipline as `getnpicard`. Full tables
in `../2026-08-21-calibration-measurements.md`; instrument in
`bench/calibrate_quadrature.jl`.

**Node budget.** At `(msub, nfin, ntail) = (8, 12, 16)` — 744 nodes at K=5 — the
worst case over five histories, three α and three `th_discr` is
6.7e-9 / 2.1e-8 / 6.7e-8 / 2.1e-6 sigma at 1e5 / 1e6 / 1e7 / 1e8 segments,
against 0.69 / 2.18 / 6.90 / 21.8 for the pre-panel production setting (old
global map, 800 nodes). Seven orders better at fewer nodes.

The counts deliberately carry **no** dependence on the total count. Poisson sigma
grows as √Ntot while the corrected rule converges geometrically in `msub`/`nfin`,
so a 0.01-sigma target would not bind until ~2e16 segments. The dependence is
real and vacuous; a selector returning one triple everywhere would be a knob
pretending to be a calibration.

**Theory binning.** `th_discr` has **no convergent limit**: successive
differences stop shrinking after 400→800 and then grow, on every history. Three
measurements place this outside the quadrature — the pre-panel map reproduces
the same numbers to three digits; the quadrature is converged at `th_discr` =
6400 (0.0038 sigma against a 1884-node reference); and with `lo = 2000`, where
no fine bin is one base pair wide, the binning converges cleanly at
O(1/th_discr). The cause is the `w <= 1` branch of `fusedsweep!`: the unit-bin
and wide-bin conventions disagree, and their crossover moves right as
`th_discr` grows. `th_discr` stays at 800, the flat spot.

**The r direction is now the ceiling.** At 1e8 segments the binning-induced
spread is 2–8 sigma while the quadrature contributes 2e-6. Reconciling the two
bin conventions is the next thing worth fixing, and it is a modelling question,
not a quadrature one.
