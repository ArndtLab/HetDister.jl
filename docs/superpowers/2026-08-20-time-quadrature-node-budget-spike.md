# Spike: can a different time-quadrature map recover the 4x node budget?

Date: 2026-08-20 · Branch: `performance` · Status: measurement complete, no `src/` change

## Question

PANEL-ALG removed the kinks but needs ~4x the nodes of GLOBAL-ALG to reach the same
accuracy in Poisson units. The node counts were calibrated against `firstorder`,
which cannot see the transition-operator integrals. Does a different (still
TN-independent) node map do better **after the full r convolution sweep**, which is
what actually matters?

## Answer

**No — and no map can.** Every node map tested converges at exactly **O(1/N)** in the
total node count, differing only in the constant, and the constant varies by less than
a factor of 2 among the sensible candidates. The 4x is not a defect of PANEL-ALG; it is
the cost of Nystrom-discretising the transition operator, whose kernel has a **corner at
`t' = t` that moves with the row**. No fixed shared node set can resolve a corner whose
location changes for every row.

Removing that corner — splitting the quadrature at `t'=t*` — takes the row integral from
`5.4e-4` to `3.2e-15` at m=32, i.e. it restores machine precision at 1/32 of the nodes.
That, not the map, is where the 4x lives.

## Method

Throwaway node rules driven through a copy of `fusedsweep!` that takes `(ts, om)` instead
of calling `timenodes!`. The copy was verified bit-identical to `fusedsweep!` on the
shipped grid (max relative difference **0.0** across 200 bins).

Scoring: expected bin counts `w_i = ys_i * dr_i`, Poisson sigma
`|w_i - w_i^ref| / sqrt(w_i^ref)`, against a **single common reference** shared by all
arms — `PANEL m=16384, mtail=8192` (73 728 nodes). Earlier σ numbers in the design doc
scored each scheme against *its own* high-resolution limit, which cannot rank two
schemes against each other.

Reference validated: with the finite panels held fixed, the tail is converged well before
the reference setting (`mtail` 2048 vs 16384: **0.027 σ**, 0/800 bins > 1σ), so the
reference's residual is not tail-limited. All arms below use `mtail = 2048` so the tail
is not the discriminant.

Histories: `TNFIT` (the fitted 5-epoch real history), `TNSTALL` (N3 at the 1e8 bound),
`TNSTAT` (K=1), plus the two adversarial 3-epoch cases. 800 bins, lo=1, hi=3e6,
mu=1e-8, rho=2e-8.

## Results: accuracy vs total nodes

`TNFIT`, 800 bins, max Poisson sigma against the common reference:

| rule | nodes | max sigma | | nodes | max sigma |
|---|---:|---:|---|---:|---:|
| PANEL-ALG (shipped) m=64 | 2304 | 0.872 | m=256 | 3072 | 0.244 |
| PANEL-ALG m=128 | 2560 | 0.456 | m=512 | 4096 | 0.138 |
| UNIFORM composite 8x8 | 2304 | 0.802 | 16x16 | 3072 | 0.231 |
| GRADED (toward left end) 4x16 | 2304 | 1.932 | 8x32 | 3072 | 1.015 |
| LAG1 rescaled `t=T2*x/x_max` | 2304 | ~1e230 | | 3072 | NaN |
| LAG1 tau-scaled (r_max) m=64 | 2304 | 0.872 | m=256 | 3072 | NaN |
| GLOBAL-ALG frozen at exact TN | 1024 | 0.663 | 2048 | 0.325 |

`TNSTALL`: same ordering — PANEL 2.577 / 1.320 / 0.669 / 0.338 at 2304 / 2560 / 3072 /
4096 nodes; UNIFORM within 5%; GLOBAL-ALG frozen 0.606 at 1024 and 0.249 at 2048.

`TNSTAT` (K=1, tail only, the sequential fit's first step): `mtail` 384 / 768 / 1536 /
3072 / 6144 / 12288 gives 2.44 / 1.22 / 0.60 / 0.29 / 0.13 / 0.049 sigma — the same
O(1/N).

`TNBOUND` and `TNEMPTY` saturate at 0.240 sigma for every finite-panel setting; their
residual is the `mtail=2048` tail, i.e. the finite panels are already converged there.

### Reading

1. **Every arm is O(1/N).** Doubling the nodes halves the error, in all arms, on all
   histories. There is no geometric regime to recover.
2. **Arrangement barely matters.** Uniform composite sub-panels match single high-order
   panels within 5% at equal cost. Grading toward the left endpoint is 2x *worse* — it
   spends nodes where there is nothing to resolve.
3. **The boundary-layer story is wrong.** The panel that limits the sweep is the
   **widest** one, not the first. Holding the others at m=1024 and coarsening one to
   m=64: `TNFIT` gives 0.086 / 0.183 / 0.636 / 0.339 sigma for panels 1-4 — panel 3 is
   `[2069.6, 19856.5]`, the widest. `TNSTALL` gives 0.171 / 0.341 / 0.171 / 2.548 —
   panel 4, `[9495.9, 37798.0]`, again the widest.
4. **Frozen GLOBAL-ALG buys ~2.5x, not 4x** — and that is its *best case*, frozen at
   the exact TN being scored. A genuinely frozen reference drifting from the explored
   parameters can only be worse. It is a constant-factor win from allocating nodes by
   coalescent mass, i.e. exactly the adaptivity the panel scheme traded for smoothness;
   it does not change the rate. Not worth the clunkiness.

### The two Laguerre variants, specifically

- **Rescaled, `t = T2 * x_i/x_max` with folded weights `v_i e^{x_i}`**: diverges by ~230
  orders of magnitude. Gauss-Laguerre is exact for `poly(x) * e^{-x}` on `[0,inf)`; after
  rescaling, the integrand's physical decay in x is `e^{-x * T2/(x_max * tau)}`, which
  equals `e^{-x}` only if `T2 = x_max * tau`. Off that coincidence the huge folded
  weights at large `x_i` multiply an integrand that has not decayed, and the sum is
  garbage. This is not a tuning problem; the rule is only valid when the integrand
  carries the weight function, and here it does not.
- **tau-scaled, `t = T2*(1 - exp(-tau*x_i/T2))`** (the C-infinity way to get the absolute
  `r_max` scale without a `min()` clip): well-posed, smooth in TN, and it reproduces
  PANEL-ALG to 3 digits (0.8722 vs 0.8715). It also overflows past m ~ 150 (`e^{x_max}`
  with `x_max ~ 4m`). Confirms independently that panel 1 is not the bottleneck.

## Mechanism

From `sepkernel!` / `transition!` (`src/Spectra/SMCpIntegrals.jl:86-128`), with
`G(t) = N(t) + R(t) - N(0)e^{-C(t)}`:

```
K(t,t') = G(t)/N(t)                              for t' > t
        = [e^{-C(t)/2}/N(t)] * [e^{C(t')/2}G(t')] for t' < t
        + delta(t-t') * (t - G(t))                the exact atom
```

Both branches agree in value at `t' = t` (both give `G(t)/N(t)`), so the kernel is
**C0**. But the `t' > t` branch is *constant* in `t'` while the `t' < t` branch is not,
so the slope jumps: `K(t, .)` has a **corner at `t' = t`**, and its location is different
for every row.

Three measurements, all inside the real code path:

**(a) One row's integral**, `t* = 8000` (interior of the widest panel), relative error:

| m | shared-node rule | same rule, split at `t*` |
|---:|---:|---:|
| 32 | 5.389e-04 | 3.249e-15 |
| 64 | 7.482e-05 | 2.631e-15 |
| 128 | 1.651e-05 | 4.178e-15 |
| 256 | 7.203e-07 | 3.249e-15 |
| 1024 | 1.079e-07 | 3.714e-15 |

Splitting at the corner is the entire difference: 11 orders, at 1/32 of the nodes.

**(b) One real `transition!` apply**, scored against exact split-at-node evaluation
(`J1` is analytic, so "exact" is exact):

| m | nodes | max rel err | median |
|---:|---:|---:|---:|
| 32 | 2176 | 4.773e-02 | 2.111e-03 |
| 64 | 2304 | 2.429e-02 | 1.007e-03 |
| 128 | 2560 | 1.224e-02 | 5.032e-04 |
| 256 | 3072 | 6.141e-03 | 2.515e-04 |

Exactly O(1/m) — the same rate the full sweep shows.

**(c) The full sweep** is O(1/N) for every map (table above).

### What was tried and did not work

Two attempts to isolate causality at the *sweep* level by substituting a corner-free
operator (a rank-1 smooth surrogate, and the same operator with the corner pinned to a
fixed panel boundary) both made the Picard iteration diverge (NaN at every m). They are
reported as failed diagnostics; the causal evidence rests on (a) and (b), which are
direct measurements of the real operator rather than surrogates.

## Recommendation

- **Drop the map search.** PANEL-ALG-ALL, graded meshes, rescaled Laguerre and frozen
  GLOBAL-ALG have all been measured; none changes the rate, and the best constant
  available (frozen GLOBAL-ALG, best case) is 2.5x at the cost of reintroducing a
  TN-dependent grid. Paying the 4x in runtime, as you suggested, is the right call
  *unless* the diagonal is addressed.
- **The lever is the corner, not the nodes.** The row-level result says a rule that
  integrates `[0, t_j]` and `[t_j, inf)` with weights correct for those sub-intervals is
  spectrally accurate. Done naively that is O(n^2) per apply and destroys the
  semiseparable O(n) structure. But the corner always sits *at a node*, so with composite
  sub-panels of `msub` nodes the correction is confined to the single sub-panel
  containing `t_j`: exact prefix sums over whole sub-panels, plus a per-row partial
  integral of the local interpolant over `[a_p, t_j]`, which is a fixed `msub x msub`
  matrix applied per row — O(n * msub^2), near-linear for small `msub`. Worth its own
  design pass.
- **K=1 is tail-limited and equally O(1/N)** (2.44 sigma at the shipped `mtail=384`).
  The sequential fit starts there, so the same fix matters for the first step.

## Not touched

`src/` is unchanged (`git status --porcelain src/` empty). No MLE runs, no optimizer
flatness exploration. All experiment code is throwaway, in the session scratchpad.

---

# Follow-up: prototype of the diagonal correction

Built and measured the same day. Still throwaway; `src/` unchanged.

## What was built

Composite node rule (`nfin` equal sub-panels per finite epoch, `ntail` sub-panels in
the tail's `z` variable, `msub` Gauss-Legendre nodes each — all counts fixed, so every
node stays affine in TN), plus a transition apply that treats the partial sub-panel
containing the row by integrating the local interpolant:

```
Lpart[p,i] = int_{-1}^{z_p} l_i(z) dz          # fixed msub x msub, Legendre basis
lower partial for row p:  h_S * sum_i Lpart[p,i] * x_i
upper partial for row p:  (whole-panel sum) - (lower partial)
```

Cost is **O(n * msub)** — one length-`msub` dot product per row, because the cut is
always at a node. (The earlier O(n*msub^2) estimate was wrong.)

## Binning control

Run at two binnings, because geometric bins carry their own r-direction error:

- unit bins, `r in [1,8000]`, where the r direction is essentially exact (`w <= 1` takes
  the exact branch; only `MJ` held constant over a width-1 bin remains)
- the production adaptive grid, 800 bins, `r in [1,3e6]`

Both give the same rate and the same ranking; the geometric grid inflates absolute sigma
by roughly 10x. Nothing was being masked.

## Results

Max Poisson sigma vs a common reference, `TNFIT`:

| nodes | unit bins, plain | unit bins, corrected | 800 bins, plain | 800 bins, corrected |
|---:|---:|---:|---:|---:|
| 384 | 0.1803 | 0.000000 | 1.8080 | 0.000016 |
| 768 | 0.0911 | 0.000000 | 0.9166 | 0.000000 |
| 1536 | 0.0458 | 0.000000 | 0.4616 | 0.000000 |
| 3072 | 0.0237 | 0.000000 | 0.2390 | 0.000000 |
| 6144 | 0.0118 | 0.000000 | 0.1197 | 0.000000 |

The corrected sweep is converged at **384 nodes**, the smallest setting tried. The plain
sweep stays O(1/N).

One apply, against exact split-at-node evaluation: corrected gives 6.4e-11 at 256 finite
nodes and 5.9e-15 at 1024, against 2.8e-2 / 7.2e-3 plain.

**Same operator?** Yes. Scored against an independent *plain* 24 576-node reference, the
corrected sweep gives **0.02998 sigma at 384, 768 and 3072 nodes** — identical to four
digits, i.e. that residual is the plain reference's own error, not the corrected rule's.
The plain rule at 6144 nodes gives 0.08978 against the same reference, consistent with
1/N.

## Cost

800 bins, one sweep, single thread, `TNFIT`:

| configuration | nodes | time | error |
|---|---:|---:|---:|
| shipped `fusedsweep!`, m=64 mtail=384 (default) | 640 | 0.069 s | ~3 sigma |
| shipped `fusedsweep!`, m=256 mtail=1536 (spec's "match old fidelity") | 2560 | 0.280 s | ~0.25 sigma |
| corrected apply, comp 4x8 tail 32 | 384 | 0.097 s | converged |
| corrected apply, comp 16x16 tail 128 | 3072 | 1.001 s | converged |

The correction costs ~2x per node at `msub = 8`, but needs ~7x fewer nodes, so against
the setting the spec says is actually required it is **~3x faster and more accurate**.
The 4x node penalty is not merely recovered; it inverts.

## What this prototype does NOT establish

- **Float64 only.** No ForwardDiff run. `Lpart` is constant and the scale factors are
  affine, so it should be AD-clean, but that is untested.
- **No smoothness/kink testing.** The C1 invariant that Tasks 1-5 bought is the whole
  point of the branch; sub-panel counts are fixed and `Lpart` is TN-independent, so the
  invariant should survive, but `test/time_quadrature.jl` must be run against it.
- **The tail handling is a hack.** `e^{(C_k-C_j)/2}` overflows on sub-panels spanning a
  large dC, which the algebraic tail does (dC ~ 1e5); the prototype corrects only panels
  with dC < 40 and falls back to the shipped treatment elsewhere. It still converges at
  384 nodes, so tail rows were never the bottleneck — but a real implementation should
  sub-panel the tail in `u`, where d(C/2) = du is bounded by construction.
- **One history, one binning pair, one thread.** `TNFIT` only. `prordn!` has the same
  structure and would need the same treatment.
- **The inner loop allocates** (`EE` per sub-panel) and has not been optimised.
