# Calibration: node budget and theory binning after the corner correction

Date: 2026-08-21 · Branch: `performance` · Instrument: `bench/calibrate_quadrature.jl`

Follows `specs/2026-08-21-diagonal-corner-correction-design.md`. All numbers are
max Poisson sigma over the 200 observed bins spanning r ∈ [1, 5e6], model
weights renormalised to a stated total count `Ntot`, μ = 1e-8.

Histories: `TNFIT` (the fitted 5-epoch real history), `TNSTAT` (K=1),
`TNSTALL` (N at the 1e8 upper bound), `TNBOUND` (T at the 10 floor), `TNEMPTY`
(a near-empty epoch). α = ρ/(μ+ρ) ∈ {0.5, 0.667, 0.8}. `th_discr` ∈ {400, 800,
1600}, `Ntot` ∈ {1e5 … 1e8}. 1260 scored arms.

## Method

Two questions turn out to be **independent**, and the sweep separates them:

- **Node budget** — arms are scored against a node-converged reference *on the
  same theory grid*, so the r-direction error cancels exactly and only the
  quadrature is measured.
- **Theory binning** — reported as successive differences, because there is no
  converged limit to score against (see below).

The arm to beat is the pre-panel production setting: the old global
`tolegendre`/`tolaguerre` map at 800 nodes, `th_discr` = 800, reconstructed
verbatim from `5987c16^` so that both arms meet the same reference.

## Node budget: the quadrature is no longer a limiting factor

Worst sigma over every history, α and `th_discr`:

| msub | nfin | ntail | nodes (K=1 … K=5) | Ntot=1e5 | 1e6 | 1e7 | 1e8 |
|---:|---:|---:|---|---:|---:|---:|---:|
| **baseline** old global map | | | 800 | 0.690 | 2.181 | 6.898 | **21.812** |
| 6 | 8 | 12 | 222 … 438 | 3.5e-4 | 1.1e-3 | 3.5e-3 | 1.1e-2 |
| 6 | 12 | 16 | 246 … 558 | 9.0e-6 | 2.9e-5 | 9.0e-5 | 2.8e-4 |
| 8 | 8 | 12 | 296 … 584 | 5.2e-6 | 1.6e-5 | 5.2e-5 | 3.6e-4 |
| **8** | **12** | **16** | **328 … 744** | **6.7e-9** | **2.1e-8** | **6.7e-8** | **2.1e-6** |
| 8 | 16 | 16 | 328 … 872 | 8.0e-11 | 2.5e-10 | 8.0e-10 | 2.5e-9 |
| 10 | 12 | 16 | 410 … 930 | 3.0e-11 | 9.6e-11 | 3.0e-10 | 1.4e-8 |

The shipped defaults sit **seven orders** below the old map at every count, with
**fewer** nodes (744 at K=5 against 800). Per history at the shipped defaults,
worst case over α and `th_discr`:

| history | Ntot=1e5 | 1e6 | 1e7 | 1e8 | baseline at 1e8 |
|---|---:|---:|---:|---:|---:|
| TNFIT   | 6.7e-9 | 2.1e-8 | 6.7e-8 | 2.1e-7 | 16.2 |
| TNSTAT  | 6.5e-8 | 2.1e-7 | 6.5e-7 | 2.1e-6 | 18.4 |
| TNSTALL | 2.8e-8 | 8.8e-8 | 2.8e-7 | 8.8e-7 | 21.8 |
| TNBOUND | 6.5e-8 | 2.1e-7 | 6.5e-7 | 2.1e-6 | 18.4 |
| TNEMPTY | 3.1e-8 | 9.7e-8 | 3.1e-7 | 9.7e-7 | 20.3 |

### The node counts do not need to depend on `Ntot`

This was the thing to calibrate, and the calibrated answer is a constant.
Poisson sigma scales as √Ntot, so holding a target of 0.01 sigma from the
worst measured 2.1e-6 at Ntot = 1e8 would allow Ntot up to ~2e16 segments.
The dependence exists but is vacuous over any reachable dataset, because the
corrected rule converges **geometrically** in `msub` and `nfin` while the
requirement only tightens as √Ntot. Shipping a `getnodes(mu, rho, ntot)` that
returns the same triple everywhere would be a knob pretending to be a
calibration.

`(msub, nfin, ntail) = (8, 12, 16)` is kept rather than the ~20% cheaper
`(8, 8, 12)`: the leaner setting is still 30x inside a 0.01-sigma target, but it
misses the analytic `firstorder` anchor (1.1e-5 relative against the required
1e-6), and `nfin = 12` is what brings all five histories to 2.7e-8 there.

## Theory binning: no convergent limit, and it is not the quadrature

Successive differences in `th_discr` at the shipped node counts, `TNFIT`,
α = 0.667:

| Ntot | 200→400 | 400→800 | 800→1600 | 1600→3200 | 3200→6400 |
|---|---:|---:|---:|---:|---:|
| 1e5 | 0.132 | 0.052 | 0.067 | 0.085 | 0.104 |
| 1e6 | 0.417 | 0.165 | 0.211 | 0.267 | 0.330 |
| 1e7 | 1.318 | 0.521 | 0.667 | 0.845 | 1.042 |
| 1e8 | 4.167 | 1.648 | 2.108 | 2.673 | 3.295 |

The differences **stop shrinking after 400→800 and then grow**. Every history
behaves the same way; `TNSTAT` is worst (4.12 at 400→800 rising to 8.20 at
3200→6400, Ntot = 1e8).

Three facts locate this away from the time quadrature:

1. **The old map does it identically.** Same sweep with the pre-panel global
   map at 1600 nodes: 0.861 / 0.643 / 0.836 / 1.034 for 400→800 … 3200→6400,
   against the corrected rule's 0.868 / 0.645 / 0.841 / 1.041. It is pre-existing.
2. **It survives node refinement.** At `th_discr` = 6400 held fixed, the shipped
   grid scores 0.0038 sigma against the 1884-node reference — the quadrature is
   converged there.
3. **It lives in the unit-bin branch.** Successive differences at Ntot = 1e7,
   `TNFIT`, α = 0.667:

   ```
   lo=1    hi=5e6   (unit bins present)              0.521  0.667  0.845
   lo=2000 hi=5e6   (every fine bin wider than 1bp)  0.171  0.072  0.036
   lo=1    hi=2000  (almost every fine bin unit)     0.829  0.963    -
   ```

   With no unit bins the theory binning converges cleanly at O(1/th_discr) —
   the differences halve at each doubling. With unit bins present it does not
   converge at all.

The mechanism is the `w <= 1` branch of `fusedsweep!` (§6 of the fused spec):
a unit bin reports the count of segments of exactly `edges[i]` over the full
width, a wide bin reports a density at the geometric midpoint over the partial
width. The crossover between the two conventions moves right as `th_discr`
grows, so refining the theory grid keeps re-deciding which convention applies
over a widening range of r, and the two do not agree to O(1/th_discr).

### Recommendation

- **Keep `th_discr = 800`.** It sits in the flat spot; 1600 and beyond are
  measurably *worse*, not better, and cost proportionally more.
- The r direction is now the accuracy ceiling for large datasets. At
  Ntot = 1e8 the binning-induced spread is 2–8 sigma while the quadrature
  contributes 2e-6. **This is the next thing worth fixing**, and it is a
  modelling question (reconciling the unit-bin and wide-bin conventions), not
  a quadrature one.
- `getnpicard`'s own calibration is stated at `nbins = 800` and is untouched by
  this, since `th_discr` stays there. Measured directly: at `th_discr` = 6400
  the shipped grid scores 0.0038 sigma with `npicard = 3` and 0.0003 with
  `npicard = 6` against a `npicard = 16` reference, so the Picard count is not
  the binding constraint either.

## Reproducing

```
julia --project=. bench/calibrate_quadrature.jl          # ~13 min
julia --project=. bench/calibrate_quadrature.jl --quick  # ~40 s
```

The raw output of the run the tables above distil is kept verbatim in
`2026-08-21-calibration-raw-output.log` (1260 scored arms), so every number here can
be traced without re-running the sweep.
