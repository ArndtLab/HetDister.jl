# Final fix-wave report — panel time quadrature review

Applied all items from the final whole-branch review in one pass, in place on
branch `performance` (no worktree, no commit created, no push).

## I1 — missing panel-width guard

`src/Spectra/SMCpIntegrals.jl`, `timenodes!`: added a strict-monotonicity
check inside the `for k in 1:K-1` loop (`b > a || throw(ArgumentError(...))`,
naming epoch `k` and the two boundary times `a`/`b`) and a strict-positivity
check on the tail's `N_K` (`NK > 0 || throw(ArgumentError(...))`), both before
any node is written. Uses `throw(ArgumentError(...))`, matching the
`CoalescentBase.getts`/`getns` convention, not `@assert`.

Added testset `"timenodes! rejects a non-monotone TN"` to
`test/time_quadrature.jl` (appended at the end; `using` lines untouched at
the top) reproducing both vectors from the review verbatim — the zero-width
panel (`[3e9, 1e4, 0.0, 2e4, 100.0, 3e4]`) and the descending panel
(`[3e9, 1e4, -50.0, 2e4, 100.0, 3e4]`) — asserting `@test_throws
ArgumentError` for each, plus a valid monotone `TN` that does not throw.

## I2 — stale docstrings

- `src/utils.jl:271`, `src/IBSpector.jl:59`, `src/Spectra/Spectra.jl:30`:
  "Gauss-Laguerre nodes on the final semi-infinite panel" replaced with
  "Gauss-Legendre nodes under the algebraic map on the final semi-infinite
  panel."
- `src/Spectra/SMCpIntegrals.jl:131`: `TimeGrid` docstring header changed
  from `TimeGrid(K; m = 48, mtail = 48)` to `TimeGrid(K; m = 64, mtail =
  384)`, matching the constructor at line 159 (constructor itself untouched).

## Minors

a) `test/time_quadrature.jl`: `@test issorted(ts)` → `@test all(diff(ts) .>
   0)` (strict).
b) exactly-affine testset: now also builds `Om` alongside `T` and asserts
   exact linearity of `om` the same way it does for `ts`.
c) `getnpicard` docstring: "at the production binning (`nbins = ndt = 800`
   ...)" reworded to "at the production binning (`nbins = 800` ...)",
   dropping the stale `ndt` coupling; logic untouched.
d) `test/smcp_fused.jl` (found at line 177, not 517 — branch has since
   diverged from the reviewed diff's line numbers): stale comment "mldsmcp
   ignores its `ndt` kwarg now" changed to "mldsmcp has no `ndt` kwarg any
   more."
e) Convergence-testset comment: measured actual output (m=8 → 1.27e-3, m=16
   → 1.49e-5, m=32 → 6.8e-12) via a standalone script running the exact same
   computation; comment corrected from "4.8e-2, 1.7e-5, 6.8e-12" to "1.27e-3,
   1.5e-5, 6.8e-12 (floor)".
f) `.gitignore`: added `/test/Manifest.toml`.

## Spec amendment

Added a new "### Open: shipped defaults under-resolve the transition
integrals (2026-08-20)" section to
`docs/superpowers/specs/2026-08-19-panel-time-quadrature-design.md`,
immediately after the existing "Known limitation" section, with the
measurement and calibration-target text verbatim as specified. `TimeGrid`
defaults (`m = 64, mtail = 384`) were NOT touched — confirmed still `function
TimeGrid(K::Int; m::Int = 64, mtail::Int = 384)` at
`src/Spectra/SMCpIntegrals.jl:159`.

## Verification

Ran once, blocking: `julia --project=. -e 'using Pkg; Pkg.test()'`.
All testsets passed, including the new
`"timenodes! rejects a non-monotone TN"` (3/3) and the updated
`"timenodes! is exactly affine in TN"` (146880/146880, now covering both
`ts` and `om`). Full suite: all pass, "IBSpector tests passed."

No commit was made (not requested); changes are in the working tree on
branch `performance`.
