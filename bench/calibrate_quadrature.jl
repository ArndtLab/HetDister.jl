# Calibration sweep for the time-quadrature node counts and the theory-side
# binning `th_discr`, against total histogram counts.
#
# Run from the package root:
#
#     julia --project=. bench/calibrate_quadrature.jl [--quick]
#
# Two knobs are being set, both forbidden from depending on TN (same discipline
# as getnpicard):
#
#   * th_discr -- the theory-side binning demoinfer evaluates the model on
#     before coarsening it onto the observed histogram
#   * (msub, nfin, ntail) -- the sub-panel counts of the TimeGrid
#
# Scoring. The discretisation bias dw_i scales with the total count Ntot, so the
# Poisson score dw_i/sqrt(w_i) scales as sqrt(Ntot): a rule that is adequate at
# 1e5 segments is not at 1e8. Every arm is therefore scored in Poisson sigma at
# a stated Ntot, against ONE common reference per history -- scoring each arm
# against its own high-resolution limit cannot rank two arms, which is the
# mistake the 2026-08-20 spike corrected.
#
# The arm to beat is the pre-panel production setting: the old global
# tolegendre/tolaguerre map at 800 nodes with theory binning 800, reconstructed
# here from commit 5987c16^ so that it is scored against the same reference.

using IBSpector
using IBSpector.Spectra
using IBSpector.Spectra.PreallocationTools
using IBSpector.Spectra.SMCpIntegrals: TimeGrid, ndt, npanels, timenodes!,
    sepkernel!, transition!, pt, firstorder, getnpicard
using IBSpector.Spectra.CoalescentBase: getts, getns
using IBSpector: map_fine_to_coarse
using Printf

const SMCp = IBSpector.Spectra.SMCpIntegrals

const HISTORIES = [
    "TNFIT"   => [3.003e9, 12388.8, 28302.1, 6975.85, 6214.37,
                  9.99002e7, 3066.44, 2754.27, 215.101, 21782.5],
    "TNSTAT"  => [3.0e9, 20000.0],
    "TNSTALL" => [3.0e9, 12000.0, 5000.0, 1.0e8, 3000.0, 7000.0],
    "TNBOUND" => [3.0e9, 12000.0, 10.0, 3000.0, 10.0, 20000.0],
    "TNEMPTY" => [3.0e9, 15000.0, 6000.0, 9.9e7, 4000.0, 8000.0],
]

# ------------------------------------------------------- the pre-panel arm ----
# Verbatim from 5987c16^: the global coalescent map, whose epoch is selected by
# searching on the node's own value. That search is what made the likelihood
# non-C1; it is reproduced here only to score the accuracy it delivered.
function tolaguerre_old(z, TN)
    epoch = 1
    ce = 0.0
    ae = 1 / 2getns(TN, epoch)
    t = (z - ce) / ae
    while epoch < length(TN) ÷ 2 && t > getts(TN, epoch + 1)
        epoch += 1
        ce += (getts(TN, epoch) - getts(TN, epoch - 1)) * ae
        ae = 1 / 2getns(TN, epoch)
        t = (z - ce + ae * getts(TN, epoch)) / ae
    end
    return t, 1 / ae
end

function tolegendre_old(z, TN)
    y = -1 - 2 / (z - 1)
    dy = 2 / (z - 1)^2
    t, dt = tolaguerre_old(y, TN)
    return t, dt * dy
end

# Pre-correction sepkernel!/transition!: node-to-node prefix sums split at the
# row index, i.e. whole-grid weights used on a partial interval.
function sepkernel_old!(Phi, dgn, Gc, Ninv, dC, ts, TN)
    n = length(ts)
    n0 = SMCp.Nt(0, TN)
    cprev = 0.0
    for j in 1:n
        t = ts[j]
        c = SMCp.cumcr(0, t, TN)
        nt = SMCp.Nt(t, TN)
        g = nt + SMCp.margrecomb(t, TN) - n0 * exp(-c)
        Gc[j] = max(g, 0.0); Phi[j] = Gc[j] / nt
        dgn[j] = max(t - g, 0.0); Ninv[j] = 1 / nt
        j > 1 && (dC[j-1] = c - cprev)
        cprev = c
    end
    dC[n] = 0.0
    return nothing
end

function transition_old!(out, x, Phi, dgn, Gc, Ninv, dC, om, n)
    sfx = 0.0
    for j in n:-1:1
        out[j] = Phi[j] * sfx
        sfx += x[j] * om[j]
    end
    st = 0.0
    for j in 1:n
        out[j] += st * Ninv[j] + dgn[j] * x[j]
        st = exp(-dC[j] / 2) * (st + Gc[j] * x[j] * om[j])
    end
    return nothing
end

# The fused sweep on the old global map, so the two arms differ only in the
# quadrature: same exponential-Euler / Picard structure as fusedsweep!.
function sweep_old(rs, edges, mu, rho, TN, nnodes; npicard = 0)
    n = nnodes; nrs = length(rs)
    zs, wt = SMCp.gausslegendre(n)
    ts = zeros(n); om = zeros(n); qs = zeros(n)
    for j in 1:n
        t, dt = tolegendre_old(zs[j], TN)
        ts[j] = t; om[j] = wt[j] * dt; qs[j] = pt(t, TN)
    end
    Phi = zeros(n); dgn = zeros(n); Gc = zeros(n); Ninv = zeros(n); dC = zeros(n)
    sepkernel_old!(Phi, dgn, Gc, Ninv, dC, ts, TN)
    rate = mu + rho; alpha = rho / rate
    np = npicard > 0 ? npicard : getnpicard(mu, rho)
    A = zeros(n); MJ = zeros(n); Jf = zeros(n); J1 = zeros(n); ys = zeros(nrs)
    scale = 2 * mu * TN[1] * (mu / rate)
    for i in 1:nrs
        w = edges[i+1] - edges[i]
        wi = w <= 1 ? w : rs[i] - edges[i]
        for j in 1:n
            J1[j] = rate * exp(-2rate * rs[i] * ts[j]) * qs[j]
        end
        for _ in 1:np
            for j in 1:n
                t = ts[j]
                Jf[j] = J1[j] + A[j] * exp(-2rate * (rs[i] - edges[i]) * t) +
                        alpha * MJ[j] * (-expm1(-2rate * wi * t)) / 2t
            end
            transition_old!(MJ, Jf, Phi, dgn, Gc, Ninv, dC, om, n)
        end
        s = 0.0
        for j in 1:n
            t = ts[j]
            jc = A[j] * exp(-2rate * (rs[i] - edges[i]) * t) +
                 alpha * MJ[j] * (-expm1(-2rate * wi * t)) / 2t
            s += jc * 2 * t * om[j]
            A[j] = exp(-2rate * w * t) * A[j] +
                   alpha * MJ[j] * (-expm1(-2rate * w * t)) / 2t
        end
        ys[i] = (firstorder(rs[i], rate, TN) + s) * scale
    end
    return ys
end

# ------------------------------------------------------------- the new arm ----
function sweep_new(rs, edges, mu, rho, TN, grid; npicard = 0)
    bag = IntegralArrays(grid, length(rs), Val{length(TN)})
    SMCp.fusedsweep!(bag, rs, edges, mu, rho, TN; npicard)
    copy(get_tmp(bag.ys, Float64))
end

# ------------------------------------------------------------------ scoring --
# Model weights on `th_discr` theory bins, coarsened onto the observed binning
# and renormalised to a total count of Ntot -- exactly demoinfer's path.
function obsweights(sweepf, TN, mu, rho, lo, hi, th_discr, obsedges)
    ev = IBSpector.CustomEdgeVector(; lo, hi, nbins = th_discr)
    fine = collect(Float64, ev)
    rs = collect(Float64, IBSpector.midpoints(ev))
    wfine = sweepf(rs, fine, mu, rho, TN) .* diff(fine)
    w = map_fine_to_coarse(wfine, fine, obsedges)
    return w ./ sum(w)
end

maxsigma(w, wref) = maximum(abs.(w .- wref) ./ sqrt.(wref))

# ---------------------------------------------------------------- the sweep --
# Two questions, deliberately separated because they turn out to be independent:
#
#   (A) node budget -- at a FIXED theory binning, how many nodes does the
#       corrected rule need, against the old global map at 800? Scored against a
#       node-converged reference on the same theory grid, so the r-direction
#       error cancels exactly and only the quadrature is measured.
#
#   (B) theory binning -- how does the answer move with th_discr? Measured as
#       successive differences, since there is no converged limit to score
#       against (see the report).
const QUICK = "--quick" in ARGS

function nodebudget(mu, lo, hi, obsedges, alphas, ntots, thgrid, nodegrid)
    for alpha in alphas
        rho = mu * alpha / (1 - alpha)
        for (name, TN) in HISTORIES
            K = length(TN) ÷ 2
            gref = TimeGrid(K; msub = 12, nfin = 24, ntail = 32)
            for th in thgrid
                refw = obsweights((rs, e, m, r, t) -> sweep_new(rs, e, m, r, t, gref),
                                  TN, mu, rho, lo, hi, th, obsedges)
                basew = obsweights((rs, e, m, r, t) -> sweep_old(rs, e, m, r, t, 800),
                                   TN, mu, rho, lo, hi, th, obsedges)
                arms = Tuple{Int,Int,Int,Int,Vector{Float64}}[]
                for (ms, nf, nt) in nodegrid
                    g = TimeGrid(K; msub = ms, nfin = nf, ntail = nt)
                    push!(arms, (ms, nf, nt, ndt(g),
                        obsweights((rs, e, m, r, t) -> sweep_new(rs, e, m, r, t, g),
                                   TN, mu, rho, lo, hi, th, obsedges)))
                end
                @printf("\n== %s  alpha=%.3f  th_discr=%d  (reference: %d nodes, same th)\n",
                        name, alpha, th, ndt(gref))
                for Ntot in ntots
                    base = maxsigma(basew .* Ntot, refw .* Ntot)
                    @printf("  Ntot=%.0e  BASELINE old global map, 800 nodes: %10.4f sigma\n",
                            Ntot, base)
                    for (ms, nf, nt, n, w) in arms
                        sg = maxsigma(w .* Ntot, refw .* Ntot)
                        @printf("      msub=%-3d nfin=%-3d ntail=%-3d n=%-5d %11.3e sigma  %5.0fx better\n",
                                ms, nf, nt, n, sg, base / max(sg, 1e-300))
                    end
                end
            end
        end
    end
end

function binningsweep(mu, lo, hi, obsedges, alphas, ntots, thgrid)
    for alpha in alphas
        rho = mu * alpha / (1 - alpha)
        for (name, TN) in HISTORIES
            K = length(TN) ÷ 2
            g = TimeGrid(K)
            ws = Dict(th => obsweights((rs, e, m, r, t) -> sweep_new(rs, e, m, r, t, g),
                                       TN, mu, rho, lo, hi, th, obsedges) for th in thgrid)
            @printf("\n== %s  alpha=%.3f  theory-binning successive differences (shipped grid, %d nodes)\n",
                    name, alpha, ndt(g))
            for Ntot in ntots
                line = join([@sprintf("%d->%d: %7.4f", thgrid[i], thgrid[i+1],
                                maxsigma(ws[thgrid[i]] .* Ntot, ws[thgrid[i+1]] .* Ntot))
                             for i in 1:length(thgrid)-1], "  ")
                @printf("  Ntot=%.0e  %s\n", Ntot, line)
            end
        end
    end
end

function main()
    mu = 1.0e-8
    lo, hi = 1, 5_000_000
    obsedges = collect(Float64, IBSpector.CustomEdgeVector(; lo, hi, nbins = 200))

    alphas   = QUICK ? (2 / 3,) : (0.5, 2 / 3, 0.8)
    ntots    = QUICK ? (1e7,) : (1e5, 1e6, 1e7, 1e8)
    thgrid   = QUICK ? (800,) : (400, 800, 1600)
    nodegrid = QUICK ? ((8, 8, 16), (8, 12, 16)) :
        ((6, 8, 12), (6, 12, 16), (8, 8, 12), (8, 8, 16), (8, 12, 16),
         (8, 16, 16), (10, 12, 16))

    println("#### (A) node budget at fixed theory binning")
    nodebudget(mu, lo, hi, obsedges, alphas, ntots, thgrid, nodegrid)
    println("\n#### (B) theory-binning sensitivity, shipped node counts")
    binningsweep(mu, lo, hi, obsedges, alphas, ntots,
                 QUICK ? (400, 800, 1600) : (200, 400, 800, 1600, 3200, 6400))
end

"--noexec" in ARGS || main()
