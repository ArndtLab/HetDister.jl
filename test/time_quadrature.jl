using IBSpector
using IBSpector.Spectra
using Test
using HistogramBinnings
using StatsBase
using IBSpector.Spectra.PreallocationTools

const SMCp = IBSpector.Spectra.SMCpIntegrals
using IBSpector.Spectra.SMCpIntegrals: TimeGrid, timenodes!, ndt

@testset "TimeGrid" begin
    g = TimeGrid(5; m = 48, mtail = 64)
    @test g.K == 5 && g.m == 48 && g.mtail == 64
    @test ndt(g) == 4 * 48 + 64
    # a one-epoch history has no finite panels, only the tail
    @test ndt(TimeGrid(1; m = 48, mtail = 32)) == 32

    # Gauss-Legendre weights on (-1,1) sum to the interval length
    @test sum(g.wleg) ≈ 2 rtol = 1e-12
    @test length(g.zleg) == 48 && all(-1 .< g.zleg .< 1)

    # tail rule: u = (1+z)/(1-z) with weight w*2/(1-z)^2, so it integrates
    # int_0^inf e^{-u} u^p du = p! directly (no exponential folding)
    for p in 0:3
        @test sum(g.wtail .* g.utail .^ p .* exp.(-g.utail)) ≈ factorial(p) rtol = 1e-9
    end
    @test all(isfinite, g.wtail) && all(g.wtail .> 0)
    @test issorted(g.utail) && g.utail[1] > 0
end

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
    # single epoch: the whole domain is the algebraic-map tail panel, and
    # int_0^inf exp(-t/2N) dt = 2N  is reproduced by the tail weights
    N = 12345.0
    TN = [3.0e9, N]
    g = TimeGrid(1; m = 8, mtail = 64)
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

@testset "sweep runs on the panel grid and matches the order loop" begin
    mu, rho = 1.0e-8, 2.0e-8
    TN = TNSTALL
    K = length(TN) ÷ 2
    ev = IBSpector.CustomEdgeVector(lo = 1, hi = 30_000, nbins = 120)
    edges = collect(Float64, ev); rs = collect(Float64, midpoints(ev))
    grid = TimeGrid(K; m = 64, mtail = 64)

    bagf = IntegralArrays(60, grid, length(rs), Val{length(TN)})
    bago = IntegralArrays(60, grid, length(rs), Val{length(TN)})
    @test bagf.n_dt == ndt(grid)

    # npicard = 6 (the design-target rule for this alpha) only contracts the
    # error to ~3e-3 by the fixed exponential-Euler bin count here; np = 25
    # drives the Picard iteration itself to ~2.5e-8, well under the 1e-6
    # bound, confirming fused and order-loop converge to the same integral.
    SMCp.fusedsweep!(bagf, rs, edges, mu, rho, TN; npicard = 25)
    yf = copy(get_tmp(bagf.ys, Float64))
    SMCp.prordn!(bago, rs, edges, mu + rho, TN)
    Spectra.mldsmcp!(bago, 1:60, mu, rho, TN)
    yo = copy(get_tmp(bago.ys, Float64))

    @test all(isfinite, yf) && all(yf .> 0)
    # converged Picard vs converged order loop: same integral, same grid
    @test maximum(abs.(yf .- yo) ./ yo) < 1e-6
end

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
    rs = collect(Float64, midpoints(ev))

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
    FLOOR = 1e-10
    @test errs[2] < errs[1] / 10 || errs[2] < FLOOR
    @test errs[3] < errs[2] / 10 || errs[3] < FLOOR
end

@testset "FitOptions carries per-panel node counts" begin
    fop = FitOptions(3.0e9, 100_000, 1.0e-8, 2.0e-8; nepochs = 5)
    @test fop.mpanel > 0 && fop.mtail > 0
    @test !hasproperty(fop, :ndt)   # renamed, so a stale `ndt = 800` cannot pass silently
    g = TimeGrid(fop.nepochs; m = fop.mpanel, mtail = fop.mtail)
    @test ndt(g) == (fop.nepochs - 1) * fop.mpanel + fop.mtail
end
