using IBSpector
using IBSpector.Spectra
using IBSpector.Spectra.PreallocationTools
using IBSpector.Spectra.SMCpIntegrals: getnpicard, fusedsweep!, transition!, sepkernel!,
    TimeGrid, ndt, timenodes!
using HistogramBinnings
using StatsBase
using Test
using ForwardDiff

const SMCp = IBSpector.Spectra.SMCpIntegrals

# Production binning: log-spaced edges pushed up to distinct integers, then the
# geometric midpoint for wide bins and the lower edge for unit bins.
function prodgrid(nbins, hi)
    ev = IBSpector.CustomEdgeVector(lo = 1, hi = hi, nbins = nbins)
    collect(Float64, ev), collect(Float64, midpoints(ev))
end

# Order-loop reference, summed over orders with alpha and scaled exactly as
# mldsmcp! scales it. `order` must be large enough to be converged.
function orderref(rs, edges, mu, rho, grid, TN, order)
    rate = mu + rho
    alpha = rho / rate
    bag = IntegralArrays(order, grid, length(rs), Val{length(TN)})
    SMCp.prordn!(bag, rs, edges, rate, TN)
    res = get_tmp(bag.res, eltype(TN))
    scale = 2 * mu * TN[1] * (mu / rate)
    [sum(res[i, o] * alpha^(o - 1) for o in 1:order) * scale for i in eachindex(rs)]
end

# Raw fusedsweep! with freshly allocated Float64 buffers.
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
    grid = TimeGrid(length(TN) ÷ 2; m = 32, mtail = 32)
    n = ndt(grid)
    ts = zeros(n); om = zeros(n)
    timenodes!(ts, om, grid, TN)
    Phi = zeros(n); dgn = zeros(n); Gc = zeros(n)
    Ninv = zeros(n); dC = zeros(n)
    sepkernel!(Phi, dgn, Gc, Ninv, dC, ts, TN)

    x = abs.(randn(n)) .* 1e-6
    out = zeros(n)
    transition!(out, x, Phi, dgn, Gc, Ninv, dC, om, n)

    jprt = reshape(copy(x), n, 1)
    temp = zeros(1, n)
    transition!(temp, jprt, 1, Phi, dgn, Gc, Ninv, dC, om, n)
    @test out ≈ vec(temp[1, :])
end

@testset "fused sweep converges to the order loop under Picard" begin
    TN = [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0]
    mu = 1.25e-8
    grid = TimeGrid(length(TN) ÷ 2; m = 100, mtail = 100)
    edges, rs = prodgrid(200, 30_000)

    for ratio in (1.0, 4.0)
        rho = mu * ratio
        ref = orderref(rs, edges, mu, rho, grid, TN, 200)
        errs = [maximum(abs.(rawfused(rs, edges, mu, rho, grid, TN, np) .- ref) ./ abs.(ref))
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
    grid = TimeGrid(length(TN) ÷ 2; m = 64, mtail = 64)
    edges, rs = prodgrid(200, 30_000)
    for np in 1:4
        ys = rawfused(rs, edges, mu, rho, grid, TN, np)
        @test all(isfinite, ys)
        @test all(ys .> 0)
    end
end

@testset "bag wrapper matches the raw fusedsweep!" begin
    TN = [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0]
    mu = 1.25e-8
    grid = TimeGrid(length(TN) ÷ 2; m = 64, mtail = 64)
    edges, rs = prodgrid(200, 30_000)

    for ratio in (1.0, 4.0)
        rho = mu * ratio
        np = getnpicard(mu, rho)

        bag = IntegralArrays(10, grid, length(rs), Val{length(TN)})
        fusedsweep!(bag, rs, edges, mu, rho, TN)
        auto = copy(get_tmp(bag.ys, eltype(TN)))
        @test auto ≈ rawfused(rs, edges, mu, rho, grid, TN, np)

        # an explicit npicard overrides the rule
        fusedsweep!(bag, rs, edges, mu, rho, TN; npicard = 6)
        @test get_tmp(bag.ys, eltype(TN)) ≈ rawfused(rs, edges, mu, rho, grid, TN, 6)

        # calling twice with the same arguments must give the same answer:
        # A and MJ have to be reset, not carried between calls
        fusedsweep!(bag, rs, edges, mu, rho, TN)
        @test get_tmp(bag.ys, eltype(TN)) ≈ auto
    end
end

@testset "mldsmcp method keyword" begin
    TN = [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0]
    mu = 1.25e-8
    rho = 4mu
    ndt = 200
    edges, rs = prodgrid(200, 30_000)

    # mldsmcp now builds its own TimeGrid(length(TN)÷2) internally (with default
    # m/mtail) regardless of the `ndt` kwarg, which is accepted-but-unused for
    # now (Task 5 threads real per-panel counts through). Match that grid here
    # so the direct bag calls stay comparable.
    grid = TimeGrid(length(TN) ÷ 2)

    # :fused agrees with the direct bag call
    got = mldsmcp(rs, edges, mu, rho, TN; ndt = ndt, method = :fused)
    bag = IntegralArrays(10, grid, length(rs), Val{length(TN)})
    fusedsweep!(bag, rs, edges, mu, rho, TN)
    @test got ≈ get_tmp(bag.ys, eltype(TN))

    # :order reproduces the pre-change behaviour bit for bit
    order = 12
    want = mldsmcp(rs, edges, mu, rho, TN; order = order, ndt = ndt, method = :order)
    bag2 = IntegralArrays(order, grid, length(rs), Val{length(TN)})
    SMCp.prordn!(bag2, rs, edges, mu + rho, TN)
    mldsmcp!(bag2, 1:order, mu, rho, TN)
    @test want == get_tmp(bag2.ys, eltype(TN))

    # the mutating entry defaults to the fused sweep, like mldsmcp
    bag3 = IntegralArrays(order, grid, length(rs), Val{length(TN)})
    mldsmcp!(bag3, 1:order, rs, edges, mu, rho, TN)
    @test get_tmp(bag3.ys, eltype(TN)) == got

    # res is poisoned on the fused path so stale per-order reads are loud
    bag4 = IntegralArrays(order, grid, length(rs), Val{length(TN)})
    mldsmcp!(bag4, 1:order, rs, edges, mu, rho, TN; method = :fused)
    @test all(isnan, get_tmp(bag4.res, eltype(TN)))

    @test_throws ArgumentError mldsmcp(rs, edges, mu, rho, TN; ndt = ndt, method = :bogus)
end

# max |z| over bins that would survive adapt_histogram's tail threshold.
# mldsmcp ignores its `ndt` kwarg now (it always uses TimeGrid(length(TN)÷2)
# with default node counts), so the order-loop reference is built on the same
# default grid for an apples-to-apples comparison.
function maxz(rs, edges, mu, rho, TN; reford = 200, tailthr = 10)
    grid = TimeGrid(length(TN) ÷ 2)
    ref = orderref(rs, edges, mu, rho, grid, TN, reford)
    got = mldsmcp(rs, edges, mu, rho, TN; method = :fused)
    keep = findall(ref .> tailthr)
    @assert length(keep) > 100
    maximum(abs.(got[keep] .- ref[keep]) ./ sqrt.(ref[keep]))
end

@testset "fused error is far below Poisson noise at the production binning" begin
    TN = [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0]
    mu = 1.25e-8
    nbins = 800

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
        z = maxz(rs, edges, mu, rho, TN)
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
    ys = mldsmcp(rs, edges, mu, rho, TN; method = :fused)
    @test all(isfinite, ys)
    @test all(ys .> 0)
    @test maxz(rs, edges, mu, rho, TN) < 1e-2
end

@testset "fused error does not depend on the demography" begin
    mu = 1.25e-8
    rho = 4mu
    edges, rs = prodgrid(800, 10_000_000)
    for TN in ([3.0e9, 10000.0],
               [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0],
               [3.0e9, 20000.0, 60000.0, 8000.0, 8000.0, 16000.0, 1600.0, 2000.0, 400.0, 10000.0])
        @test maxz(rs, edges, mu, rho, TN) < 1e-2
    end
end

@testset "fused sweep is ForwardDiff-differentiable" begin
    mu = 1.25e-8
    rho = 4mu
    edges = collect(1.0:1.0:40.0)
    rs = collect(1.0:1.0:39.0)
    TN0 = [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0]
    grid = TimeGrid(length(TN0) ÷ 2; m = 32, mtail = 32)

    function total(TN)
        bag = IntegralArrays(4, grid, length(rs), Val{length(TN)})
        fusedsweep!(bag, rs, edges, mu, rho, TN)
        sum(get_tmp(bag.ys, eltype(TN)))
    end

    g = ForwardDiff.gradient(total, TN0)
    @test length(g) == length(TN0)
    @test all(isfinite, g)
    @test any(!iszero, g)

    # central differences on all live TN entries (indices 2-6): population
    # sizes (2, 4, 6) and epoch times (3, 5), the latter flowing through the
    # epoch-pinned panel construction in timenodes!.
    for k in 2:6
        h = 1e-3 * TN0[k]
        tp = copy(TN0); tp[k] += h
        tm = copy(TN0); tm[k] -= h
        fd = (total(tp) - total(tm)) / 2h
        @test isapprox(g[k], fd; rtol = 1e-4, atol = 1e-8 * abs(g[k]))
    end
end
