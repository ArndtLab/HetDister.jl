using IBSpector
using IBSpector.Spectra
using IBSpector.Spectra.PreallocationTools
using IBSpector.Spectra.SMCpIntegrals: getnpicard, fusedsweep!, transition!, sepkernel!
using HistogramBinnings
using StatsBase
using Test

const SMCp = IBSpector.Spectra.SMCpIntegrals

# Production binning: log-spaced edges pushed up to distinct integers, then the
# geometric midpoint for wide bins and the lower edge for unit bins.
function prodgrid(nbins, hi)
    ev = IBSpector.CustomEdgeVector(lo = 1, hi = hi, nbins = nbins)
    collect(Float64, ev), collect(Float64, midpoints(ev))
end

# Order-loop reference, summed over orders with alpha and scaled exactly as
# mldsmcp! scales it. `order` must be large enough to be converged.
function orderref(rs, edges, mu, rho, ndt, TN, order)
    rate = mu + rho
    alpha = rho / rate
    bag = IntegralArrays(order, ndt, length(rs), Val{length(TN)})
    SMCp.prordn!(bag, rs, edges, rate, TN)
    res = get_tmp(bag.res, eltype(TN))
    scale = 2 * mu * TN[1] * (mu / rate)
    [sum(res[i, o] * alpha^(o - 1) for o in 1:order) * scale for i in eachindex(rs)]
end

# Raw fusedsweep! with freshly allocated Float64 buffers.
function rawfused(rs, edges, mu, rho, ndt, TN, npicard)
    nrs = length(rs)
    zs, wt = SMCp.gausslegendre(ndt)
    v() = zeros(Float64, ndt)
    ys = zeros(Float64, nrs)
    fusedsweep!(ys, v(), v(), v(), v(), v(), v(), v(), v(), v(),
                v(), v(), v(), v(), v(),
                zs, wt, rs, edges, mu, rho, npicard, ndt, nrs, TN)
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
    ndt = 120
    zs, wt = SMCp.gausslegendre(ndt)
    ts = zeros(ndt); om = zeros(ndt)
    for j in 1:ndt
        t, dt = SMCp.tolegendre(zs[j], TN)
        ts[j] = t
        om[j] = wt[j] * dt
    end
    Phi = zeros(ndt); dgn = zeros(ndt); Gc = zeros(ndt)
    Ninv = zeros(ndt); dC = zeros(ndt)
    sepkernel!(Phi, dgn, Gc, Ninv, dC, ts, TN)

    x = abs.(randn(ndt)) .* 1e-6
    out = zeros(ndt)
    transition!(out, x, Phi, dgn, Gc, Ninv, dC, om, ndt)

    jprt = reshape(copy(x), ndt, 1)
    temp = zeros(1, ndt)
    transition!(temp, jprt, 1, Phi, dgn, Gc, Ninv, dC, om, ndt)
    @test out ≈ vec(temp[1, :])
end

@testset "fused sweep converges to the order loop under Picard" begin
    TN = [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0]
    mu = 1.25e-8
    ndt = 200
    edges, rs = prodgrid(200, 30_000)

    for ratio in (1.0, 4.0)
        rho = mu * ratio
        ref = orderref(rs, edges, mu, rho, ndt, TN, 200)
        errs = [maximum(abs.(rawfused(rs, edges, mu, rho, ndt, TN, np) .- ref) ./ abs.(ref))
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
    edges, rs = prodgrid(200, 30_000)
    for np in 1:4
        ys = rawfused(rs, edges, mu, rho, 200, TN, np)
        @test all(isfinite, ys)
        @test all(ys .> 0)
    end
end

@testset "bag wrapper matches the raw fusedsweep!" begin
    TN = [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0]
    mu = 1.25e-8
    ndt = 200
    edges, rs = prodgrid(200, 30_000)

    for ratio in (1.0, 4.0)
        rho = mu * ratio
        np = getnpicard(mu, rho)

        bag = IntegralArrays(10, ndt, length(rs), Val{length(TN)})
        fusedsweep!(bag, rs, edges, mu, rho, TN)
        auto = copy(get_tmp(bag.ys, eltype(TN)))
        @test auto ≈ rawfused(rs, edges, mu, rho, ndt, TN, np)

        # an explicit npicard overrides the rule
        fusedsweep!(bag, rs, edges, mu, rho, TN; npicard = 6)
        @test get_tmp(bag.ys, eltype(TN)) ≈ rawfused(rs, edges, mu, rho, ndt, TN, 6)

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

    # :fused agrees with the direct bag call
    got = mldsmcp(rs, edges, mu, rho, TN; ndt = ndt, method = :fused)
    bag = IntegralArrays(10, ndt, length(rs), Val{length(TN)})
    fusedsweep!(bag, rs, edges, mu, rho, TN)
    @test got ≈ get_tmp(bag.ys, eltype(TN))

    # :order reproduces the pre-change behaviour bit for bit
    order = 12
    want = mldsmcp(rs, edges, mu, rho, TN; order = order, ndt = ndt, method = :order)
    bag2 = IntegralArrays(order, ndt, length(rs), Val{length(TN)})
    SMCp.prordn!(bag2, rs, edges, mu + rho, TN)
    mldsmcp!(bag2, 1:order, mu, rho, TN)
    @test want == get_tmp(bag2.ys, eltype(TN))

    # the mutating entry still defaults to the order loop
    bag3 = IntegralArrays(order, ndt, length(rs), Val{length(TN)})
    mldsmcp!(bag3, 1:order, rs, edges, mu, rho, TN)
    @test get_tmp(bag3.ys, eltype(TN)) == want

    # res is poisoned on the fused path so stale per-order reads are loud
    bag4 = IntegralArrays(order, ndt, length(rs), Val{length(TN)})
    mldsmcp!(bag4, 1:order, rs, edges, mu, rho, TN; method = :fused)
    @test all(isnan, get_tmp(bag4.res, eltype(TN)))

    @test_throws ArgumentError mldsmcp(rs, edges, mu, rho, TN; ndt = ndt, method = :bogus)
end

# max |z| over bins that would survive adapt_histogram's tail threshold
function maxz(rs, edges, mu, rho, ndt, TN; reford = 200, tailthr = 10)
    ref = orderref(rs, edges, mu, rho, ndt, TN, reford)
    got = mldsmcp(rs, edges, mu, rho, TN; ndt = ndt, method = :fused)
    keep = findall(ref .> tailthr)
    @assert length(keep) > 100
    maximum(abs.(got[keep] .- ref[keep]) ./ sqrt.(ref[keep]))
end

@testset "fused error is far below Poisson noise at the production binning" begin
    TN = [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0]
    mu = 1.25e-8
    nbins = ndt = 800

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
        z = maxz(rs, edges, mu, rho, ndt, TN)
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
    ys = mldsmcp(rs, edges, mu, rho, TN; ndt = 800, method = :fused)
    @test all(isfinite, ys)
    @test all(ys .> 0)
    @test maxz(rs, edges, mu, rho, 800, TN) < 1e-2
end

@testset "fused error does not depend on the demography" begin
    mu = 1.25e-8
    rho = 4mu
    edges, rs = prodgrid(800, 10_000_000)
    for TN in ([3.0e9, 10000.0],
               [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0],
               [3.0e9, 20000.0, 60000.0, 8000.0, 8000.0, 16000.0, 1600.0, 2000.0, 400.0, 10000.0])
        @test maxz(rs, edges, mu, rho, 800, TN) < 1e-2
    end
end
