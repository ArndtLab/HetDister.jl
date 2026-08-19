using IBSpector.Spectra
using IBSpector.Spectra.PreallocationTools
using IBSpector.Spectra.SMCpIntegrals: sepkernel!, transition!, ptt, TimeGrid, ndt, timenodes!
using IBSpector.Spectra.CoalescentBase: Nt, cumcr
using ForwardDiff
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

function nodes(TN)
    g = TimeGrid(length(TN) ÷ 2; m = 32, mtail = 32)
    n = ndt(g)
    ts = zeros(n); dts = zeros(n)
    timenodes!(ts, dts, g, TN)
    ts, dts
end

@testset "semiseparable transition == dense transition" begin
    TNs = [
        [3.0e9, 10000.0],
        [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0],
        [3.0e9, 20000.0, 60000.0, 8000.0, 8000.0, 16000.0, 1600.0, 2000.0, 400.0, 10000.0],
    ]
    for TN in TNs
        ts, om = nodes(TN)
        n = length(ts)
        @test issorted(ts)

        Phi = zeros(n); dgn = zeros(n); Gc = zeros(n)
        Ninv = zeros(n); dC = zeros(n)
        sepkernel!(Phi, dgn, Gc, Ninv, dC, ts, TN)

        @test all(isfinite, Phi)
        @test all(isfinite, Gc)
        @test all(dC .>= 0)          # C is monotone increasing
        @test all(Phi .>= 0)
        @test all(dgn .>= 0)

        nrs = 4
        jprt = abs.(randn(n, nrs)) .* 1e-6
        temp = zeros(nrs, n)
        for i in 1:nrs
            transition!(temp, jprt, i, Phi, dgn, Gc, Ninv, dC, om, n)
            ref = dense_transition(jprt, i, ts, om, TN)
            @test temp[i, :] ≈ ref rtol = 1e-10
        end
    end
end

# Full pre-change prordn!, kept verbatim as the regression reference.
function dense_prordn(rs, edges, rate, order, TN)
    nrs = length(rs)
    ts, om = nodes(TN)
    n = length(ts)
    qs = [SMCp.pt(t, TN) for t in ts]
    qtt = zeros(n, n)
    for a in 1:n, b in 1:n
        w = a == b ? 1.0 : om[b]
        qtt[b, a] = max(ptt(ts[a], ts[b], TN), 0.0) * w
    end
    res = zeros(nrs, order)
    jprt = zeros(n, nrs)
    temp = zeros(nrs, n)
    for i in 1:nrs, j in 1:n
        jprt[j, i] = rate * exp(-2rate * rs[i] * ts[j]) * qs[j]
    end
    for i in 1:nrs
        res[i, 1] = SMCp.firstorder(rs[i], rate, TN)
    end
    for o in 1:order-1
        for i in 1:nrs, j in 1:n
            temp[i, j] = sum(jprt[k, i] * qtt[k, j] for k in 1:n)
        end
        for j in 1:n
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
            res[i, o+1] = sum(jprt[j, i] * 2 * ts[j] * om[j] for j in 1:n)
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
    order = 6
    for TN in TNs, (mu, rho) in ((1.25e-8, 1.0e-8), (1.0e-8, 8.0e-8))
        rate = mu + rho
        grid = TimeGrid(length(TN) ÷ 2; m = 32, mtail = 32)
        bag = IntegralArrays(order, grid, length(rs), Val{length(TN)})
        SMCp.prordn!(bag, rs, edges, rate, TN)
        got = get_tmp(bag.res, eltype(TN))
        want = dense_prordn(rs, edges, rate, order, TN)
        @test size(got) == size(want)
        @test got ≈ want rtol = 1e-12
    end
end

@testset "prordn! is ForwardDiff-differentiable" begin
    edges = collect(1.0:1.0:40.0)
    rs = collect(1.0:1.0:39.0)
    order = 4
    TN0 = [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0]
    grid = TimeGrid(length(TN0) ÷ 2; m = 32, mtail = 32)

    function total(TN)
        bag = IntegralArrays(order, grid, length(rs), Val{length(TN)})
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
