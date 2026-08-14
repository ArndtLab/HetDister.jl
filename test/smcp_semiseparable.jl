using IBSpector.Spectra
using IBSpector.Spectra.PreallocationTools
using IBSpector.Spectra.SMCpIntegrals: sepkernel!, transition!, ptt, tolegendre
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

function nodes(ndt, TN)
    zs, wt = SMCp.gausslegendre(ndt)
    ts = zeros(ndt); dts = zeros(ndt)
    for j in 1:ndt
        ts[j], dts[j] = tolegendre(zs[j], TN)
    end
    ts, wt .* dts
end

@testset "semiseparable transition == dense transition" begin
    TNs = [
        [3.0e9, 10000.0],
        [3.0e9, 20000.0, 2500.0, 2000.0, 500.0, 10000.0],
        [3.0e9, 20000.0, 60000.0, 8000.0, 8000.0, 16000.0, 1600.0, 2000.0, 400.0, 10000.0],
    ]
    for TN in TNs
        ndt = 120
        ts, om = nodes(ndt, TN)
        @test issorted(ts)

        Phi = zeros(ndt); dgn = zeros(ndt); Gc = zeros(ndt)
        Ninv = zeros(ndt); dC = zeros(ndt)
        sepkernel!(Phi, dgn, Gc, Ninv, dC, ts, TN)

        @test all(isfinite, Phi)
        @test all(isfinite, Gc)
        @test all(dC .>= 0)          # C is monotone increasing
        @test all(Phi .>= 0)
        @test all(dgn .>= 0)

        nrs = 4
        jprt = abs.(randn(ndt, nrs)) .* 1e-6
        temp = zeros(nrs, ndt)
        for i in 1:nrs
            transition!(temp, jprt, i, Phi, dgn, Gc, Ninv, dC, om, ndt)
            ref = dense_transition(jprt, i, ts, om, TN)
            @test temp[i, :] ≈ ref rtol = 1e-10
        end
    end
end

# Full pre-change prordn!, kept verbatim as the regression reference.
function dense_prordn(rs, edges, rate, order, ndt, TN)
    nrs = length(rs)
    ts, om = nodes(ndt, TN)
    qs = [SMCp.pt(t, TN) for t in ts]
    qtt = zeros(ndt, ndt)
    for a in 1:ndt, b in 1:ndt
        w = a == b ? 1.0 : om[b]
        qtt[b, a] = max(ptt(ts[a], ts[b], TN), 0.0) * w
    end
    res = zeros(nrs, order)
    jprt = zeros(ndt, nrs)
    temp = zeros(nrs, ndt)
    for i in 1:nrs, j in 1:ndt
        jprt[j, i] = rate * exp(-2rate * rs[i] * ts[j]) * qs[j]
    end
    for i in 1:nrs
        res[i, 1] = SMCp.firstorder(rs[i], rate, TN)
    end
    for o in 1:order-1
        for i in 1:nrs, j in 1:ndt
            temp[i, j] = sum(jprt[k, i] * qtt[k, j] for k in 1:ndt)
        end
        for j in 1:ndt
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
            res[i, o+1] = sum(jprt[j, i] * 2 * ts[j] * om[j] for j in 1:ndt)
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
    order, ndt = 6, 120
    for TN in TNs, (mu, rho) in ((1.25e-8, 1.0e-8), (1.0e-8, 8.0e-8))
        rate = mu + rho
        bag = IntegralArrays(order, ndt, length(rs), Val{length(TN)})
        SMCp.prordn!(bag, rs, edges, rate, TN)
        got = get_tmp(bag.res, eltype(TN))
        want = dense_prordn(rs, edges, rate, order, ndt, TN)
        @test size(got) == size(want)
        @test got ≈ want rtol = 1e-12
    end
end
