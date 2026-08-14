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
