using IBSpector
using IBSpector.Spectra
using Test
using HistogramBinnings
using StatsBase

const SMCp = IBSpector.Spectra.SMCpIntegrals
using IBSpector.Spectra.SMCpIntegrals: TimeGrid, ndt, timenodes!

@testset "TimeGrid" begin
    g = TimeGrid(5; m = 48, mtail = 32)
    @test g.K == 5 && g.m == 48 && g.mtail == 32
    @test ndt(g) == 4 * 48 + 32
    # a one-epoch history has no finite panels, only the tail
    @test ndt(TimeGrid(1; m = 48, mtail = 32)) == 32

    # Gauss-Legendre weights on (-1,1) sum to the interval length
    @test sum(g.wleg) ≈ 2 rtol = 1e-12
    @test length(g.zleg) == 48 && all(-1 .< g.zleg .< 1)

    # The Laguerre weights are stored FOLDED (w_i * exp(u_i)), so recovering
    # int_0^inf e^{-u} u^p du = p! requires multiplying the integrand by e^{-u}
    # exactly as `pt` does inside the sweep.
    for p in 0:5
        @test sum(g.wlag .* g.ulag .^ p .* exp.(-g.ulag)) ≈ factorial(p) rtol = 1e-9
    end
    # folded weights must be well conditioned, not e^{+u}-huge
    @test maximum(g.wlag) < 1e3
    @test minimum(g.wlag) > 1e-3
end
