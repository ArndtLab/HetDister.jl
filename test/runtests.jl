using DemoInfer
using DemoInfer.PopSimIBX: VaryingPopulation, IBSIterator, SMCprime
using DemoInfer.HistogramBinnings: LogEdgeVector, Histogram
using DemoInfer.Distributions: Gamma
using DemoInfer.StatsAPI: pvalue
using Test

using DemoInfer.Logging
disable_logging(Logging.Warn)
logger = ConsoleLogger(stderr, Logging.Error)
global_logger(logger)


TNs = [
    [3000000000, 10000],
    # [3000000000, 20000, 60000, 8000, 4000, 16000, 2000, 8000],
    # [3000000000, 20000, 60000, 8000, 8000, 16000, 1600, 2000, 400, 10000],
    # [3000000000, 20000, 60000, 8000, 8000, 16000, 1600, 2000, 400, 8000, 60, 300]
]
mus = [2.36e-8, 1.25e-8, 1e-8]
rhos = [1e-8]
itr = Base.Iterators.product(mus,rhos,TNs)

@testset "Test core functionality $(length(TN)÷2) epochs, mu $mu, rho $rho" for (mu,rho,TN) in itr

    h = Histogram(LogEdgeVector(lo = 1, hi = 1_000_000, nbins = 200))
    get_sim!(TN, h, mu, rho)

    stat = pre_fit(h, 1, mu, TN[1])
    @test isassigned(stat, 1)
    stat = stat[1]

    tsplit = DemoInfer.initializer(h, mu, get_para(stat))
    @test tsplit > 0
    tsplit = DemoInfer.initializer(h, mu, get_para(stat); frame = 10)
    @test tsplit > 0

    nep = estimate_nepochs(h, mu, TN[1])
    @test nep >= (length(TN) ÷ 2)

    f = demoinfer(h, length(TN)÷2, mu, rho, TN[1], Float64.(TN);
        iters = 1,
        burnin = 0
    )
    @test length(f.opt.chain) == 1
    @test !isinf(f.evidence)
    @test !any(f.opt.chain[1].opt.at_lboundary)
    @test !any(f.opt.chain[1].opt.at_uboundary[2:end])
    @test !iszero(get_para(f))

    compare_models(FitResult[f])
    compute_residuals(h, mu, TN)
end

# TODO: numerical stability of mle opt, boundary checks