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
    # [3000000000, 20000, 60000, 8000, 8000, 16000, 1600, 2000, 400, 8000, 60, 300],
    # [3000000000, 10000, 20000, 8000, 30000, 9000, 2000, 6000, 4000, 8000, 60, 300],
    # [3000000000, 20000, 1600, 2000, 400, 10000],
    # [3000000000, 10000, 8000, 15000, 50000, 7000],
    # [3000000000, 10000, 8000, 15000, 10000, 7000],
    # [3000000000, 10000, 80000, 13000, 3000, 10000],
    # [3000000000, 25469, 96567, 9520, 2992, 13273]
]

mus = [2.36e-8] #, 1.25e-8, 1e-8]
rhos = [1e-8]

itr = Base.Iterators.product(mus,rhos,TNs)

@testset "Preliminary fit: mu=$mu, rho=$rho, scenario $TN" for (mu,rho,TN) in itr

    h = Histogram(LogEdgeVector(lo = 1, hi = 1_000_000, nbins = 200))
    get_sim!(TN, h, mu, rho)

    nmax = estimate_nepochs(h, mu, TN[1])
    res = pre_fit(h, nmax, mu, TN[1])
    nepochs = findlast(i->isassigned(res, i), eachindex(res))
    res = res[1:nepochs]

    @test (nmax == nepochs)
    @test all(map(x->x.converged, res))
    @test iszero(DemoInfer.initializer(h, mu, res[end].para; frame = 20, pos = true)) 
    @test iszero(DemoInfer.initializer(h, mu, res[end].para; frame = 20, pos = false))
end

@testset "Fit: mu=$mu, rho=$rho, scenario $TN" for (mu,rho,TN) in itr

    h = Histogram(LogEdgeVector(lo = 1, hi = 1_000_000, nbins = 200))
    get_sim!(TN, h, mu, rho)

    # nmax = estimate_nepochs(h, mu, TN[1])
    nmax = 3

    res = map(n->DemoInfer.fit(h, n, mu, rho, TN[1], iters = 1, factor=100), 1:nmax)
    
    best_model = compare_models(res, threshold = 0.05)

    @test !isnothing(best_model)
    @test best_model.nepochs == length(TN)÷2
    
    corr = try
        best_model.opt.corrections[1]
    catch
        best_model.opt.chain[2][1]
    end
    pred_w = DemoInfer.integral_ws(h.edges[1].edges, mu, get_para(best_model)) + corr
    ressq = (h.weights .- pred_w) .^2 ./ (h.weights .+ pred_w)
    ressq = filter(!isnan, ressq)
    chisq = sum(ressq)
    k = length(ressq)
    p = pvalue(Gamma(k/2, 2), chisq; tail=:right)
    @test p > 0.05 skip = true
end