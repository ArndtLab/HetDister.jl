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


TNs = [[3000000000, 10000]]
mus = [2.36e-8, 1.25e-8, 1e-8]
rhos = [1e-8]
itr = Base.Iterators.product(mus,rhos,TNs)

@test "Fit: mu=$mu, rho=$rho, scenario $TN" for (mu,rho,TN) in itr

    h = Histogram(LogEdgeVector(lo = 1, hi = 1_000_000, nbins = 200))
    get_sim!(TN, h, mu, rho)

    nmax = 3

    res = map(n->DemoInfer.fit(h, n, mu, rho, TN[1], iters = 1, factor=100), 1:nmax)
    
    best_model = compare_models(res, threshold = 0.05)

    @test !isnothing(best_model)
    @test best_model.nepochs == length(TN)÷2
    
    corr = best_model.opt.corrections[1]
    pred_w = DemoInfer.integral_ws(h.edges[1].edges, mu, get_para(best_model)) + corr
    ressq = (h.weights .- pred_w) .^2 ./ (h.weights .+ pred_w)
    ressq = filter(!isnan, ressq)
    chisq = sum(ressq)
    k = length(ressq)
    p = pvalue(Gamma(k/2, 2), chisq; tail=:right)
    @test p > 0.05 skip = true
end