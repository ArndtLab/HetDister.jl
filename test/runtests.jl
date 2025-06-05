using DemoInfer
using PopSimIBX
using HistogramBinnings
using Distributions
using StatsBase, StatsAPI
using PyPlot
using Test

using DemoInfer.Logging
disable_logging(Logging.Warn)
logger = ConsoleLogger(stderr, Logging.Error)
global_logger(logger)

const savewhenlocal = false

TNs = [
    [3000000000, 10000],
    [3000000000, 20000, 60000, 8000, 4000, 16000, 2000, 8000],
    [3000000000, 20000, 60000, 8000, 8000, 16000, 1600, 2000, 400, 10000],
    [3000000000, 20000, 60000, 8000, 8000, 16000, 1600, 2000, 400, 8000, 60, 300]
]
mus = [2.36e-8, 1.25e-8, 1e-8]
rhos = [1e-8]
itr = Base.Iterators.product(mus,rhos,TNs)

function get_sim(TN::Vector, mu::Number, rho::Number)
    L = TN[1]
    Ns = reverse(TN[2:2:end])
    Ts = cumsum(reverse(TN[3:2:end]))
    Ts = [0, Ts...]
    
    pop = VaryingPopulation(;
        genome_length = L, 
        mutation_rate = mu, recombination_rate = rho,
        population_sizes = Ns,
        times = Ts,
    )

    ibs_segments = segment_length.(collect(IBSIterator(SMCprime.IBDIterator(pop), pop.mutation_rate)));
    ibs_segments
end

@testset "eaxhaustive pre-fit $(length(TN)÷2) epochs,  mu $mu, rho $rho" for (mu,rho,TN) in itr
    h = Histogram(LogEdgeVector(lo = 1, hi = 1_000_000, nbins = 200))
    ibs_segments = get_sim(TN, mu, rho)
    append!(h, ibs_segments)
    Ltot = sum(ibs_segments)
    fits = pre_fit(h, 7, mu, Ltot, smallest_segment=30)
    nepochs = findlast(i->isassigned(fits, i), eachindex(fits))
    residuals = compute_residuals(h, mu, fits[nepochs].para)
    if savewhenlocal
        x = midpoints(h.edges[1])
        scatter(x, residuals; s = 3)
        xscale("log")
        savefig("test.pdf")
        close()
    end
    @test abs(mean(residuals)) < 3/sqrt(200)
    @test abs(std(residuals) - 1) < 3/sqrt(200)
end

# @testset "Test core functionality $(length(TN)÷2) epochs, mu $mu, rho $rho" for (mu,rho,TN) in itr

#     h = Histogram(LogEdgeVector(lo = 1, hi = 1_000_000, nbins = 200))
#     get_sim!(TN, h, mu, rho)

#     stat = pre_fit(h, 1, mu, TN[1])
#     @test isassigned(stat, 1)
#     stat = stat[1]

#     tsplit = DemoInfer.initializer(h, mu, get_para(stat))
#     @test tsplit > 0
#     tsplit = DemoInfer.initializer(h, mu, get_para(stat); frame = 10)
#     @test tsplit > 0

#     nep = estimate_nepochs(h, mu, TN[1])
#     @test nep >= (length(TN) ÷ 2)

#     f = demoinfer(h, length(TN)÷2, mu, rho, TN[1], Float64.(TN);
#         iters = 1,
#         burnin = 0
#     )
#     @test length(f.opt.chain) == 1
#     @test !isinf(f.evidence)
#     @test !any(f.opt.chain[1].opt.at_lboundary)
#     @test !any(f.opt.chain[1].opt.at_uboundary[2:end])
#     @test !iszero(get_para(f))

#     compare_models(FitResult[f])
#     compute_residuals(h, mu, TN)
# end

# TODO: numerical stability of mle opt, boundary checks