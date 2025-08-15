using DemoInfer
using PopSimIBX
using HistogramBinnings
using Distributions
using StatsBase, StatsAPI
using Test
using MLDs

using DemoInfer.Logging
disable_logging(Logging.Warn)
logger = ConsoleLogger(stderr, Logging.Error)
global_logger(logger)

include("Aqua.jl")

TNs = [
    [3000000000, 10000],
    [3000000000, 20000, 60000, 8000, 4000, 16000, 2000, 8000],
    [3000000000, 20000, 60000, 8000, 8000, 16000, 1600, 2000, 400, 10000],
    [3000000000, 20000, 60000, 8000, 8000, 16000, 1600, 2000, 400, 8000, 60, 300]
]
mus = [2.36e-8, 1.25e-8, 1e-8]
rhos = [1e-8]
itr = Base.Iterators.product(mus,rhos,TNs)

@testset "Test core functionality" for (mu,rho,TN) in zip(mus, rhos, TNs)

    h = Histogram(LogEdgeVector(lo = 1, hi = 1_000_000, nbins = 200))
    get_sim!(TN, h, mu, rho)

    stat = pre_fit(h, 1, mu, TN[1])
    @test isassigned(stat, 1)
    stat = stat[1]

    tsplit = DemoInfer.deviant(h, mu, get_para(stat))
    @test length(tsplit) >= 1
    tsplit = DemoInfer.deviant(h, mu, get_para(stat); frame = 10)
    @test length(tsplit) >= 1

    f = demoinfer(h, length(TN)÷2, mu, rho, TN[1];
        iters = 1
    )
    f = demoinfer(h, length(TN)÷2, mu, rho, TN[1], Float64.(TN);
        iters = 1
    )
    @test length(f.opt.chain) == 1
    @test !isinf(evd(f))
    @test !any(f.opt.chain[1].opt.at_lboundary)
    @test !any(f.opt.chain[1].opt.at_uboundary[2:end])
    @test !iszero(get_para(f))

    compare_models(FitResult[f])
    compute_residuals(h, mu, TN)
end

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

function noft(t::Number, ts::Vector, ns::Vector)
    pnt = 1
    while (pnt < length(ts)) && (ts[pnt] < t)
        pnt += 1
    end
    return ns[pnt]
end

@testset "fitting procedure" begin
    @testset "exhaustive pre-fit $(length(TN)÷2) epochs,  mu $mu, rho $rho" for (mu,rho,TN) in itr
        ibs_segments = get_sim(TN, mu, rho)
        h = adapt_histogram(ibs_segments)
        @test all(h.weights .> 0)
        Ltot = sum(ibs_segments)
        fits = pre_fit(h, 8, mu, Ltot; force = true)
        nepochs = findlast(i->isassigned(fits, i), eachindex(fits))
        bestll = argmax(i->fits[i].lp, 1:nepochs)
        residuals = compute_residuals(h, mu, get_para(fits[bestll]))
        @test abs(mean(residuals)) < 3/sqrt(200)
        @test abs(std(residuals) - 1) < 3/sqrt(200)
    end

    # @testset "fit $(length(TN)÷2) epochs,  mu $mu, rho $rho" for (mu,rho,TN) in itr
    #     h = Histogram(LogEdgeVector(lo = 1, hi = 1_000_000, nbins = 200))
    #     ibs_segments = get_sim(TN, mu, rho)
    #     append!(h, ibs_segments)
    #     Ltot = sum(ibs_segments)
    #     fits = Vector{FitResult}(undef, 7)
    #     Threads.@threads for n in 1:7
    #         fits[n] = demoinfer(h, n, mu, rho, Ltot)
    #     end
    #     best = compare_models(fits)
    #     @show length(get_para(best))÷2

    #     logrid = 1:log(1e7)/1000:log(1e7)
    #     grid = exp.(logrid)
    #     fts = MLDs.ordts(get_para(best))
    #     fns = MLDs.ordns(get_para(best))
    #     erfns = MLDs.ordns(sds(best))
    #     ints = MLDs.ordts(TN)
    #     inns = MLDs.ordns(TN)

    #     inN = map(t->noft(t, ints, inns), grid)
    #     fN = map(t->noft(t, fts, fns), grid)
    #     eN = map(t->noft(t, fts, erfns), grid)
    #     @test all(abs.((inN - fN) ./ eN) .< 3)
    #     if savewhenlocal
    #         plot(grid, inN, label = "input N", color = "red")
    #         plot(grid, fN, label = "fitted N", color = "blue")
    #         plot(grid, fN .+ eN, color = "grey")
    #         plot(grid, fN .- eN, color = "grey")
    #         xscale("log")
    #         legend()
    #         savefig("fit$(length(TN)÷2)epochs$mu$rho.pdf")
    #         close()
    #     end
    # end
end

# TODO: numerical stability of mle opt, boundary checks