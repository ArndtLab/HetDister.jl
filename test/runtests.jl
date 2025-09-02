using DemoInfer
using DemoInfer: npar, setinit!, fit_model_epochs, PInit, 
    setnepochs!, deviant, timesplitter, integral_ws, next!,
    reset_perturb!, perturb_fit!
using PopSimIBX
using HistogramBinnings
using Distributions
using StatsBase, StatsAPI
using Test
using MLDs

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

@testset "Test FitOptions" begin
    fop = FitOptions(30)
    @test npar(fop) == 2
    @test fop.nepochs == 1
    @test all(fop.init .== zeros(npar(fop)))
    setinit!(fop, ones(npar(fop)))
    @test all(fop.init .!= ones(npar(fop)))
    @test all(fop.init .> fop.low)
    @test all(fop.upp .!= zeros(npar(fop)))
    @test all(fop.low .!= zeros(npar(fop)))
    h = Histogram([1,2,3,4])
    append!(h, [1,1,1,2,3,1,2])
    setinit!(fop, h.weights, 1.0)
    @test any(fop.init .!= ones(npar(fop)))
    @test all(fop.init .> zeros(npar(fop)))
    @test all(fop.init .> fop.low)
    @test all(fop.init .< fop.upp)
    @test !any(fop.perturb)
    @test all(fop.low .< rand.(fop.prior) .< fop.upp)
    setnepochs!(fop, 5)
    @test npar(fop) == 10
    @test fop.init == zeros(npar(fop))
    setinit!(fop, ones(npar(fop)))
    @test fop.perturb == falses(npar(fop))
    @test length(fop.low) == npar(fop)
    @test length(fop.upp) == npar(fop)
    @test all(fop.low .<= fop.init .<= fop.upp)
end

@testset "Test PInit" begin
    fop = FitOptions(30)
    p = PInit(fop)
    @test fop.delta.state == 0
    @test length(p) == npar(fop)
    @test all(p .== fop.init)
    @test all(fop.perturb .== false)
    fop.perturb .= trues(npar(fop))
    setinit!(fop, ones(npar(fop)))
    next!(fop.delta)
    @test length(p) == npar(fop)
    @test any(p .!= fop.init)
    @test all(fop.low .<= p .<= fop.upp)
    @test fop.delta.state == 1
    reset_perturb!(fop)
    @test all(fop.perturb .== false)
end

@testset "Test fit" begin
    h = Histogram([1,2,3,4])
    append!(h, [1,1,1,2,3,1,2])
    fop = DemoInfer.FitOptions(30)
    f = fit_model_epochs(h.edges[1], h.weights, 1.0, fop)
    f = fit_model_epochs(h, 1.0, fop)
    @test f.converged
    perturb_fit!(f, h, 1.0, fop)
end

@testset "Test core functionality" for (mu,rho,TN) in zip(mus, rhos, TNs)

    h = Histogram(LogEdgeVector(lo = 1, hi = 1_000_000, nbins = 200))
    get_sim!(TN, h, mu, rho)

    stat = pre_fit(h, 2, mu, FitOptions(TN[1]))
    @test isassigned(stat, 1)
    stat = stat[1]

    tsplit = deviant(h, mu, get_para(stat))
    @test length(tsplit) >= 1
    tsplit = deviant(h, mu, get_para(stat); frame = 10)
    @test length(tsplit) >= 1
    ts = timesplitter(h, mu, get_para(stat); frame = 10)
    @test length(ts) >= 2

    f = demoinfer(h, 1:length(TN)÷2, mu, rho, TN[1];
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

    best = compare_models(FitResult[f])
    @test !isnothing(best)
    resid = compute_residuals(h, mu, TN)
    @test !any(isnan.(resid))
    ws = integral_ws(h.edges[1], mu, TN)
    @test !any(isnan.(ws))
    @test !any(ws .< 0)
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

@testset "fitting procedure" begin
    @testset "exhaustive pre-fit $(length(TN)÷2) epochs,  mu $mu, rho $rho" for (mu,rho,TN) in itr
        ibs_segments = get_sim(TN, mu, rho)
        h = adapt_histogram(ibs_segments)
        @test all(h.weights .> 0)
        Ltot = sum(ibs_segments)
        fits = pre_fit(h, 8, mu, Ltot; require_convergence = false)
        nepochs = length(fits)
        bestll = argmax(i->fits[i].lp, 1:nepochs)
        residuals = compute_residuals(h, mu, get_para(fits[bestll]))
        @test abs(mean(residuals)) < 3/sqrt(length(residuals))
        @test abs(std(residuals) - 1) < 3/sqrt(length(residuals))
    end

    @testset "Iterative fit" begin
        mu, rho, TN = mus[1], rhos[1], TNs[3]
        ibs_segments = get_sim(TN, mu, rho)
        h = adapt_histogram(ibs_segments)
        Ltot = sum(ibs_segments)
        pfits = pre_fit(h, 5, mu, Ltot; require_convergence = false)
        fits = demoinfer(h, 4:5, mu, rho, Ltot; annealing = (L,it)->1)
        best = compare_models(fits)
        @test !isnothing(best)
        @test best.nepochs == 5
        f = fits[2]
        for i in 1:length(f.opt.chain)
            p = get_para(f.opt.chain[i])
            wth = integral_ws(h.edges[1], mu, p)
            ws = wth .+ f.opt.corrections[i]
            ws = max.(0,ws)
            resid = (h.weights .- ws) ./ sqrt.(h.weights .+ ws)
            resid[ws .== 0 .& h.weights .== 0] .= 0
            @test abs(std(resid) - 1) < 3/sqrt(length(resid))
        end
    end
end