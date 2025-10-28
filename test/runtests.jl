using DemoInfer
using DemoInfer: npar, setinit!, fit_model_epochs!, PInit, 
    setnepochs!, deviant, timesplitter, integral_ws, next!,
    reset_perturb!, perturb_fit!
using PopSim
using HistogramBinnings
using Distributions
using StatsBase, StatsAPI
using Test
using DemoInfer.Spectra

include("Aqua.jl")
include("spectra.jl")

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
    fop = FitOptions(30, 1.0, 1.0)
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
    setinit!(fop, h.weights)
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
    fop = FitOptions(30, 1.0, 1.0)
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
    fop = DemoInfer.FitOptions(7, 1.0, 1.0; order = 2, ndt = 10)
    f = fit_model_epochs!(fop, h.edges[1], h.weights, Val(true))
    f = fit_model_epochs!(fop, h)
    @test DemoInfer.Optim.converged(f.opt.mle.optim_result)
    perturb_fit!(f, fop, h)
    DemoInfer.setnaive!(fop, false)
    fit_model_epochs!(fop, h)
end

function get_sim(params::Vector, mu::Float64, rho::Float64)

    tnv = map(x -> ceil(Int, x), params)
    pop = VaryingPopulation(; TNvector = tnv, mutation_rate = mu, recombination_rate = rho)

    map(IBSIterator(PopSim.SMCprimeapprox.IBDIterator(pop), mu)) do ibs_segment
        length(ibs_segment)
    end
end

@testset "Test core functionality" for (mu,rho,TN) in zip(mus, rhos, TNs)

    h = Histogram(LogEdgeVector(lo = 1, hi = 1_000_000, nbins = 200))
    ibs_segments = get_sim(TN, mu, rho)
    append!(h, ibs_segments)

    stat = pre_fit(h, 2, mu, rho, 10, 100, sum(ibs_segments); require_convergence = false)
    @test isassigned(stat, 1)
    stat = stat[1]

    fop = FitOptions(sum(ibs_segments), mu, rho)
    tsplit = deviant(h, get_para(stat), fop)
    @test length(tsplit) >= 1
    tsplit = deviant(h, get_para(stat), fop; frame = 10)
    @test length(tsplit) >= 1
    ts = timesplitter(h, get_para(stat), fop; frame = 10)
    @test length(ts) >= 2

    res = demoinfer(ibs_segments, length(TN)÷2, mu, rho;
        iters = 1
    )
    @test length(res.chain) == 1
    @test !any(isinf.(evd.(res.fits)))
    best = compare_models(res.fits)
    @test !isnothing(best)
    @test !any(best.opt.at_lboundary)
    @test !any(best.opt.at_uboundary[2:end])
    fcor = correctestimate!(fop, best, h)

    resid = compute_residuals(h, mu, rho, TN)
    @test !any(isnan.(resid))
    resid = compute_residuals(h, mu, rho, TN; naive=true)
    @test !any(isnan.(resid))
    ws = integral_ws(h.edges[1], mu, TN)
    @test !any(isnan.(ws))
    @test !any(ws .< 0)

    ibs2 = get_sim(TN, mu, rho)
    h2 = Histogram(h.edges)
    append!(h2, ibs2)
    resid2 = compute_residuals(h, h2)
    @test !any(isnan.(resid2))
end

@testset "fitting procedure" begin
    @testset "exhaustive pre-fit $(length(TN)÷2) epochs,  mu $mu, rho $rho" for (mu,rho,TN) in itr
        ibs_segments = get_sim(TN, mu, rho)
        h = adapt_histogram(ibs_segments)
        @test h.weights[end] .> 0
        Ltot = sum(ibs_segments)
        fop = FitOptions(Ltot, mu, rho; maxnts = 8, force = false)
        fits = pre_fit!(fop, h, 8; require_convergence = false)
        nepochs = length(fits)
        bestll = argmax(i->fits[i].lp, 1:nepochs)
        residuals = compute_residuals(h, mu, rho, get_para(fits[bestll]); naive = true)
        @test abs(mean(residuals)) < 3/sqrt(length(residuals))
        @test std(residuals) - 1 < 3/sqrt(length(residuals))
    end

    @testset "Iterative fit" begin
        mu, rho, TN = mus[1], rhos[1], TNs[3]
        ibs_segments = get_sim(TN, mu, rho)
        h = adapt_histogram(ibs_segments)
        Ltot = sum(ibs_segments)
        pfits = pre_fit(h, 5, mu, rho, 10, 100, Ltot; require_convergence = false)
        fop = FitOptions(Ltot, mu, rho; order = 10, ndt = 800)
        res = demoinfer(h, 5, fop)
        best = compare_models(res.fits)
        @test !isnothing(best)
        @test best.nepochs == 5
        m = 5
        for i in 1:length(res.chain)
            p = get_para(res.chain[i][m])
            wth = integral_ws(h.edges[1], mu, p)
            ws = wth .+ res.corrections[i]
            ws = max.(0,ws)
            resid = (h.weights .- ws) ./ sqrt.(h.weights .+ ws)
            resid[ws .== 0 .& h.weights .== 0] .= 0
            @test std(resid) - 1 < 3/sqrt(length(resid))
        end
    end
end