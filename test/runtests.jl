using DemoInfer
using DemoInfer.PopSimIBX: VaryingPopulation, IBSIterator, SMCprime
using DemoInfer.HistogramBinnings: LogEdgeVector, Histogram
using Test

@testset "Sequential fit: recent bottleneck" begin
    L = 1_000_000_000
    mu = 2.36e-8
    rho = 1e-8
    pop = VaryingPopulation(;
        genome_length = L, 
        mutation_rate = mu, recombination_rate = rho,
        population_sizes = [10_000, 2000, 10_000],
        times = [0, 10_000, 12000],
    )
    ĥ = Histogram(LogEdgeVector(lo = 30, hi = 1_000_000, nbins = 200));
    append!(ĥ, IBSIterator(SMCprime.IBDIterator(pop), pop.mutation_rate))
    res = sequential_fit(ĥ, mu, 5, L)

    @test all(map(x->x.converged, res))
    @test isnothing(
        DemoInfer.initializer(ĥ, mu, res[end].para, pos = true)) && 
        isnothing(DemoInfer.initializer(ĥ, mu, res[end].para, pos = false)
        ) skip=true
end

@testset "Corrected fit: recent bottleneck" begin
    L = 1_000_000_000
    mu = 2.36e-8
    rho = 1e-8
    pop = VaryingPopulation(;
        genome_length = L, 
        mutation_rate = mu, recombination_rate = rho,
        population_sizes = [10_000, 2000, 10_000],
        times = [0, 10_000, 12000],
    )
    ĥ = Histogram(LogEdgeVector(lo = 30, hi = 1_000_000, nbins = 200));
    append!(ĥ, IBSIterator(SMCprime.IBDIterator(pop), pop.mutation_rate))
    res, chains = fit(ĥ, 3, mu, rho, L, start = 3, iters = 20)

    TN = [L, 10_000, 2_000, 2_000, 10_000, 10_000]
    estimate = res[end].para
    sds = res[end].opt.stderrors

    @test all((estimate .- 3sds) .< TN .< (estimate .+ 3sds)) skip=true
end