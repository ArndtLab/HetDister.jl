using DemoInfer
using DemoInfer.PopSimIBX: VaryingPopulation, IBSIterator, SMCprime
using DemoInfer.HistogramBinnings: LogEdgeVector, Histogram
using Test

@testset "Sequential fit: recent bottleneck" begin
    pop = VaryingPopulation(;
        genome_length = 1_000_000_000, 
        mutation_rate = 2.36e-8, recombination_rate = 1e-8,
        population_sizes = [10_000, 2000, 10_000],
        times = [0, 10_000, 12000],
    )
    ĥ = Histogram(LogEdgeVector(lo = 30, hi = 1_000_000, nbins = 200));
    append!(ĥ, IBSIterator(SMCprime.IBDIterator(pop), pop.mutation_rate))
    res = sequential_fit(ĥ, 2.36e-8, 5)

    @test all(map(x->x.converged, res))
    @test isnothing(
        DemoInfer.initializer(ĥ, 2.36e-8, res[end].para, pos = true)) && 
        isnothing(DemoInfer.initializer(ĥ, 2.36e-8, res[end].para, pos = false)
        ) skip=true
end

@testset "Corrected fit: recent bottleneck" begin
    pop = VaryingPopulation(;
        genome_length = 1_000_000_000, 
        mutation_rate = 2.36e-8, recombination_rate = 1e-8,
        population_sizes = [10_000, 2000, 10_000],
        times = [0, 10_000, 12000],
    )
    ĥ = Histogram(LogEdgeVector(lo = 30, hi = 1_000_000, nbins = 200));
    append!(ĥ, IBSIterator(SMCprime.IBDIterator(pop), pop.mutation_rate))
    res, chains = corrected_fit(ĥ, 3, 2.36e-8, 1e-8, start = 3, iters = 10, final_factor=1000)

    TN = [1_000_000_000, 10_000, 2_000, 2_000, 10_000, 10_000]
    estimate = res[end].para
    sds = res[end].opt.stderrors

    @test all((estimate .- 3sds) .< TN .< (estimate .+ 3sds)) skip=true
end