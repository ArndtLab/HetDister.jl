"""
    get_sim!(params::Vector, h::Histogram, mu::Float64, rho::Float64; factor = 1)

Simulate a population according to the demographic parameters in `params` and stores 
the IBS segments in the `h`.

# Arguments
- 'params': vector of the form [L, N0, T1, N1, T2, N2] where L is the genome length,
    N0 is the ancestral population size and successive pairs of T_i and N_i are the
    durations and sizes of subsequent epochs.
- `factor`: determine how many genomes are simulated and averaged
"""
function get_sim!(params::Vector, h::Histogram, mu::Float64, rho::Float64; factor = 1)

    tnv = map(x -> ceil(Int, x), params)
    pop = VaryingPopulation(; TNvector = tnv, mutation_rate = mu, recombination_rate = rho)


    h.weights .= 0
    for _ in 1:factor
        for ibs_segment in IBSIterator(PopSim.SMCprimeapprox.IBDIterator(pop), mu)
            push!(h, length(ibs_segment))
        end
    end
    
end