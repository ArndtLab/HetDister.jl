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
    L = Int(ceil(params[1])*factor)
    Ns = Int.(ceil.(reverse(params[2:2:end])))
    any(Ns .< 10) && error("Population sizes must be at least 10")
    Ts = Int.(ceil.(cumsum(reverse(params[3:2:end]))))
    Ts = [0, Ts...]
    any(Ts .< 0) && error("Times must be non-negative")
    
    pop_sim = VaryingPopulation(;
        genome_length = L,
        mutation_rate = mu, recombination_rate = rho,
        population_sizes = Ns,
        times = Ts,
    )
    h.weights .= 0
    append!(h, IBSIterator(SMCprime.IBDIterator(pop_sim), pop_sim.mutation_rate))
end