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

    d = Demography()
    PopSim.set_via_TNvector!(d, tnv)
    g = Genome(UniformRate(rho), UniformRate(mu),  tnv[1])

    anc = sim_ancestry(SMCprime(), d, g, 2)

    
    h.weights .= 0
    @threads for _ in 1:factor
        for ibs_segment in ibs(anc)
            push!(h, length(ibs_segment))
        end
    end
    
end