"""
    struct FitResult

A data structure to store the results of a fit.

Some methods are defined for this type to get the vector of parameters, std errors, 
model evidence, etc. See [`get_para`](@ref), [`sds`](@ref), [`evd`](@ref), 
[`pop_sizes`](@ref), [`durations`](@ref).
"""
struct FitResult
    nepochs::Int
    bin::Int
    mu::Float64
    para::Vector
    stderrors::Vector
    para_name
    TN::Vector
    method::String
    converged::Bool
    lp::Float64
    evidence::Float64
    opt
end

function Base.show(io::IO, f::FitResult) 
    model = (f.nepochs == 1 ? "stationary" : "$(f.nepochs) epochs") *
            (f.bin > 1 ? " (binned $(f.bin))" : "")
    print(io, "Fit ", model, " ")
    print(io, f.method, " ")
    print(io, f.converged ? "●" : "○", " ")
    print(io, "[", @sprintf("%.1e",f.para[1]))
    for i in 2:length(f.para)
        print(io, " ,", @sprintf("%.1f",f.para[i]))
    end
    print(io, "] ", @sprintf("logL %.3f",f.lp), @sprintf(" | evidence %.3f",f.evidence))
end

"""
    pars(fit::FitResult)

Return the parameters of the fit.
"""
get_para(fit::FitResult) = fit.para

"""
    sds(fit::FitResult)

Return the standard deviations of the parameters of the fit.
"""
sds(fit::FitResult) = fit.stderrors

"""
    evd(fit::FitResult)

Return the evidence of the fit.
"""
evd(fit::FitResult) = fit.evidence

"""
    pop_sizes(fit::FitResult)

Return the fitted population sizes, from past to present.
"""
pop_sizes(fit::FitResult) = fit.para[2:2:end]

"""
    durations(fit::FitResult)

Return the fitted durations of the epochs.
"""
durations(fit::FitResult) = fit.para[3:2:end-1]