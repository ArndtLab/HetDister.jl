function plot_demography(fits::Vector{FitResult}, ax; id="", c = nothing, alp=1)
    oldest_epoch = length(fits[end].para) > 2 ? fits[end].para[3] : 1000
    old_t = maximum([sum(fit.para[3:2:end-1]) for fit in fits]) + oldest_epoch
    epochs = [length(fit.para)÷2 for fit in fits]
    for (fit,nepochs) in zip(fits,epochs)
        TN = fit.para[end:-1:2]
        stdTN = fit.opt.stderrors[end:-1:2]

        Polygon = matplotlib.patches.Polygon

        mean_size = []
        upp_size = []
        low_size = []
        for (n,sn) in zip(TN[1:2:end], stdTN[1:2:end])
            append!(mean_size, [n,n])
            append!(upp_size, [n+sn,n+sn])
            append!(low_size, [n-sn,n-sn])
        end

        mean_epochs = [0.]
        upp_epochs = [0.]
        low_epochs = [0.]
        for i in 1:nepochs-1
            t = sum(TN[2:2:end-1][1:i])
            st = stdTN[2:2:end-1][i]
            append!(mean_epochs, [t,t])
            if (TN[1:2:end][i] + stdTN[1:2:end][i]) > (TN[1:2:end][i+1] + stdTN[1:2:end][i+1])
                append!(upp_epochs, [t+st,t+st])
            else
                append!(upp_epochs, [t-st,t-st])
            end
            if (TN[1:2:end][i] - stdTN[1:2:end][i]) < (TN[1:2:end][i+1] - stdTN[1:2:end][i+1])
                append!(low_epochs, [t+st,t+st])
            else
                append!(low_epochs, [t-st,t-st])
            end
        end
        push!(mean_epochs, old_t)
        push!(upp_epochs, old_t)
        push!(low_epochs, old_t)

        if isnothing(c)
            c = "tab:" .* split("blue orange red purple olive brown cyan pink")[nepochs%8+nepochs÷8]
        end

        err = Polygon(collect(zip([upp_epochs;low_epochs[end:-1:1]],[upp_size;low_size[end:-1:1]])),facecolor=c,edgecolor="none",alpha=0.5*alp)

        ax.plot(mean_epochs, mean_size, linewidth=1, label="$nepochs epochs ($id)", color = c, alpha=alp)
        ax.add_patch(err)
        c = nothing
    end
end

"""
    plot_demography(fit::FitResult, ax; id="", c = nothing, alp=1)
    plot_demography(fits::Vector{FitResult}, ax; id="", c = nothing, alp=1)

Plot the demographic profile encoded in the parameters inferred by the fit.

If given a vector of fits, it will plot all the demographic profiles in the same axis with color coded epochs.

# Arguments
- `fit`: the fit result
- `ax`: the pyplot axis where to plot the demographic profile
- `id`: the optional label of the sample
- `c`: the optional color
- `alp`: the optional transparency, in [0,1]
"""
function plot_demography(fit::FitResult, ax; id="", c = nothing, alp=1)
    plot_demography([length(fit.para)÷2], [fit], ax; id=id, c=c, alp=alp)
end

function xy(h::HistogramBinnings.Histogram{T, 1, E}; mode = :density) where {T, E}
    hn = StatsBase.normalize(h; mode)
    return midpoints(h.edges[1]), hn.weights
end

function integral_weigths(edges::Vector{T}, mu::Float64, TN::Vector) where {T <: Number}
    a = 0.5
    last_hid_I = hid_integral(TN, mu, edges[1] - a)
    weights = Vector{Float64}(undef, length(edges)-1)
    for i in eachindex(edges[1:end-1])
        @inbounds this_hid_I = hid_integral(TN, mu, edges[i+1] - a)
        weights[i] = this_hid_I - last_hid_I
        last_hid_I = this_hid_I
    end
    weights
end

"""
    plot_hist(h::Histogram; kwargs...)

Plot the histogram `h` using PyPlot.

Optional arguments are passed to `scatter` from pyplot.
"""
function plot_hist(h::HistogramBinnings.Histogram{T, 1, E}; kwargs...) where {T, E}
    x, y = xy(h)
    scatter(x, y; kwargs...)
end

"""
    plot_residuals(h_obs::Histogram, fit::FitResult, μ::Float64, ρ::Float64; kwargs...)
    plot_residuals(h_obs::Histogram, para::Vector{T}, μ::Float64, ρ::Float64; kwargs...)

Plot the residuals of the simulation, with given `fit` result` or `para` as input, 
with respect to the observed histogram `h_obs`.

Optional arguments are passed to `scatter` from pyplot.
"""
function plot_residuals(h_obs::Histogram, fit::FitResult, μ::Float64, ρ::Float64; kwargs...)
    plot_residuals(h_obs, fit.para, μ, ρ; kwargs...)
end

function plot_residuals(h_obs::Histogram, para::Vector{T}, μ::Float64, ρ::Float64; kwargs...) where {T <: Number}
    h_sim = HistogramBinnings.Histogram(h_obs.edges)
    get_sim!(para, h_sim, μ, ρ, factor=1)
    residuals = (h_obs.weights .- h_sim.weights) ./ sqrt.(h_obs.weights)
    x, y = xy(h_obs) 
    x_ = x[(y .!= 0).&(x.>1e0)]
    y_ = residuals[(y .!= 0).&(x.>1e0)]
    scatter(x_, y_; kwargs...)
end

"""
    plot_naive_residuals(h_obs::Histogram, fit::FitResult, μ::Float64; kwargs...)
    plot_naive_residuals(h_obs::Histogram, para::Vector{T}, μ::Float64; kwargs...)

Plot of residuals between observed histogram `h_obs` and the naive theory.

See `plot_residuals` for more details.
"""
function plot_naive_residuals(h_obs::Histogram, fit::FitResult, μ::Float64; kwargs...)
    plot_naive_residuals(h_obs, fit.para, μ; kwargs...)
end

function plot_naive_residuals(h_obs::Histogram, para::Vector{T}, μ::Float64; kwargs...) where {T <: Number}
    weights_th = integral_weigths(h_obs.edges[1].edges, μ, para)
    residuals = (h_obs.weights .- weights_th) ./ sqrt.(h_obs.weights)
    x, y = xy(h_obs) 
    x_ = x[(y .!= 0).&(x.>1e0)]
    y_ = residuals[(y .!= 0).&(x.>1e0)]
    scatter(x_, y_; kwargs...)
end

"""
    get_evidence(fit::FitResult)

Return the evidence of the fit.
"""
get_evidence(fit::FitResult) = fit.opt.evidence

"""
    get_sds(fit::FitResult)

Return the standard deviations of the parameters of the fit.
"""
get_sds(fit::FitResult) = fit.opt.stderrors