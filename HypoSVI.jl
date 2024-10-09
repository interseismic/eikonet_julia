module HypoSVI
__precompile__

using DataFrames
using CSV
using JSON
using Lux
using Geodesy
using JLD2
using Dates
using StatsBase
using StatsPlots
using ForwardDiff
using Zygote
using Random
using Optimisers
using Combinatorics
using LinearAlgebra
using Distributions
using NearestNeighbors
using Distributed
using ProgressMeter
using Optim
using LineSearches
using CovarianceEstimation
using Dates
using KernelDensity
using Turing

include("./Eikonet.jl")
using .Eikonet
include("./SVIExtras.jl")

abstract type InversionMethod end
abstract type MAP <: InversionMethod end
abstract type SVI <: InversionMethod end
abstract type HMC <: InversionMethod end

struct SVIParams <: SVI
    plot_π::Bool
    evid::Int
end

struct GridSearch <: InversionMethod end
struct VI <: InversionMethod end

struct MAPParams <: MAP end

struct HMCParams <: HMC
    plot_π::Bool
    evid::Int
end

struct SGLD <: InversionMethod
    plot_π::Bool
    evid::Int
end

#function init_eikonet(params)
#    τ = Lux.Chain(
#        Dense(7, 16, Lux.elu),
#        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
#        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
#        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
#        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
#        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
#        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
#        Dense(16, 1))
#
#    return EikoNet(τ, Float32(params["scale"]))
#end

function init_eikonet(params)
    τ = Lux.Chain(
        CylindricalSymmetry(6+7, 3+1),
        Dense(4, 16, Lux.elu),
        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
        Dense(16, 1, Lux.relu))

    return EikoNet(τ, Float32(params["scale"]))
end

function timedelta(t1::DateTime, t2::DateTime)
    # returns total seconds between t1,t2
    (t1-t2) / Millisecond(1000)
end

function format_arrivals(params::Dict, phases::DataFrame, stations::DataFrame)
    phase_sta = innerjoin(phases, stations, on = [:network, :station])
    X_obs = zeros(Float32, 4, size(phase_sta, 1))
    X_obs[1,:] .= phase_sta.X
    X_obs[2,:] .= phase_sta.Y
    X_obs[3,:] .= phase_sta.Z
    arrival_times = DateTime.(phase_sta[!, "time"])
    T_obs = zeros(Float32, length(arrival_times))
    for (i, row) in enumerate(eachrow(phase_sta))
        if row.phase == "P"
            X_obs[4,i] = 0f0
        elseif row.phase == "S"
            X_obs[4,i] = 1f0
        else
            println("Error: unknown Phase label (not P or S). Exiting...")
            println(row)
        end
        T_obs[i] = timedelta(arrival_times[i], minimum(arrival_times))
    end
    T_ref = minimum(arrival_times)
    return X_obs, T_obs, T_ref, phase_sta
end

function logit(p::Float32)
    log(p / (1f0-p))
end

function sigmoid(x::Float32)
    1f0 / (1f0 + exp(-x))
end

function tukey(x::Real, c::T) where {T}
    if abs(x) <= c
        return (c^2 / T(6)) * (T(1) - (T(1) - (x/c)^2)^3)
    else
        return (c^2 / T(6))
    end
end

function tukey_loss(y::AbstractArray, c::Float32)
    return mean(map(x->tukey(x, c), y))
end

function huber_loss(x::Real, δ::Float32)
    if abs(x) < δ
        ρ = 5.0f-1 * x^2
    else
        ρ = δ * abs(x) - 5.0f-1 * δ^2
    end
    return ρ
end

function huber_loss(x::AbstractArray, δ::Float32)
    map(y->huber_loss(y, δ), x)
end

function cauchy_loss(x::Real, c::Float32)
    return 5f-1 * c^2 * log(1f0 + (x/c)^2)
end

function cauchy_loss(x::AbstractArray, δ::Float32)
    mean(map(y->cauchy_loss(y, δ), x))
end

function locate(params, X_obs::Array{T}, T_obs::Array{T}, eikonet::EikoNet, ps::NamedTuple, st::NamedTuple, ::MAP; return_input=false, normed_ll=true, σ=nothing) where {T}

    scale = Float32(params["scale"])
    θ̂ = Float32.([0.5, 0.5, Float32(params["prior_z_mean"])/scale])
    iter_tol = Float32(params["iter_tol"]) / scale

    if isnothing(σ)
        σ = Float32.(map(x -> x <= 5f-1 ? Float32(params["pick_unc_p"]) : Float32(params["pick_unc_s"]), X_obs[4,:]))
    else
        σ = Float32.(map(x -> x <= 5f-1 ? σ[1] : σ[2], X_obs[4,:]))
    end
    log_pdf = assign_likelihood(params)

    prior_μ = Float32(params["prior_z_mean"])/scale
    prior_σ = Float32(params["prior_z_std"])/scale
    prior = PGeneralizedGaussian(prior_μ, prior_σ, params["prior_scale_param"])

    # function ℓπ(θ::AbstractArray)
    #     X_in = cat(repeat(θ, 1, size(X_obs, 2)), X_obs, dims=1)
    #     T_pred = dropdims(eikonet(X_in, ps, st), dims=1)
    #     bias = mean(vec(T_obs) - T_pred)
    #     log_L = -sum(log_pdf.((T_obs .- T_pred .- bias) ./ σ))
    #     if normed_ll
    #         log_L /= length(T_obs)
    #     end
    #     log_p = -sum(logpdf.(prior, θ[3,:]))
    #     return log_L + log_p
    # end

    ipairs = collect(combinations(collect(1:size(X_obs, 2)), 2))
    ipairs = permutedims(hcat(ipairs...))
    ΔT_obs = T_obs[ipairs[:,1]] - T_obs[ipairs[:,2]]
    σ = sqrt.(σ[ipairs[:,1]].^2 + σ[ipairs[:,2]].^2)

    function ℓπ(θ::AbstractArray)
        X_in = cat(repeat(θ, 1, size(X_obs, 2)), X_obs, dims=1)
        T_pred = dropdims(eikonet(X_in, ps, st), dims=1)
        ΔT_pred = T_pred[ipairs[:,1]] - T_pred[ipairs[:,2]]
        log_L = -sum(log_pdf.((ΔT_obs - ΔT_pred) ./ σ))
        if normed_ll
            log_L /= length(ΔT_obs)
        end
        log_p = -sum(logpdf.(prior, θ[3,:]))
        return log_L + log_p
    end

    options = Optim.Options(iterations=params["n_epochs"], g_tol=0f0, f_tol=0f0, x_tol=iter_tol, allow_f_increases=true)
    result = optimize(ℓπ, θ̂, NewtonTrustRegion(), options, autodiff = :forward)
    X_best = vec(Optim.minimizer(result))

    if ~Optim.converged(result)
        # println(result)
    end

    X_in = cat(repeat(X_best, 1, size(X_obs, 2)), X_obs, dims=1)
    T_src, resid = get_origin_time(X_in, eikonet, T_obs, ps, st)

    min_lla = LLA(lat=params["lat_min"], lon=params["lon_min"], alt=0.0)
    trans = ENUfromLLA(min_lla, wgs84)
    inv_trans = LLAfromENU(min_lla, wgs84)
    X_input = copy(X_best)
    X_best .*= 1f3 .* scale
    X_best = Float64.(X_best)
    X_lla = inv_trans(ENU(X_best[1], X_best[2]))
    X_src = Dict("longitude" => X_lla.lon, "latitude" => X_lla.lat, "depth" => X_best[3], "time" => sec2date(T_src))
    if ~return_input
        return X_src, resid
    else
        return X_src, resid, X_input
    end
end

function locate(params, X_obs::Array{T}, T_obs::Array{T}, eikonet::EikoNet, ps::NamedTuple, st::NamedTuple, ::GridSearch; return_input=false) where {T}

    scale = Float32(params["scale"])
    min_depth = Float32(params["z_min"] / scale)
    max_depth = Float32(params["z_max"] / scale)
    if params["prevent_airquakes"]
        min_depth = 0f0
    end
    σ = map(x -> x <= 5f-1 ? Float32(params["pick_unc_p"]) : Float32(params["pick_unc_s"]), X_obs[4,:])
    log_pdf = assign_likelihood(params)

    prior_μ = Float32(params["prior_z_mean"])/scale
    prior_σ = Float32(params["prior_z_std"])/scale
    prior = PGeneralizedGaussian(prior_μ, prior_σ, params["prior_scale_param"])
    
    function ℓπ(θ::AbstractArray; print=false)
        X_in = cat(repeat(θ, 1, size(X_obs, 2)), X_obs, dims=1)
        T_pred = dropdims(eikonet(X_in, ps, st), dims=1)
        bias = mean(vec(T_obs) - T_pred)
        log_L = -sum(log_pdf.((T_obs .- T_pred .- bias) ./ σ))
        log_p = -sum(logpdf.(prior, θ[3,:]))
        return log_L + log_p
    end

    best_loss = Inf32
    X_best = nothing
    for xx in range(0f0, 1f0, length=100)
        for yy in range(0f0, 1f0, length=100)
            for zz in range(min_depth, max_depth, length=20)
                val = ℓπ([xx, yy, zz])
                if val < best_loss
                    X_best = [xx, yy, zz]
                    best_loss = val
                end
            end
        end
    end

    X_in = cat(repeat(X_best, 1, size(X_obs, 2)), X_obs, dims=1)
    T_src, resid = get_origin_time(X_in, eikonet, T_obs, ps, st)

    min_lla = LLA(lat=params["lat_min"], lon=params["lon_min"], alt=0.0)
    trans = ENUfromLLA(min_lla, wgs84)
    inv_trans = LLAfromENU(min_lla, wgs84)
    X_input = copy(X_best)
    X_best .*= 1f3 .* scale
    X_best = Float64.(X_best)
    X_ENU = inv_trans(ENU(X_best[1], X_best[2]))
    X_src = Dict("longitude" => X_ENU.lon, "latitude" => X_ENU.lat, "depth" => X_best[3], "time" => sec2date(T_src))
    if ~return_input
        return X_src, resid
    else
        return X_src, resid, X_input
    end
end

struct HuberDensity{T}
    δ::T
    ε::T
end

function HuberDensity(δ::Float32)
    # source: https://stats.stackexchange.com/questions/210413/generating-random-samples-from-huber-density
    y = 2f0 * pdf(Normal(0f0, 1f0), δ) / δ - 2f0 * cdf(Normal(0f0, 1f0), -δ)
    ε = y / (1f0 + y)
    return HuberDensity(δ, ε)
end

function pdf(dist::HuberDensity, x::T) where T
    if abs(x) < dist.δ
        ρ = 5.0f-1 * x^2
    else
        ρ = dist.δ * abs(x) - 5.0f-1 * dist.δ^2
    end
    return (1f0-dist.ε)/sqrt(2f0 * T(π)) * exp(-ρ)
end

function log_prob(dist::HuberDensity, x::T) where T
    if abs(x) < dist.δ
        ρ = 5.0f-1 * x^2
    else
        ρ = dist.δ * abs(x) - 5.0f-1 * dist.δ^2
    end
    return log((1f0-dist.ε)/sqrt(2f0 * T(π))) - ρ
end

function log_prob(dist::HuberDensity, x::AbstractArray)
    map(x->log_prob(dist, x), x)
end

function RBF_kernel(X::AbstractArray{Float32})
    # This is a specific algorithm for computing pairwise distances fast
    n = size(X, 2)
    G = X' * X
    d² = diag(G) .+ diag(G)' .- 2f0 .* G 
    h = median(d²) / (2f0 * log(n+1))
    γ = 1f0 / (1f-8 + 2 * h)
    K = exp.(-γ * d²)
    return K
end

function RBF_kernel(X::AbstractArray, h::Float32; return_dists=false)
    # This is a specific algorithm for fast computation of pairwise distances
    n = size(X, 2)
    G = X' * X
    d² = diag(G) .+ diag(G)' .- 2f0 .* G 
    γ = 1f0 / (1f-8 + 2f0 * h^2)
    K = exp.(-γ * d²)
    if ~return_dists
        return K
    else
        return K, d²
    end
end

function IMQ_kernel(X::AbstractArray{T}, h::T, β::T) where {T}
    G = X * X'
    d² = diag(G) .+ diag(G)' .- T(2.0) .* G
    K = (h^2 .+ d²).^β
    return K
end

# function RBF_kernel(X::AbstractArray{Float32}, C::AbstractArray, h::Float32)
#     # This is a specific algorithm for fast computation of pairwise distances
#     G = X' * X
#     d² = diag(G) .+ diag(G)' .- 2f0 .* G 
#     γ = 1f0 / (1f-8 + 2f0 * h^2)
#     K = exp.(-γ * d²) .* C
#     return K
# end

function median_bw_heuristic(X::AbstractArray)
    n = size(X, 2)
    G = X' * X
    d² = diag(G) .+ diag(G)' .- 2f0 .* G 
    h² = median(d²) / (2f0 * log(n+1f0))
    return sqrt(h²)
end

function get_origin_time(X::AbstractArray{Float32}, eikonet::EikoNet, T_obs::AbstractArray{Float32}, ps::NamedTuple, st::NamedTuple)
    # # Then determine origin time given hypocenter
    T_pred = dropdims(eikonet(X, ps, st), dims=1)
    T_obs = reshape(T_obs, :, 1)
    T_obs = repeat(T_obs, 1, size(X, 3))
    resid = T_obs - T_pred
    origin_offset = dropdims(mean(resid, dims=1), dims=1)
    i_best = argmin(abs.(origin_offset))
    return origin_offset[i_best], resid[:,i_best] .- origin_offset[i_best]
end

function reparameterize(μ::AbstractArray, log_std::AbstractArray, n_samp)
    σ = exp.(log_std) .+ 1f-5
    ε = randn(Float32, 3, n_samp)
    return μ .+ σ .* ε
end

function reparameterize(μ::AbstractArray, L::LowerTriangular, n_samp)
    ε = randn(Float32, 3, n_samp)
    return μ .+ L * ε
end

function clipped_mask(θ̂, min_depth, max_depth)
    clipped_idx = (θ̂[1,:] .< 0f0) .|| (θ̂[1,:] .> 1f0) .||
                  (θ̂[2,:] .< 0f0) .|| (θ̂[2,:] .> 1f0) .||
                  (θ̂[3,:] .< min_depth) .|| (θ̂[3,:] .> max_depth)
    N = size(θ̂, 2)
    C = ones(Float32, N, N)
    C[clipped_idx,:] .= 0f0
    C[:,clipped_idx] .= 0f0
    return C
end

function assign_kernel(params)
    if ~haskey(params, "kernel_fn")
        return (x, h) -> RBF_kernel(x, h)
    elseif uppercase.(params["kernel_fn"]) == "RBF"
        return (x, h) -> RBF_kernel(x, h)
    elseif uppercase.(params["kernel_fn"]) == "IMQ"
        return (x, h) -> IMQ_kernel(x, h, Float32(-0.5))
    end
end

function assign_likelihood(params)
    if lowercase.(params["likelihood_fn"]) == "laplace"
        return x -> logpdf(Laplace(0f0, 1f0), x)
    elseif lowercase.(params["likelihood_fn"]) == "cauchy"
        return x -> logpdf(Cauchy(0f0, 1f0), x)
    elseif lowercase.(params["likelihood_fn"]) == "gaussian"
        return x -> logpdf(Normal(0f0, 1f0), x)
    elseif lowercase.(params["likelihood_fn"]) == "huber"
        return x -> -huber_loss(x, 1f0)
    end
end

function locate(params, X_obs::Array{T}, T_obs::Array{T}, eikonet::EikoNet, ps::NamedTuple, st::NamedTuple, ::VI) where {T}
    # This version of the SVI functions does not use differential times.
    # Note: the origin time is subtracted out beforehand

    scale = Float32(params["scale"])
    iter_tol = Float32(params["iter_tol"]) / scale
    n_samp = params["n_particles"]
    n_phase = size(X_obs, 2)
    X_src, resid, X_input = locate(params, X_obs, T_obs, eikonet, ps, st, MAPParams(), return_input=true)

    σ = map(x -> x <= 5f-1 ? Float32(params["pick_unc_p"]) : Float32(params["pick_unc_s"]), X_obs[4,:])
    σ = repeat(σ, 1, n_samp)
    X_obs = repeat(reshape(X_obs, 4, n_phase, 1), 1, 1, n_samp)

    θ̂ = deepcopy(X_input)
    log_std = log.(Float32.([1f0, 1f0, 1f0]) / scale)
    θ̂ = [θ̂..., log_std...]

    T_obs0 = reshape(T_obs, :, 1)
    T_obs0 = repeat(T_obs0, 1, n_samp)

    prior_zμ = Float32(params["prior_z_mean"]) / scale
    prior_zσ = Float32(params["prior_z_std"]) / scale
    prior_z = PGeneralizedGaussian(prior_zμ, prior_zσ, params["prior_scale_param"])
    prior_xy = PGeneralizedGaussian(5f-1, 5f-1, params["prior_scale_param"])
    likelihood_fn = Normal(0f0, 1f0)
    
    function KL(θ)
        q_μ = θ[1:3]
        log_std = θ[4:end]
        q_σ = exp.(log_std) .+ 1f-5
        hypo_sample = reparameterize(q_μ, log_std, n_samp)

        X_in = cat(repeat(reshape(hypo_sample, 3, 1, n_samp), 1, n_phase, 1), X_obs, dims=1)
        T_pred = eikonet(X_in, ps, st)[1,:,:]
        T_bias = mean(T_obs0 - T_pred)
        
        likelihood = logpdf.(likelihood_fn, (T_obs0 - T_pred .- T_bias) ./ σ)
        log_p_x = logpdf.(prior_xy, hypo_sample[1,:])
        log_p_y = logpdf.(prior_xy, hypo_sample[2,:])
        log_p_z = logpdf.(prior_z, hypo_sample[3,:])

        likelihood = sum(likelihood, dims=1)
        log_p = log_p_x + log_p_y + log_p_z
        log_q = logpdf(MvNormal(q_μ, q_σ), hypo_sample)

        # by taking the mean we approximate the expectation
        return -sum(likelihood .+ log_p .- log_q) / n_samp
    end

    # println(θ̂)
    # println(KL(θ̂))
    # return
    η = Float32(params["lr"])
    state = Optimisers.setup(Optimisers.AdaGrad(η), θ̂)

    # options = Optim.Options(iterations=params["n_epochs"], g_tol=0f0, f_tol=0f0, x_tol=iter_tol, allow_f_increases=true)
    # result = optimize(KL, θ̂, Newton(), options, autodiff = :forward)
    # println(result)
    # result = Optim.minimizer(result)
    # println(result)

    for i in 1:params["n_epochs"]
        ∇loss = ForwardDiff.gradient(KL, θ̂)
        state, θ̂ = Optimisers.update(state, θ̂, ∇loss)
        if norm(∇loss, Inf) <= iter_tol
            if Int(params["svi_verbose"]) >= 1
                println("Converged at epoch $i ", norm(∇loss, Inf))
            end
            break
        end 

        if (i == Int(params["n_epochs"])) && (Int(params["svi_verbose"]) >= 1 )
            println("Failed to converge; ", norm(∇loss, Inf))
        end

        if Int(params["svi_verbose"]) == 2
            println("Epoch $i ", norm(∇loss, Inf))
        end
    end 

    # A reminder that you can't use second order methods with stochasticity (e.g. sampling from q)

    # q_σ = result[4:end]
    q_σ = θ̂[4:end]
    σ = exp.(q_σ) .* scale

    X_src["h_unc_min"] = σ[1]
    X_src["h_unc_max"] = σ[2]
    X_src["z_unc"] = σ[3]
    return X_src, resid

end

function locate(params, X_obs::Array{T}, T_obs::Array{T}, eikonet::EikoNet, ps::NamedTuple, st::NamedTuple, other_params::SVI; σ=nothing) where {T}
    # SVI backend for computing the posterior
    # This version of the SVI functions does not use differential times.
    # Note: the origin time is subtracted out beforehand
    X_src, resid, X_input = locate(params, X_obs, T_obs, eikonet, ps, st, MAPParams(), return_input=true)

    scale = Float32(params["scale"])
    N = params["n_particles"]
    M = size(X_obs, 2)
    η = Float32(params["lr"])
    iter_tol = Float32(params["svi_iter_tol"])/scale

    log_pdf = assign_likelihood(params)

    prior_μ = Float32(params["prior_z_mean"])/scale
    prior_σ = Float32(params["prior_z_std"])/scale
    prior = PGeneralizedGaussian(prior_μ, prior_σ, params["prior_scale_param"])

    θ̂ = repeat(X_input, 1, 1, N)
    θ̂ = θ̂ + rand(Float32, 3, 1, N) .* 1f0 / scale
    # θ̂[3,:,:] .= Float32(params["prior_z_mean"])/scale

    if isnothing(σ)
        σ = Float32.(map(x -> x <= 5f-1 ? Float32(params["pick_unc_p"]) : Float32(params["pick_unc_s"]), X_obs[4,:]))
    else
        σ = Float32.(map(x -> x <= 5f-1 ? σ[1] : σ[2], X_obs[4,:]))
    end
    σ = repeat(σ, 1, N)

    X_obs = repeat(reshape(X_obs, 4, M, 1), 1, 1, N)
    T_obs = repeat(reshape(T_obs, :, 1), 1, N)

    state = Optimisers.setup(Optimisers.Adam(η), θ̂)
    
    function ℓπ(θ::AbstractArray)
        X_in = cat(repeat(θ, 1, M, 1), X_obs, dims=1)
        T_pred = eikonet(X_in, ps, st)[1,:,:]
        T_bias = mean(T_obs - T_pred)
        log_L = sum(log_pdf.((T_obs - T_pred .- T_bias) ./ σ), dims=1)
        log_p = logpdf.(prior, θ[3,:,:])
        return sum(log_L + log_p)
    end

    function ∇_stein_kernel(θ::AbstractArray)
        ∇ℓπ = Zygote.gradient(ℓπ, θ)[1][:,1,:]'
        h = median_bw_heuristic(θ[:,1,:])
        K = RBF_kernel(θ[:,1,:], h)
        ∇K = Zygote.gradient(x -> sum(RBF_kernel(x, h)), θ[:,1,:])[1]
        ϕ = (K * ∇ℓπ - ∇K')' ./ N
        ϕ = Float32.(-1.0 .* reshape(ϕ, size(ϕ,1), 1, size(ϕ,2)))
        return ϕ
    end

    for i in 1:params["n_epochs"]
        ϕ = ∇_stein_kernel(θ̂)
        state, θ̂_new = Optimisers.update(state, θ̂, ϕ)
        Δθ = θ̂_new - θ̂
        θ̂ = θ̂_new

        if (i % params["lr_decay_interval"]) == 0
            η /= 1f1
            Optimisers.adjust!(state, η)
        end

        if norm(ϕ, Inf) <= iter_tol
            if Int(params["svi_verbose"]) >= 1
                println("Converged at epoch $i ", norm(ϕ, Inf))
            end
            break
        end 

        if (i == Int(params["n_epochs"])) && (Int(params["svi_verbose"]) >= 1 )
            println("Failed to converge; ", norm(ϕ, Inf))
        end

        if Int(params["svi_verbose"]) == 2
            println("Epoch $i ", norm(ϕ, Inf))
        end
    end

    # Convert X back to meters
    X_best = θ̂ .* scale
    X_best = dropdims(X_best, dims=2)

    # Gaussian approx of posterior uncertainty
    X_cov = cov(BiweightMidcovariance(), X_best')
    z_unc = sqrt(X_cov[3,3])
    h_unc = sqrt.(eigvals(X_cov[1:2,1:2]))
    sort!(h_unc)

    if other_params.plot_π
        inv_trans = LLAfromENU(LLA(lat=params["lat_min"], lon=params["lon_min"]), wgs84)
        plot_particles(params, X_best .* 1f3, inv_trans, other_params.evid)
    end
    X_src["h_unc_min"] = h_unc[1]
    X_src["h_unc_max"] = h_unc[2]
    X_src["z_unc"] = z_unc
    return X_src, resid
end

function locate(params, X_obs::Array{T}, T_obs::Array{T}, eikonet::EikoNet, ps::NamedTuple, st::NamedTuple, other_params::SGLD; σ=nothing) where {T}
    # Stochastic Gradient Langevin Dynamics backend for computing the posterior
    # This version uses differential times
    # Note: the origin time is subtracted out beforehand
    X_src, resid, X_input = locate(params, X_obs, T_obs, eikonet, ps, st, MAPParams(), return_input=true)

    scale = Float32(params["scale"])
    N = params["n_particles"]
    η = Float32(params["lr"])
    iter_tol = Float32(params["svi_iter_tol"])/scale

    log_pdf = assign_likelihood(params)

    prior_μ = Float32(params["prior_z_mean"])/scale
    prior_σ = Float32(params["prior_z_std"])/scale
    prior = PGeneralizedGaussian(prior_μ, prior_σ, params["prior_scale_param"])

    θ̂ = X_input

    if isnothing(σ)
        σ = Float32.(map(x -> x <= 5f-1 ? Float32(params["pick_unc_p"]) : Float32(params["pick_unc_s"]), X_obs[4,:]))
    else
        σ = Float32.(map(x -> x <= 5f-1 ? σ[1] : σ[2], X_obs[4,:]))
    end
    
    ipairs = collect(combinations(collect(1:size(X_obs, 2)), 2))
    ipairs = permutedims(hcat(ipairs...))
    ΔT_obs = T_obs[ipairs[:,1]] - T_obs[ipairs[:,2]]
    σ = sqrt.(σ[ipairs[:,1]].^2 + σ[ipairs[:,2]].^2)

    function ℓπ(θ::AbstractArray)
        X_in = cat(repeat(θ, 1, size(X_obs, 2)), X_obs, dims=1)
        T_pred = dropdims(eikonet(X_in, ps, st), dims=1)
        ΔT_pred = T_pred[ipairs[:,1]] - T_pred[ipairs[:,2]]
        log_L = -sum(log_pdf.((ΔT_obs - ΔT_pred) ./ σ))
        log_L /= length(ΔT_obs)
        log_p = -sum(logpdf.(prior, θ[3,:]))
        return log_L + log_p
    end

    θ_samples = zeros(Float32, params["n_epochs"], 3)
    for i in 1:params["n_epochs"]
        ∇ℓπ = Zygote.gradient(ℓπ, θ̂)[1][:,1,:]
        θ̂ = θ̂ + η .* ∇ℓπ .+ sqrt(2f0 * η) .* randn(3)
        θ_samples[i,:] .= θ̂
    end

    # Convert X back to meters
    X_best = θ_samples .* scale

    # Gaussian approx of posterior uncertainty
    X_cov = cov(BiweightMidcovariance(), X_best')
    z_unc = sqrt(X_cov[3,3])
    h_unc = sqrt.(eigvals(X_cov[1:2,1:2]))
    sort!(h_unc)

    if other_params.plot_π
        inv_trans = LLAfromENU(LLA(lat=params["lat_min"], lon=params["lon_min"]), wgs84)
        plot_particles(params, X_best .* 1f3, inv_trans, other_params.evid)
    end

    X_src["h_unc_min"] = h_unc[1]
    X_src["h_unc_max"] = h_unc[2]
    X_src["z_unc"] = z_unc
    return X_src, resid
end

function locate(params, X_obs::Array{T}, T_obs::Array{T}, eikonet::EikoNet, ps::NamedTuple, st::NamedTuple, other_params::HMC) where {T}
    # This is a HMC backend for computing the posterior
    # This version of the SVI functions does not use differential times.
    # Note: the origin time is subtracted out beforehand
    X_src, resid, X_input = locate(params, X_obs, T_obs, eikonet, ps, st, MAPParams(), return_input=true)
    scale = Float32(params["scale"])
    σ = map(x -> x <= 5f-1 ? Float32(params["pick_unc_p"]) : Float32(params["pick_unc_s"]), X_obs[4,:])

    @model function Hypocenter(T_true::AbstractArray{Float64})
        # Assumptions
        X ~ Uniform(0.0, 1.0)
        Y ~ Uniform(0.0, 1.0)
        Z ~ Uniform(0.0, params["z_max"] / scale)
        # σ ~ Uniform(0.05, 0.20)
        θ = [X, Y, Z]
        X_in = cat(repeat(θ, 1, size(X_obs, 2)), X_obs, dims=1)
        T_pred = dropdims(eikonet(X_in, ps, st), dims=1)
        T_bias = mean(T_true - T_pred)
        for i in eachindex(T_true)
            T_true[i] ~ Cauchy(T_pred[i] .+ T_bias, σ[i])
        end
        # T_true ~ MvNormal(T0 .+ T_pred, σ)
    end
    iterations = params["n_particles"]
    Turing.setadbackend(:forwarddiff)
    # println(optimize(Hypocenter(Float64.(T_obs)), Turing.MAP()))
    chain = sample(Hypocenter(Float64.(T_obs)), NUTS(1000, 0.65), iterations, progress=true, verbose=false)
    # advi = ADVI(10, 1000)
    # q = vi(Hypocenter(Float64.(T_obs)), advi)

    θ̂ = hcat(chain[:X].data, chain[:Y].data, chain[:Z].data)'

    # Convert X back to meters
    X_best = θ̂ .* scale

    # Gaussian approx of posterior uncertainty
    X_cov = cov(BiweightMidcovariance(), X_best')
    z_unc = sqrt(X_cov[3,3])
    h_unc = sqrt.(eigvals(X_cov[1:2,1:2]))
    sort!(h_unc)

    if other_params.plot_π
        inv_trans = LLAfromENU(LLA(lat=params["lat_min"], lon=params["lon_min"]), wgs84)
        plot_particles(params, X_best .* 1f3, inv_trans, other_params.evid)
    end

    X_src["h_unc_min"] = h_unc[1]
    X_src["h_unc_max"] = h_unc[2]
    X_src["z_unc"] = z_unc
    return X_src, resid
end

function plot_particles(params, X, inv_trans, evid)
    lats = Vector{Float32}()
    lons = Vector{Float32}()    
    for p in eachslice(X, dims=2)
        hypo = inv_trans(ENU(p[1], p[2], 0f0))
        push!(lats, hypo.lat)
        push!(lons, hypo.lon)
    end
    xlims=(minimum(lons)-0.05, maximum(lons)+0.05)
    ylims=(minimum(lats)-0.05, maximum(lats)+0.05)
    zlims=(-1(maximum(X[3,:]/1f3)-1.0), -1*(minimum(X[3,:]/1f3)+1.0))
    # p1 = scatter(lons, lats, xlabel="Longitude", ylabel="Latitude", xlims=xlims, ylims=ylims, markersize=1.0)
    # p2 = scatter(lons, X[3,:]/1f3, xlabel="Longitude", ylabel="Depth", xlims=xlims, ylims=zlims, yflip=true, markersize=1.0)
    # p3 = scatter(lats, X[3,:]/1f3, xlabel="Latitude", ylabel="Depth", xlims=ylims, ylims=zlims, yflip=true, markersize=1.0)
    # savefig(plot(p1, p2, p3, layout=(3,1), size=(400,800), left_margin = 20Plots.mm), "$(params["plot_dir"])/event_$(evid).png")

    kde1 = marginalkde(lons, lats, xlabel="Longitude", ylabel="Latitude", xlims=xlims, ylims=ylims)
    kde2 = marginalkde(lons, -X[3,:]/1f3, xlabel="Longitude", ylabel="Depth", xlims=xlims, ylims=zlims)
    kde3 = marginalkde(lats, -X[3,:]/1f3, xlabel="Latitude", ylabel="Depth", xlims=ylims, ylims=zlims)

    kde4 = marginalkde(X[1,:]/1f3, X[2,:]/1f3, xlabel="X (km)", ylabel="Y (km)", xlims=xlims, ylims=ylims)
    kde5 = marginalkde(X[1,:]/1f3, -X[3,:]/1f3, xlabel="X (km)", ylabel="Depth (km)", xlims=xlims, ylims=zlims)
    kde6 = marginalkde(X[2,:]/1f3, -X[3,:]/1f3, xlabel="Y (km)", ylabel="Depth (km)", xlims=ylims, ylims=zlims)
    savefig(plot(kde1, kde4, kde2, kde5, kde3, kde6, layout=(3,2), size=(1000,800), left_margin = 1Plots.mm), "$(params["plot_dir"])/event_$(evid).png")

end

function update(params, ssst::DataFrame, row::DataFrameRow, max_dist::Float32, kdtree::KDTree,
                resid::DataFrame, origins::DataFrame)   

    idx = inrange(kdtree, [row.X, row.Y, row.depth], max_dist)
    evid_list = setdiff(Set(origins.evid[idx]), row.evid)
    sub_resid = filter(:evid => in(evid_list), resid)

    local_ssst = Dict()
    gdf = groupby(sub_resid, [:network, :station, :phase])
    for (key, subdf) in pairs(gdf)
        network, station, phase = values(key)
        if nrow(subdf) >= params["k-NN"]
            local_ssst[(network, station, phase)] = median(subdf.residual)
        end
    end

    return local_ssst
end

function compute_ssst(params::Dict, ssst::DataFrame, origins::DataFrame, resid::DataFrame)
    resid_origin = innerjoin(resid, origins, on=:evid)
    resid_origin[!, :id] = collect(1:nrow(resid_origin))
    kNN = Int(params["k-NN"])+1
    @time for gdf in groupby(resid_origin, [:network, :station, :phase])
        if nrow(gdf) < kNN
            continue
        end
        kdtree = KDTree(cat(gdf.X, gdf.Y, gdf.depth, dims=2)')
        idx, dists = knn(kdtree, cat(gdf.X, gdf.Y, gdf.depth, dims=2)', kNN, true)
        for i in 1:length(idx)
            resid_origin[gdf.id[i], :residual] = median(gdf.residual[idx[i][2:end]])
        end
    end
    select!(resid_origin, names(ssst))
    return resid_origin
end

function compute_ssst(params::Dict, ssst::DataFrame, origins::DataFrame, resid::DataFrame, max_dist::Float32)
    kdtree = KDTree(cat(origins.X, origins.Y, origins.depth, dims=2)')

    println("Begin updating SSSTs")
    results = @showprogress @distributed (append!) for row in eachrow(origins)
        local_ssst = update(params, ssst, row, max_dist, kdtree, resid, origins)
        [(row.evid, local_ssst)]
    end

    return results
end

function compute_ssst(params::Dict, ssst::DataFrame, origins::DataFrame, resid::DataFrame, max_dist::Float32)
    println("Begin updating SSSTs")
    resid_origin = innerjoin(resid, origins, on=:evid)
    resid_origin[!, :id] = collect(1:nrow(resid_origin))
    kNN = Int(params["k-NN"])+1

    @time for gdf in groupby(resid_origin, [:network, :station, :phase])
        if nrow(gdf) < kNN
            continue
        end

        kdtree = KDTree(cat(gdf.X, gdf.Y, gdf.depth, dims=2)')
        idx = inrange(kdtree, cat(gdf.X, gdf.Y, gdf.depth, dims=2)', max_dist, true)
        for i in 1:length(idx)
            if length(idx[i]) < kNN
                continue
            end
            resid_origin[gdf.id[i], :residual] = median(gdf.residual[idx[i][2:end]])
        end
    end
    select!(resid_origin, names(ssst))
    return resid_origin
end

function update_ssst(new_ssst::DataFrame, ssst::DataFrame, origins::DataFrame)
    ssst = leftjoin(ssst, new_ssst, on=[:evid, :network, :station, :phase], makeunique=true)
    for row in eachrow(ssst)
        if ismissing(row.residual_1)
            continue
        end
        row.residual += row.residual_1
    end
    select!(ssst, Not(:residual_1))
    return ssst
end

function update_ssst(new_ssst::Vector, ssst::DataFrame, origins::DataFrame)
    @showprogress for (evid, local_ssst) in new_ssst
        for row in eachrow(ssst[ssst.evid .== evid, :])
            key = (row.network, row.station, row.phase)
            if haskey(local_ssst, key)
                ssst.residual[row.idx] += local_ssst[key]
            end
        end
    end
    return ssst
end

function apply_ssst(phases_old::DataFrame, ssst::DataFrame)
    phases = deepcopy(phases_old)
    sort!(phases, [:evid, :network, :station, :phase])
    sort!(ssst, [:evid, :network, :station, :phase])
    for i in 1:nrow(phases)
        sgn = sign(ssst.residual[i])
        resid = abs(ssst.residual[i])
        sec = Second(floor(resid))
        msec = Millisecond(floor((resid - floor(resid)) * 1000.))
        if sgn >= 0
            phases.time[i] = phases.time[i] - (sec + msec)
        else
            phases.time[i] = phases.time[i] + (sec + msec)
        end
    end
    return phases
end

function init_ssst(phases::DataFrame, resid::DataFrame)
    ssst_dict = Dict()
    for group in groupby(resid, [:network, :station, :phase])
        row = group[1,:]
        ssst_dict[(row.network, row.station, row.phase)] = median(group.residual)
    end

    ssst = DataFrame(evid=phases.evid, network=phases.network, station=phases.station, phase=phases.phase,
                     residual=zeros(Float32, nrow(phases)))

    for row in eachrow(ssst)
        if haskey(ssst_dict, (row.network, row.station, row.phase))
            row.residual = ssst_dict[(row.network, row.station, row.phase)]
        end
    end
    return ssst
end

function init_ssst(phases::DataFrame)
    ssst = DataFrame(evid=phases.evid, network=phases.network, station=phases.station, phase=phases.phase,
                     residual=zeros(Float32, nrow(phases)))
    return ssst
end

function plot_events(origins::DataFrame)
    scatter(origins[!,:longitude], origins[!,:latitude], left_margin = 20Plots.mm)
    savefig("events.png")
end

function logrange(x1, x2, n)
    (10^y for y in range(log10(x1), log10(x2), length=n))
end

function filter_catalog!(params::Dict, origins::DataFrame)
    if haskey(params, "lat_min_filter")
        filter!(x -> (x.latitude >= params["lat_min_filter"]) & (x.latitude < params["lat_max_filter"]), origins)
    end
    if haskey(params, "lon_min_filter")
        filter!(x -> (x.longitude >= params["lon_min_filter"]) & (x.longitude < params["lon_max_filter"]), origins)
    end
    if haskey(params, "depth_min_filter")
        filter!(x -> x.depth >= params["depth_min_filter"], origins)
    end
    if haskey(params, "depth_max_filter")
        filter!(x -> x.depth < params["depth_max_filter"], origins)
    end
end

function remove_outlier_picks(params, phases::DataFrame, origins::DataFrame, residuals::DataFrame, max_resid::Real)
    phase_cols = names(phases)
    phases = innerjoin(phases, residuals, on=[:arid, :evid, :network, :station, :phase])
    filter!(x -> abs(x.residual) <= max_resid, phases)
    good_evids = []
    for group in groupby(phases, :evid)
        if nrow(group) >= params["n_det"]
            push!(good_evids, group.evid[1])
        end
    end
    filter!(:evid => in(Set(good_evids)), phases)
    filter!(:evid => in(Set(good_evids)), origins)
    phases = phases[!, phase_cols]
    return phases, origins
end

function remove_outlier_picks(params, phases::DataFrame, origins::DataFrame, residuals::DataFrame)
    phase_cols = names(phases)
    phases = innerjoin(phases, residuals, on=[:arid, :evid, :network, :station, :phase])
    # mads = combine(groupby(phases, :evid), :residual => mad)
    # phases = innerjoin(phases, mads, on=[:evid])
    phases = subset(groupby(phases, :evid), :residual => x -> abs.(x) .<= Float32(params["outlier_ndev"]) * mad(x))
    good_evids = []
    for group in groupby(phases, :evid)
        if nrow(group) >= params["n_det"]
            push!(good_evids, group.evid[1])
        end
    end
    filter!(:evid => in(Set(good_evids)), phases)
    filter!(:evid => in(Set(good_evids)), origins)
    phases = phases[!, phase_cols]
    return phases, origins
end

function remove_duplicate_picks(params, phases, origins, residuals)
    phase_cols = names(phases)
    phases = innerjoin(phases, residuals, on=[:arid, :evid, :network, :station, :phase])
    phases = combine(sdf -> sdf[argmin(abs.(sdf.residual)), :], groupby(phases, [:evid, :network, :station, :phase]))

    # Remove events with fewer than n_det picks
    good_evids = []
    for group in groupby(phases, :evid)
        if nrow(group) >= params["n_det"]
            push!(good_evids, group.evid[1])
        end
    end
    filter!(:evid => in(Set(good_evids)), phases)
    filter!(:evid => in(Set(good_evids)), origins)
    phases = phases[!, phase_cols]
    return phases, origins
end

function prune_events(params, phases, origins)
    # Remove events with fewer than n_det picks
    good_evids = []
    for group in groupby(phases, :evid)
        if nrow(group) >= params["n_det"]
            push!(good_evids, group.evid[1])
        end
    end
    filter!(:evid => in(Set(good_evids)), phases)
    filter!(:evid => in(Set(good_evids)), origins)
    return phases, origins
end

function read_phases(params, origins)
    phases = CSV.read(params["phase_file"], DataFrame)
    phases.phase = uppercase.(phases.phase)
    unique!(phases)
    filter!(:evid => in(Set(origins.evid)), phases)
    return phases
end

function init_resid_df()
    return DataFrame(arid=Int[], evid=Int[], network=String[], station=String[], phase=String[], residual=Float32[])
end

function init_origin_df(params, evid, X_src, T_src, mag, resid)
    min_lla = LLA(lat=params["lat_min"], lon=params["lon_min"], alt=0.0)
    trans = ENUfromLLA(min_lla, wgs84)
    XX, YY, ZZ = trans(LLA(X_src["latitude"], X_src["longitude"]))
    if ~haskey(X_src, "z_unc")
        origin_df = DataFrame(time=T_src, evid=evid, latitude=X_src["latitude"],
                        longitude=X_src["longitude"], depth=X_src["depth"]/1f3, magnitude=mag, X=XX/1f3, Y=YY/1f3,
                        rmse=std(resid), mae=mean(abs.(resid)))
    else
        origin_df = DataFrame(time=T_src, evid=evid, latitude=X_src["latitude"],
                        longitude=X_src["longitude"], depth=X_src["depth"]/1f3, magnitude=mag, X=XX/1f3, Y=YY/1f3,
                        unc_h_max=X_src["h_unc_max"], unc_h_min=X_src["h_unc_min"],
                        unc_z=X_src["z_unc"], rmse=std(resid), mae=mean(abs.(resid)))
    end
    resid_df = DataFrame(arid=Int[], evid=Int[], network=String[], station=String[], phase=String[], residual=Float32[])
    return origin_df, resid_df
end

function init_origin_df()
    origin_df = DataFrame(time=DateTime[], evid=Int[], latitude=Float32[], longitude=Float32[], depth=Float32[],
                            magnitude=Float32[], rmse=Float32[], mae=Float32[], X=Float32[], Y=Float32[])
    resid_df = DataFrame(arid=Int[], evid=Int[], network=String[], station=String[], phase=String[], residual=Float32[])
    return origin_df, resid_df
end

function init_ssst_radius(params)
    ssst_radius = logrange(Float32(params["min_k-NN_dist"]), Float32(params["max_k-NN_dist"]), params["n_ssst_iter"])
    ssst_radius = reverse(collect(ssst_radius))
    return ssst_radius
end

function locate_events(params, origins::DataFrame, phases::DataFrame, stations::DataFrame; mode="MAP", outfile=nothing, showprogress=false, σ=nothing)
    @load params["model_file"] ps st

    model = init_eikonet(params)
    println("Begin HypoSVI")

    results = @showprogress @distributed (append!) for phase_sub in groupby(phases, :evid)
    # results = []
    # for phase_sub in groupby(phases, :evid)
        if mode == "MAP"
            method = MAPParams()
        elseif mode == "SVI"
            method = SVIParams(params["plot_svi_results"], phase_sub.evid[1])
        elseif mode == "HMC"
            method = HMCParams(params["plot_svi_results"], phase_sub.evid[1])
        elseif mode == "GridSearch"
            method = GridSearch()
        elseif mode == "VI"
            method = VI()
        elseif mode == "SGLD"
            method = SGLD(params["plot_svi_results"], phase_sub.evid[1])
        end
        X_obs, T_obs, T_ref, phase_key = format_arrivals(params, DataFrame(phase_sub), stations)
        X_src, resid = locate(params, X_obs, T_obs, model, ps, st, method, σ=σ)

        mag = filter(row -> row.evid == phase_sub.evid[1], origins).mag[1]

        temp_origin_df, temp_resid_df = init_origin_df(params, phase_sub.evid[1], X_src, T_ref + X_src["time"], mag, resid)
        for (i, row) in enumerate(eachrow(phase_key))
            push!(temp_resid_df, (row.arid, phase_sub.evid[1], row.network, row.station, row.phase, resid[i]))
        end
        # push!(results, (temp_origin_df, temp_resid_df))
        [(temp_origin_df, temp_resid_df)]
    end

    origin_df = vcat([x[1] for x in results]...)
    resid_df = vcat([x[2] for x in results]...)

    if params["verbose"]
        println(first(origin_df, 25))
    end
    if isnothing(outfile)
        CSV.write(params["catalog_outfile"], origin_df)
    else
        CSV.write(outfile, origin_df)
    end

    return origin_df, resid_df
end


function locate_events_svi(pfile; stop=nothing, evid=nothing, showprogress=false, mode="SVI")
    # This function is meant to be used only if you want to skip the MAP steps and go straight to SVI
    params = JSON.parsefile(pfile)
    outfile = params["catalog_outfile"]
    resid_file = params["resid_outfile"]

    origins = CSV.read(params["catalog_infile"], DataFrame)
    if ~isnothing(stop)
        origins = origins[1:stop,:]
    end
    if params["verbose"]
        println("Read in ", nrow(origins), " events")
    end
    filter_catalog!(params, origins)
    if params["verbose"]
        println("After filtering ", nrow(origins), " events remaining")
    end
    phases = read_phases(params, origins)
    select!(phases, [:network, :station, :phase, :time, :evid])
    phases[!, :arid] = collect(1:nrow(phases))

    stations = get_stations(params)
    unique!(stations)

    if ~isnothing(evid)
        filter!(:evid => x -> x == evid, origins)
        filter!(:evid => x -> x == evid, phases)
    end

    origins, residuals = locate_events(params, origins, phases, stations, mode=mode, outfile="$(outfile)_svi", showprogress=showprogress)
    return
end

function mad_resid(df::DataFrame, phase)
    df_sub = filter(:phase => x -> x == phase, df)
    return mad(df_sub.residual)
end

function mad_resid(df::DataFrame, phase, n_phase_min)
    gdf = groupby(df, :evid)
    gdf = filter(:nrow => x -> x >= n_phase_min, DataFrames.transform(gdf, nrow))
    df_sub = filter(:phase => x -> x == phase, gdf)
    return mad(df_sub.residual)
end

function estimate_σ_pick(params, residuals::DataFrame)
    σ = Float32.([mad_resid(residuals, "P", params["n_phase_min_pick_std"]), mad_resid(residuals, "S", params["n_phase_min_pick_std"])])
    return σ
end

function locate_events_ssst(pfile; stop=nothing, evid=nothing, showprogress=false, svi_step=false)
    params = JSON.parsefile(pfile)
    outfile = params["catalog_outfile"]
    resid_file = params["resid_outfile"]
    ssst_file = params["ssst_outfile"]

    origins = CSV.read(params["catalog_infile"], DataFrame)
    if ~isnothing(stop)
        origins = origins[1:stop,:]
    end
    if params["verbose"]
        println("Read in ", nrow(origins), " events")
    end
    filter_catalog!(params, origins)
    if params["verbose"]
        println("After filtering ", nrow(origins), " events remaining")
    end

    phases = read_phases(params, origins)
    select!(phases, [:network, :station, :phase, :time, :evid])
    phases[!, :arid] = collect(1:nrow(phases))

    # Select a single event if desired
    if ~isnothing(evid)
        filter!(:evid => x -> x == evid, origins)
        filter!(:evid => x -> x == evid, phases)
    end

    stations = get_stations(params)
    unique!(stations)
    magnitudes = origins[!,[:evid, :mag]]

    # Initial locations
    origins, residuals = locate_events(params, origins, phases, stations, mode="MAP", outfile="$(outfile)_iter_a", showprogress=showprogress)
    CSV.write("$(resid_file)_iter_a.csv", residuals)
    origins = innerjoin(origins, magnitudes, on=:evid)
    σ = estimate_σ_pick(params, residuals)
    println("MAD residual initial P: ", σ[1], " S: ", σ[2])

    # Remove duplicates and relocate
    phases, origins = remove_duplicate_picks(params, phases, origins, residuals)
    println("Removing duplicate picks and relocate with new estimate of σ")
    origins, residuals = locate_events(params, origins, phases, stations, mode="MAP", outfile="$(outfile)_iter_b", showprogress=showprogress, σ=σ)
    CSV.write("$(resid_file)_iter_b.csv", residuals)
    origins = innerjoin(origins, magnitudes, on=:evid)
    σ = estimate_σ_pick(params, residuals)
    println("MAD residual initial P: ", σ[1], " S: ", σ[2])

    phases0 = deepcopy(phases)
    ssst = init_ssst(phases)

    # Static iterations
    for i in 1:params["n_static_iter"]
        new_ssst = init_ssst(phases, residuals)
        ssst = update_ssst(new_ssst, ssst, origins)
        phases = apply_ssst(phases0, ssst)
        CSV.write("$(ssst_file)_iter_static.csv", ssst)
        CSV.write("$(resid_file)_iter_static.csv", residuals)
        origins, residuals = locate_events(params, origins, phases, stations, mode="MAP", outfile="$(outfile)_iter_static", showprogress=showprogress, σ=σ)
        origins = innerjoin(origins, magnitudes, on=:evid)
        σ = estimate_σ_pick(params, residuals)
        println("MAD residual after static correction $i P: ", σ[1], " S: ", σ[2])
    end

    if lowercase(params["ssst_mode"]) == "shrinking"
        ssst_radius = init_ssst_radius(params)
    end

    # SSST iterations
    for k in 1:params["n_ssst_iter"]
        if lowercase(params["ssst_mode"]) == "knn"
            new_ssst = compute_ssst(params, ssst, origins, residuals)
        elseif lowercase(params["ssst_mode"]) == "shrinking"
            new_ssst = compute_ssst(params, ssst, origins, residuals, ssst_radius[k])
        end
        ssst = update_ssst(new_ssst, ssst, origins)
        phases = apply_ssst(phases0, ssst)
        CSV.write("$(ssst_file)_iter_$(k).csv", ssst)
        CSV.write("$(resid_file)_iter_$(k).csv", residuals)
        origins, residuals = locate_events(params, origins, phases, stations, mode="MAP", outfile="$(outfile)_iter_$(k)", showprogress=showprogress, σ=σ)
        origins = innerjoin(origins, magnitudes, on=:evid)
        σ = estimate_σ_pick(params, residuals)
        println("MAD residual for ssst iter $(k) P: ", σ[1], " S: ", σ[2])
    end

    if svi_step != false
        new_ssst = compute_ssst(params, ssst, origins, residuals)
        ssst = update_ssst(new_ssst, ssst, origins)
        phases = apply_ssst(phases0, ssst)
        origins, residuals = locate_events(params, origins, phases, stations, mode=svi_step, outfile="$(outfile)_svi", showprogress=showprogress, σ=σ)
    end
    return
end

end
