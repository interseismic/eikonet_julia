module HypoSVI
__precompile__

using DataFrames
using CSV
using JSON
using Flux
using Geodesy
using JLD2
using Dates
using StatsBase
using Plots
using ForwardDiff
using Zygote
using Random
using Combinatorics
using LinearAlgebra
using Distributions
using NearestNeighbors
using Distributed
using ProgressMeter
using Optim
using LineSearches
using CovarianceEstimation
# using Printf
using Dates

include("./Eikonet.jl")
using .Eikonet
include("./Adam.jl")
include("./SVIExtras.jl")

abstract type InversionMethod end
abstract type MAP4p <: InversionMethod end
abstract type MAP3p <: InversionMethod end
abstract type VI <: InversionMethod end
abstract type SVI <: InversionMethod end

function timedelta(t1::DateTime, t2::DateTime)
    # returns total seconds between t1,t2
    (t1-t2) / Millisecond(1000)
end

function init_X(params::Dict, X_phase::Array{Float32}, ::Type{SVI})
    rng = MersenneTwister(1234)
    K = params["n_particles"]
    n_obs = size(X_phase, 1)
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)

    X_src = zeros(Float32, 3, n_obs, K)
    for i in 1:K
        lat1 = rand(rng, Uniform(params["lat_min"], params["lat_max"]))
        lon1 = rand(rng, Uniform(params["lon_min"], params["lon_max"]))
        z1 = rand(rng, Uniform(params["z_min"], params["z_max"]))
        point_enu = trans(LLA(lat=lat1, lon=lon1))
        X_src[1,:,i] .= point_enu.e
        X_src[2,:,i] .= point_enu.n
        X_src[3,:,i] .= z1*1f3
    end
    X_phase = reshape(X_phase', 4, n_obs, 1)
    X_phase = repeat(X_phase, 1, 1, K)
    X = cat(X_src, X_phase, dims=1)
    X[1:6,:,:] ./= 1f3
    return X
end
function init_X(params::Dict, X_phase::Array{Float32}, ::Type{VI})
    rng = MersenneTwister(1234)
    K = params["n_particles"]
    n_obs = size(X_phase, 1)
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)

    X_src = zeros(Float32, 3, n_obs, K)
    for i in 1:K
        lat1 = rand(rng, Uniform(params["lat_min"], params["lat_max"]))
        lon1 = rand(rng, Uniform(params["lon_min"], params["lon_max"]))
        z1 = rand(rng, Uniform(params["z_min"], params["z_max"]))
        point_enu = trans(LLA(lat=lat1, lon=lon1))
        X_src[1,:,i] .= point_enu.e
        X_src[2,:,i] .= point_enu.n
        X_src[3,:,i] .= z1*1f3
    end
    X_phase = reshape(X_phase', 4, n_obs, 1)
    X_phase = repeat(X_phase, 1, 1, K)
    X = cat(X_src, X_phase, dims=1)
    X[1:6,:,:] ./= 1f3
    return X
end

function init_X(params::Dict, X_phase::Array{Float32}, ::Type{MAP4p})
    rng = MersenneTwister(1234)
    K = 1
    n_obs = size(X_phase, 1)
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)

    X_src = zeros(Float32, 3, n_obs, K)
    for i in 1:K
        lat1 = rand(rng, Uniform(params["lat_min"], params["lat_max"]))
        lon1 = rand(rng, Uniform(params["lon_min"], params["lon_max"]))
        z1 = rand(rng, Uniform(params["z_min"], params["z_max"]))
        point_enu = trans(LLA(lat=lat1, lon=lon1))
        X_src[1,:,i] .= point_enu.e
        X_src[2,:,i] .= point_enu.n
        X_src[3,:,i] .= z1*1f3
    end
    X_phase = reshape(X_phase', 4, n_obs, 1)
    X_phase = repeat(X_phase, 1, 1, K)
    X = cat(X_src, X_phase, dims=1)
    X[1:6,:,:] ./= 1f3
    return X
end

function format_arrivals(params::Dict, phases::DataFrame, stations::DataFrame, method)
    phase_sta = innerjoin(phases, stations, on = [:network, :station])
    X_inp = zeros(Float32, size(phase_sta, 1), 4)
    X_inp[:,1] .= phase_sta.X
    X_inp[:,2] .= phase_sta.Y
    X_inp[:,3] .= phase_sta.Z
    arrival_times = DateTime.(phase_sta[!, "time"])
    T_obs = zeros(Float32, 1, length(arrival_times))
    for (i, row) in enumerate(eachrow(phase_sta))
        if row.phase == "P"
            X_inp[i,4] = 0f0
        elseif row.phase == "S"
            X_inp[i,4] = 1f0
        else
            println("Error: unknown Phase label (not P or S). Exiting...")
            println(row)
        end
        T_obs[i] = timedelta(arrival_times[i], minimum(arrival_times))
    end
    T_ref = minimum(arrival_times)
    X = init_X(params, X_inp, method)
    return X, T_obs, T_ref, phase_sta
end

function logit(p::Float32)
    log(p / (1f0-p))
end

function sigmoid(x::Float32)
    1f0 / (1f0 + exp(-x))
end

# function tukey(x, c::Float32)
#     if abs(x) <= c
#         return (c^2 / 6f0) * (1f0 - (1f0 - (x/c)^2)^3)
#     else
#         return (c^2 / 6f0)
#     end
# end

# function tukey_loss(y, c::Float32)
#     return sum(map(x->tukey(x, c), y))
# end

function locate(params, evid::Int, X0::Array{Float32}, T_obs::Array{Float32}, eikonet::EikoNet, T_ref, phase_unc::Float32, ::Type{MAP3p})
    n_phase = size(X0, 2)
    ipairs = collect(combinations(collect(1:n_phase), 2))
    ipairs = permutedims(hcat(ipairs...))
    ΔT_obs = T_obs[ipairs[:,1]] - T_obs[ipairs[:,2]]

    scaler = data_scaler(params)
    X = forward(mean(X0, dims=3), scaler)
    X_rec = X[4:end,:,:]
    σ = sqrt(2f0) * phase_unc

    if params["prevent_airquakes"]
        min_depth = -scaler.min[3] / scaler.scale
    else
        min_depth = 0f0
    end
    max_depth = (Float32(params["z_max"])-scaler.min[3]) / scaler.scale
    θ̂ = Float32.([0.5, 0.5, 0.5 * (max_depth + min_depth)])

    # First determine hypocenter with dtimes
    function loss(θ::AbstractArray)
        X_in = cat(repeat(θ, 1, size(X_rec, 2)), X_rec, dims=1)
        T_pred = dropdims(eikonet(X_in), dims=1)
        ΔT_pred = T_pred[ipairs[:,1]] - T_pred[ipairs[:,2]]
        # ℓL = Flux.huber_loss(ΔT_obs, ΔT_pred, δ=σ)
        ℓL = Flux.mae(ΔT_obs, ΔT_pred)
        return ℓL
    end

    lower = Float32.([0.0, 0.0, min_depth])
    upper = Float32.([1.0, 1.0, max_depth])
    bt = Fminbox(BFGS(linesearch=LineSearches.BackTracking(order=3)))
    options = Optim.Options(iterations=params["n_epochs"], g_tol=params["iter_tol"])
    result = optimize(loss, lower, upper, θ̂, bt, options, autodiff = :forward)
    X_best = vec(Optim.minimizer(result))
    if any(isnan.(X_best))
        X_best = θ̂
    end
    X[1:3,:] .= X_best

    # Then determine origin time given hypocenter
    T_src, resid = get_origin_time(X, eikonet, T_obs)

    X = inverse(X, scaler)

    # Reduce X over arrivals
    X = mean(X, dims=2)[1:3,1,:] .* 1f3

    X_best = dropdims(median(X, dims=2), dims=2)

    inv_trans = LLAfromENU(LLA(lat=params["lat_min"], lon=params["lon_min"]), wgs84)
    hypo_lla = inv_trans(ENU(X_best[1], X_best[2], 0f0))

    return Origin(Float32(hypo_lla.lat), Float32(hypo_lla.lon), X_best[3]/1f3, T_ref + sec2date(T_src),
                  NaN, NaN, NaN, NaN, X_best[1]/1f3, X_best[2]/1f3, [], [], [], []), resid
end

function locate(params, evid::Int, X0::Array{Float32}, T_obs::Array{Float32}, eikonet::EikoNet, T_ref, phase_unc::Float32, ::Type{MAP4p})
    n_phase = size(X0, 2)

    scaler = data_scaler(params)
    X = forward(X0, scaler)
    if ndims(X) == 3
        X = X[:,:,1]
    end
    X_src = Float32.([0.5, 0.5, 0.5])
    X_rec = X[4:end,:,:]

    σ = phase_unc

    if params["prevent_airquakes"]
        min_depth = -scaler.min[3] / scaler.scale
    else
        min_depth = 0f0
    end
    max_depth = (Float32(params["z_max"])-scaler.min[3]) / scaler.scale
    θ̂ = Float32.([0.5, 0.5, 0.5 * (max_depth + min_depth)])
    # θ̂ = [0f0, θ̂...]
    λ = Float32(params["lambda"])
    iter_tol = Float32(params["iter_tol"]) / scaler.scale
    function loss(θ::AbstractArray)
        # X_src = θ[2:4]
        # t0 = θ[1]
        X_src = θ
        X_in = cat(repeat(X_src, 1, size(X_rec, 2)), X_rec, dims=1)
        # T_pred = dropdims(eikonet(X_in), dims=1) .+ t0
        T_pred = dropdims(eikonet(X_in), dims=1)
        bias = mean(vec(T_obs)-T_pred)
        ℓL = Flux.huber_loss(vec(T_obs) .- bias, T_pred, δ=σ, agg=mean)
        x_penalty = min(0f0, X_src[1])^2 + max(0f0, X_src[1]-1f0)^2
        y_penalty = min(0f0, X_src[2])^2 + max(0f0, X_src[2]-1f0)^2
        z_penalty = min(0f0, X_src[3]-min_depth)^2 + max(0f0, X_src[3]-max_depth)^2
        penalty = x_penalty + y_penalty + z_penalty
        return ℓL + λ * penalty
    end

    # alg = Fminbox(BFGS(linesearch=LineSearches.BackTracking(order=3)))
    alg = BFGS()
    # alg = Fminbox(ConjugateGradient())
    # alg = Fminbox(GradientDescent(linesearch=LineSearches.Static(), alphaguess = LineSearches.InitialStatic(alpha=Float32(params["lr"]))))
    options = Optim.Options(iterations=params["n_epochs"], g_tol=0f0, f_tol=0f0, x_tol=iter_tol, allow_f_increases=true)
    result = optimize(loss, θ̂, alg, options, autodiff = :forward)
    X_best = vec(Optim.minimizer(result))

    # println(result)
    # println()
    if ~Optim.converged(result)
        println(result)
        # X_best = θ̂
    end
    # if any(isnan.(X_best))
    #     X_best = θ̂
    # end

    # T_src = X_best[1]
    # X_src = X_best[2:4]
    X_src = X_best

    X[1:3,:] .= X_src

    # X_in = cat(repeat(X_src, 1, size(X_rec, 2)), X_rec, dims=1)
    # T_pred = dropdims(eikonet(X_in), dims=1) .+ T_src
    # resid = vec(T_obs) - T_pred
    T_src, resid = get_origin_time(X, eikonet, T_obs)

    X = inverse(X, scaler)

    # Reduce X over arrivals
    X = mean(X, dims=2)[1:3,1,:] .* 1f3

    # Convert X back to meters
    X_best = dropdims(median(X, dims=2), dims=2)

    inv_trans = LLAfromENU(LLA(lat=params["lat_min"], lon=params["lon_min"]), wgs84)
    hypo_lla = inv_trans(ENU(X_best[1], X_best[2], 0f0))

    return Origin(Float32(hypo_lla.lat), Float32(hypo_lla.lon), X_best[3]/1f3, T_ref + sec2date(T_src),
                  NaN, NaN, NaN, NaN, X_best[1]/1f3, X_best[2]/1f3, [], [], [], []), resid
end

function locate_alt(params, evid::Int, X0::Array{Float32}, T_obs::Array{Float32}, eikonet::EikoNet, T_ref, phase_unc::Float32, ::Type{MAP4p})
    n_phase = size(X0, 2)

    scaler = data_scaler(params)
    X = forward(X0, scaler)
    if ndims(X) == 3
        X = X[:,:,1]
    end
    X_src = Float32.([0.5, 0.5, 0.5])
    X_rec = X[4:end,:,:]

    σ = phase_unc

    if params["prevent_airquakes"]
        min_depth = -scaler.min[3] / scaler.scale
    else
        min_depth = 0f0
    end
    max_depth = (Float32(params["z_max"])-scaler.min[3]) / scaler.scale
    θ̂ = Float32.([0.5, 0.5, 0.5 * (max_depth + min_depth)])
    θ̂ = [0f0, θ̂...]
    λ = Float32(params["lambda"])
    iter_tol = Float32(params["iter_tol"]) / scaler.scale

    function loss(θ::AbstractArray)
        X_src = θ[2:4]
        t0 = θ[1]
        X_in = cat(repeat(X_src, 1, size(X_rec, 2)), X_rec, dims=1)
        T_pred = dropdims(eikonet(X_in), dims=1) .+ t0
        ℓL = Flux.huber_loss(vec(T_obs), T_pred, δ=σ, agg=mean)
        # ℓL = Flux.mae(vec(T_obs), T_pred)
        # ℓL = tukey_loss(T_obs .- T_pred, 5f-1)
        x_penalty = min(0f0, X_src[1])^2 + max(0f0, X_src[1]-1f0)^2
        y_penalty = min(0f0, X_src[2])^2 + max(0f0, X_src[2]-1f0)^2
        z_penalty = min(0f0, X_src[3]-min_depth)^2 + max(0f0, X_src[3]-max_depth)^2
        penalty = x_penalty + y_penalty + z_penalty
        return ℓL + λ * penalty
    end

    η = Float32(params["lr"])
    # opt = Adam(θ̂, η)
    L_best = Inf
    L = Int
    i_best = 0
    norm_min = Inf
    for i in 1:params["n_epochs"]
        L = loss(θ̂)
        ∇L = ForwardDiff.gradient(loss, θ̂)
        # step!(opt, ∇L)
        # Δθ = opt.theta - θ̂
        Δθ = η * ∇L
        # θ̂ = opt.theta
        θ̂ .-= Δθ
        if norm(Δθ, Inf) < norm_min
            norm_min = norm(Δθ, Inf)
        end
        if L < L_best
            X_best = θ̂
            L_best = L
            i_best = i
        end
        if norm(Δθ, Inf) <= iter_tol
            println("Converged Epoch $i $i_best $L $L_best")
            break
        end
        # if (i % 100) == 0
        #     η /= 2f0
        # end
        # if (i-i_best) > 50
        #     # println("Epoch $i $i_best $L $L_best")
        #     break
        # end 
        # println("Epoch $i $i_best $L $L_best")
        if i == params["n_epochs"]
            println("Event failed to converge, L∞=", norm_min, " scaled tol=$(iter_tol)")
        end 
    end
    # println("Epoch $i_best $L $L_best")

    T_src = X_best[1]
    X_src = X_best[2:4]

    X[1:3,:] .= X_src

    X_in = cat(repeat(X_src, 1, size(X_rec, 2)), X_rec, dims=1)
    T_pred = dropdims(eikonet(X_in), dims=1) .+ T_src
    resid = vec(T_obs) - T_pred

    X = inverse(X, scaler)

    # Reduce X over arrivals
    X = mean(X, dims=2)[1:3,1,:] .* 1f3

    # Convert X back to meters
    X_best = dropdims(median(X, dims=2), dims=2)

    inv_trans = LLAfromENU(LLA(lat=params["lat_min"], lon=params["lon_min"]), wgs84)
    hypo_lla = inv_trans(ENU(X_best[1], X_best[2], 0f0))

    return Origin(Float32(hypo_lla.lat), Float32(hypo_lla.lon), X_best[3]/1f3, T_ref + sec2date(T_src),
                  NaN, NaN, NaN, NaN, X_best[1]/1f3, X_best[2]/1f3, [], [], [], []), resid
end

struct HuberDensity{T}
    δ::T
    ε::T
end

function HuberDensity(δ::Float32)
    # source: https://stats.stackexchange.com/questions/210413/generating-random-samples-from-huber-density
    y = 2f0 * pdf(Normal(0f0, 1f0), δ) / δ - 2f0 * cdf(Normal(0f0, 1f0), -δ)
    ε = y / (1+y)
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
    n = size(X, 1)
    G = X * X'
    d² = diag(G) .+ diag(G)' .- 2f0 .* G 
    h = median(d²) / (2.0 * log(n+1))
    γ = 1f0 / (1f-8 + 2 * h)
    K = exp.(-γ * d²)
    return K
end

function RBF_kernel(X::AbstractArray{Float32}, h::Float32)
    # This is a specific algorithm for fast computation of pairwise distances
    n = size(X, 1)
    G = X * X'
    d² = diag(G) .+ diag(G)' .- 2f0 .* G 
    γ = 1f0 / (1f-8 + 2f0 * h^2)
    K = exp.(-γ * d²)
    return K
end

function median_bw_heuristic(X::AbstractArray{Float32})
    n = size(X, 1)
    G = X * X'
    d² = diag(G) .+ diag(G)' .- 2f0 .* G 
    h² = median(d²) / (2f0 * log(n+1f0))
    return sqrt(h²)
end

function get_origin_time(X::AbstractArray{Float32}, eikonet::EikoNet, T_obs::AbstractArray{Float32})
    # # Then determine origin time given hypocenter
    T_pred = dropdims(eikonet(X), dims=1)
    T_obs = reshape(T_obs, :, 1)
    T_obs = repeat(T_obs, 1, size(X, 3))
    resid = T_obs - T_pred
    origin_offset = dropdims(mean(resid, dims=1), dims=1)
    i_best = argmin(abs.(origin_offset))
    return origin_offset[i_best], resid[:,i_best] .- origin_offset[i_best]
end

# function get_origin_time(X::AbstractArray{Float32}, eikonet::EikoNet, T_obs::AbstractArray{Float32})
#     # # Then determine origin time given hypocenter
#     T_pred = dropdims(eikonet(X), dims=1)
#     println("t_pred ", size(T_pred))
#     T_obs = reshape(T_obs, :, 1)
#     T_obs = repeat(T_obs, 1, size(X, 3))
#     resid = T_obs - T_pred
#     println("resid ", size(resid))
#     origin_offset = mean(resid)
#     return origin_offset, mean(resid, dims=2) .- origin_offset
# end

function reparameterize(μ::AbstractArray, log_var::AbstractArray, n_samp)
    σ = exp.(log_var) .+ 1f-5
    ε = randn(Float32, 4, n_samp)
    return μ .+ σ .* ε
end

function locate(params, evid::Int, X::Array{Float32}, T_obs::Array{Float32}, eikonet::EikoNet, T_ref, phase_unc::Float32, ::Type{VI})

    origin, resid = locate(params, evid, X, T_obs, eikonet, T_ref, phase_unc, MAP4p)

    n_samp = params["n_particles"]
    η = Float32(params["lr"])
    n_phase = size(X, 2)
    scaler = data_scaler(params)
    X = forward(X, scaler)
    X_rec = X[4:end,:,:]

    if params["prevent_airquakes"]
        min_depth = -scaler.min[3] / scaler.scale
    else
        min_depth = 0f0
    end
    max_depth = (Float32(params["z_max"])-scaler.min[3]) / scaler.scale
    # θ̂ = Float32.(rand(MvNormal([0f0, 0.5, 0.5, 0.5], [5f0, 0.1, 0.1, 0.1]), N))
    # θ̂ = reshape(θ̂, 3, 1, N)
    θ̂ = Float32.([0.5, 0.5, 0.5 * (max_depth + min_depth)])
    θ̂ = [0f0, θ̂...]
    log_var = log.([10f0, 10f0, 10f0, 1f0])
    θ̂ = [θ̂..., log_var...]

    T_obs0 = reshape(T_obs, :, 1)
    T_obs0 = repeat(T_obs0, 1, n_samp)
    μ_prior = Float32.([0.0, 0.5, 0.5, 0.5 * (max_depth + min_depth)])
    σ_prior = Float32.([20.0, 0.3, 0.3, 0.5*(max_depth - min_depth)])

    function elbo(θ)
        μ = θ[1:4]
        log_var = θ[5:end]
        σ = exp.(log_var) .+ 1f-5
        hypo_sample = reparameterize(μ, log_var, n_samp)
    
        X_src = hypo_sample[2:4,:]
        t0 = hypo_sample[1,:]
        X_src = reshape(X_src, 3, 1, n_samp)
        t0 = reshape(t0, 1, n_samp)
        t0 = repeat(t0, size(X_rec, 2), 1)
        X_in = cat(repeat(X_src, 1, size(X_rec, 2), 1), X_rec, dims=1)
        T_pred = dropdims(eikonet(X_in), dims=1) .+ t0
        
        likelihood = logpdf.(Normal(0f0, phase_unc), T_obs0 - T_pred)
        log_prior = logpdf(MvNormal(μ_prior, σ_prior), hypo_sample)
        log_p_q = logpdf(MvNormal(μ, σ), hypo_sample)
    
        likelihood = sum(likelihood, dims=1)
        log_prior = sum(log_prior, dims=1)

        # by taking the mean we approximate the expectation
        return -mean(likelihood .+ log_prior .- log_p_q)
    end

    # bt = BFGS(linesearch=LineSearches.BackTracking(order=3))
    hz = BFGS(linesearch=LineSearches.HagerZhang())
    options = Optim.Options(iterations=params["n_epochs"], g_tol=params["iter_tol"])
    result = optimize(elbo, θ̂, hz, options, autodiff = :forward)
    println(result)
    X_best = vec(Optim.minimizer(result))
    if any(isnan.(X_best))
        X_best = θ̂
    end

    T_src = X_best[1]
    X_src = X_best[2:4]
    X_rec = X_rec[:,:,1]

    X[1:3,:,:] .= X_src
    X = dropdims(mean(X, dims=3), dims=3)

    X_in = cat(repeat(X_src, 1, size(X_rec, 2)), X_rec, dims=1)
    T_pred = dropdims(eikonet(X_in), dims=1) .+ T_src
    resid = vec(T_obs) - T_pred
    println(size(resid))

    X = inverse(X, scaler)

    # Reduce X over arrivals
    X = mean(X, dims=2)[1:3,1,:] .* 1f3

    # Convert X back to meters
    X_best = dropdims(median(X, dims=2), dims=2)

    inv_trans = LLAfromENU(LLA(lat=params["lat_min"], lon=params["lon_min"]), wgs84)
    hypo_lla = inv_trans(ENU(X_best[1], X_best[2], 0f0))

    return Origin(Float32(hypo_lla.lat), Float32(hypo_lla.lon), X_best[3]/1f3, T_ref + sec2date(T_src),
                  NaN, NaN, NaN, NaN, X_best[1]/1f3, X_best[2]/1f3, [], [], [], []), resid

end

function locate(params, evid::Int, X::Array{Float32}, T_obs::Array{Float32}, eikonet::EikoNet, T_ref, phase_unc::Float32, ::Type{SVI})

    origin, resid = locate(params, evid, X, T_obs, eikonet, T_ref, phase_unc, MAP4p)

    N = params["n_particles"]
    η = Float32(params["lr"])
    n_phase = size(X, 2)
    ipairs = collect(combinations(collect(1:n_phase), 2))
    ipairs = permutedims(hcat(ipairs...))
    if size(ipairs, 1) > params["n_max_pairs"]
        idx = sample(collect(1:size(ipairs, 1)), params["n_max_pairs"], replace=false)
        ipairs = ipairs[idx,:]
    end
    ΔT_obs = T_obs[ipairs[:,1]] - T_obs[ipairs[:,2]]

    scaler = data_scaler(params)

    X = forward(X, scaler)
    θ̂ = Float32.(rand(MvNormal([0.5, 0.5, 0.5], [0.1, 0.1, 0.1]), N))
    θ̂ = reshape(θ̂, 3, 1, N)
    X_rec = X[4:end,:,:]

    ΔT_obs = reshape(ΔT_obs, :, 1)
    ΔT_obs = repeat(ΔT_obs, 1, N)
    K = zeros(Float32, N, N)
    ∇K = zeros(Float32, 3, N, N)

    opt = Adam(θ̂, η)
    X_last = zeros(Float32, 3, 1, N)
    if params["likelihood_fn"] == "laplace"
        σ = sqrt(2f0) * max(mean(abs.(resid)), phase_unc)
        dist = Laplace(0f0, σ)
    elseif params["likelihood_fn"] == "cauchy"
        σ = sqrt(2f0) * max(5f-1 * iqr(resid), phase_unc)
        dist = Cauchy(0f0, σ)
    end

    function ℓπ(θ::Array)
        X_in = cat(repeat(θ, 1, n_phase, 1), X_rec, dims=1)
        println(size(X_in))
        T_pred = dropdims(eikonet(X_in), dims=1)
        ΔT_pred = T_pred[ipairs[:,1],:] - T_pred[ipairs[:,2],:]
        # loss = -Flux.huber_loss(ΔT_obs./σ, ΔT_pred./σ, agg=sum)
        loss = sum(logpdf.(dist, ΔT_obs-ΔT_pred))
        return loss
    end

    L_best = -Inf
    i_best = 0
    for i in 1:params["n_epochs"]
        L, ∇L = Zygote.withgradient(ℓπ, θ̂)
        ∇L = ∇L[1]
        ∇L = dropdims(∇L, dims=2)'

        h = median_bw_heuristic(dropdims(θ̂, dims=2)')
        K = RBF_kernel(dropdims(θ̂, dims=2)', h)
        ∇K = Zygote.gradient(x -> sum(RBF_kernel(x, h)), dropdims(θ̂, dims=2)')[1]
        ϕ = transpose((K * ∇L .- ∇K) ./ size(K, 1))

        step!(opt, Float32.(Flux.unsqueeze(-1f0 * ϕ, 2)))
        θ̂ = opt.theta

        if (i-i_best) > 50
            # println("Epoch $i $i_best $L $L_best")
            break
        end 

        if L > L_best
            X[1:3,:,:] .= θ̂
            L_best = L
            i_best = i
        end
        # println("Epoch $i $i_best $L $L_best")
    end

    X = inverse(X, scaler)

    # Reduce X over arrivals
    X = mean(X, dims=2)[1:3,1,:]

    # Convert X back to meters
    X .*= 1f3
    X_best = dropdims(mean(X, dims=2), dims=2)

    # Gaussian approx of posterior uncertainty
    X_cov = cov(BiweightMidcovariance(), X' ./ 1.0f3)
    z_unc = sqrt(X_cov[3,3])
    h_unc = sqrt.(eigvals(X_cov[1:2,1:2]))
    sort!(h_unc)

    inv_trans = LLAfromENU(LLA(lat=params["lat_min"], lon=params["lon_min"]), wgs84)
    hypo_lla = inv_trans(ENU(X_best[1], X_best[2], 0f0))

    if false
        plot_particles(params, X, inv_trans, evid)
    end

    return Origin(origin.lat, origin.lon, origin.depth, origin.time, origin.mag,
                  h_unc[2], h_unc[1], z_unc, origin.X, origin.Y, origin.arids,
                  origin.resid, origin.mags, origin.prob), resid
    # return Origin(Float32(hypo_lla.lat), Float32(hypo_lla.lon), X_best[3]/1f3,
    #               T_ref + sec2date(T_src), NaN, h_unc[2], h_unc[1], z_unc, X_best[1]/1f3, X_best[2]/1f3,
    #               [], [], [], []), resid
end

function locate(params, evid::Int, X::Array{Float32}, T_obs0::Array{Float32}, eikonet::EikoNet, T_ref, phase_unc::Float32, ::Type{SVI})

    origin, resid = locate(params, evid, X, T_obs0, eikonet, T_ref, phase_unc, MAP4p)
    T_src_best = origin.time - T_ref

    N = params["n_particles"]
    η = Float32(params["lr"])
    n_phase = size(X, 2)

    scaler = data_scaler(params)

    if params["prevent_airquakes"]
        min_depth = -scaler.min[3] / scaler.scale
    else
        min_depth = 0f0
    end
    max_depth = (Float32(params["z_max"])-scaler.min[3]) / scaler.scale

    X = forward(X, scaler)
    θ̂ = Float32.(rand(MvNormal([0.5, 0.5, 0.5*(max_depth-min_depth)], [0.1, 0.1, 0.1]), N))
    θ̂ = reshape(θ̂, 3, 1, N)
    X_rec = X[4:end,:,:]

    T_obs = reshape(T_obs0, :, 1)
    T_obs = repeat(T_obs, 1, N)
    K = zeros(Float32, N, N)
    ∇K = zeros(Float32, 3, N, N)

    opt = Adam(θ̂, η)
    X_last = zeros(Float32, 3, 1, N)
    if params["likelihood_fn"] == "laplace"
        σ = max(mean(abs.(resid)), phase_unc)
        dist = Laplace(0f0, σ)
    elseif params["likelihood_fn"] == "cauchy"
        σ = max(5f-1 * iqr(resid), phase_unc)
        dist = Cauchy(0f0, σ)
    end

    function ℓπ(θ::Array)
        X_in = cat(repeat(θ, 1, n_phase, 1), X_rec, dims=1)
        T_pred = dropdims(eikonet(X_in), dims=1)
        # loss = -Flux.huber_loss(ΔT_obs./σ, ΔT_pred./σ, agg=sum)
        resid = T_obs - T_pred .- T_src_best.value/1f3
        loss = sum(logpdf.(dist, resid))
        return loss
    end

    L_best = -Inf
    i_best = 0
    for i in 1:params["n_epochs"]
        L, ∇L = Zygote.withgradient(ℓπ, θ̂)
        ∇L = ∇L[1]
        ∇L = dropdims(∇L, dims=2)'

        h = median_bw_heuristic(dropdims(θ̂, dims=2)')
        K = RBF_kernel(dropdims(θ̂, dims=2)', h)
        ∇K = Zygote.gradient(x -> sum(RBF_kernel(x, h)), dropdims(θ̂, dims=2)')[1]
        ϕ = transpose((K * ∇L .- ∇K) ./ size(K, 1))

        step!(opt, Float32.(Flux.unsqueeze(-1f0 * ϕ, 2)))
        θ̂ = opt.theta

        if (i-i_best) > 50
            # println("Epoch $i $i_best $L $L_best")
            break
        end 

        if L > L_best
            X[1:3,:,:] .= θ̂
            L_best = L
            i_best = i
        end
        # println("Epoch $i $i_best $L $L_best")
    end

    X = inverse(X, scaler)

    # Reduce X over arrivals
    X = mean(X, dims=2)[1:3,1,:]

    # Convert X back to meters
    X .*= 1f3
    X_best = dropdims(mean(X, dims=2), dims=2)

    # Gaussian approx of posterior uncertainty
    X_cov = cov(BiweightMidcovariance(), X' ./ 1.0f3)
    z_unc = sqrt(X_cov[3,3])
    h_unc = sqrt.(eigvals(X_cov[1:2,1:2]))
    sort!(h_unc)

    inv_trans = LLAfromENU(LLA(lat=params["lat_min"], lon=params["lon_min"]), wgs84)
    hypo_lla = inv_trans(ENU(X_best[1], X_best[2], 0f0))

    if false
        plot_particles(params, X, inv_trans, evid)
    end

    return Origin(origin.lat, origin.lon, origin.depth, origin.time, origin.mag,
                  h_unc[2], h_unc[1], z_unc, origin.X, origin.Y, origin.arids,
                  origin.resid, origin.mags, origin.prob), resid
    # return Origin(Float32(hypo_lla.lat), Float32(hypo_lla.lon), X_best[3]/1f3,
    #               T_ref + sec2date(T_src), NaN, h_unc[2], h_unc[1], z_unc, X_best[1]/1f3, X_best[2]/1f3,
    #               [], [], [], []), resid
end

function plot_particles(params, X_in)
    X = copy(X_in) / 1.0f3
    X[1:2,:] .-= mean(X[1:2,:], dims=2)
    xlims=(-5.0, 5.0)
    ylims=(-5.0, 5.0)
    zlims=(-params["z_max"], -params["z_min"])
    p1 = scatter(X[1,:], X[2,:], xlabel="Longitude", ylabel="Latitude", xlims=xlims, ylims=ylims)
    p2 = scatter(X[1,:], -X[3,:], xlabel="Longitude", ylabel="Depth", xlims=xlims, ylims=zlims)
    p3 = scatter(X[2,:], -X[3,:], xlabel="Latitude", ylabel="Depth", xlims=ylims, ylims=zlims)
    plot(p1, p2, p3, layout=(3,1), size=(400,800), left_margin = 20Plots.mm)
    savefig("test.png")
end

function plot_particles(params, X, inv_trans, evid)
    lats = Vector{Float32}()
    lons = Vector{Float32}()    
    for p in eachslice(X, dims=2)
        hypo = inv_trans(ENU(p[1], p[2], 0f0))
        push!(lats, hypo.lat)
        push!(lons, hypo.lon)
    end
    xlims=(params["lon_min"], params["lon_max"])
    ylims=(params["lat_min"], params["lat_max"])
    zlims=(params["z_min"], params["z_max"])
    p1 = scatter(lons, lats, xlabel="Longitude", ylabel="Latitude", xlims=xlims, ylims=ylims)
    p2 = scatter(lons, X[3,:]/1f3, xlabel="Longitude", ylabel="Depth", xlims=xlims, ylims=zlims, yflip=true)
    p3 = scatter(lats, X[3,:]/1f3, xlabel="Latitude", ylabel="Depth", xlims=ylims, ylims=zlims, yflip=true)
    plot(p1, p2, p3, layout=(3,1), size=(400,800), left_margin = 20Plots.mm)
    savefig("event_$(evid).png")
end

function update(params, ssst::DataFrame, row::DataFrameRow, max_dist::Float32, kdtree::KDTree,
                resid::DataFrame, origins::DataFrame)   
    if params["k-NN"] >= 1
        max_kNN = min(params["k-NN"], size(origins, 1))
        idx, dists = knn(kdtree, [row.X, row.Y, row.depth], max_kNN)
    else
        idx = inrange(kdtree, [row.X, row.Y, row.depth], max_dist)
    end

    evid_list = setdiff(Set(origins.evid[idx]), row.evid)
    sub_resid = filter(:evid => in(evid_list), resid)

    local_ssst = Dict()
    gdf = groupby(sub_resid, [:network, :station, :phase])
    for (key, subdf) in pairs(gdf)
        network, station, phase = values(key)
        if nrow(subdf) >= params["min_neighbors"]
            local_ssst[(network, station, phase)] = median(subdf.residual)
        end
    end

    return local_ssst
end

function compute_ssst(ssst::DataFrame, origins::DataFrame, resid::DataFrame, params::Dict)
    resid_origin = innerjoin(resid, origins, on=:evid)
    resid_origin[!, :id] = collect(1:nrow(resid_origin))
    kNN = Int(params["k-NN"])+1
    # @time for gdf in collect(SubDataFrame, groupby(resid_origin, [:network, :station, :phase]))
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

function compute_ssst(ssst::DataFrame, origins::DataFrame, resid::DataFrame, params::Dict, max_dist::Float32)
    kdtree = KDTree(cat(origins.X, origins.Y, origins.depth, dims=2)')

    println("Begin updating SSSTs")
    results = @showprogress @distributed (append!) for row in eachrow(origins)
        local_ssst = update(params, ssst, row, max_dist, kdtree, resid, origins)
        [(row.evid, local_ssst)]
    end

    return results
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
        for phase in eachrow(ssst[ssst.evid .== evid, :])
            key = (phase.network, phase.station, phase.phase)
            if haskey(local_ssst, key)
                ssst.residual[phase.idx] += local_ssst[key]
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
        row.residual = ssst_dict[(row.network, row.station, row.phase)]
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

function filter_catalog!(params::Dict, cat_assoc::DataFrame)
    if haskey(params, "lat_min_filter")
        filter!(x -> (x.latitude >= params["lat_min_filter"]) & (x.latitude < params["lat_max_filter"]), cat_assoc)
    end
    if haskey(params, "lon_min_filter")
        filter!(x -> (x.longitude >= params["lon_min_filter"]) & (x.longitude < params["lon_max_filter"]), cat_assoc)
    end
    if haskey(params, "depth_min_filter")
        filter!(x -> x.depth >= params["depth_min_filter"], cat_assoc)
    end
    if haskey(params, "depth_max_filter")
        filter!(x -> x.depth < params["depth_max_filter"], cat_assoc)
    end
end

function remove_outlier_picks(params, phases, origins, max_resid)
    filter!(x -> abs(x.residual) <= max_resid, phases)
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

function remove_outlier_picks(params, phases, origins)
    phases = subset(groupby(phases, :evid), :residual => x -> abs.(x) .<= Float32(params["outlier_ndev"]) * mad(x))
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

function remove_duplicate_picks(params, phases, origins, residuals)
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
    return phases, origins
end

function read_phases(params, cat_assoc)
    phases = CSV.read(params["phase_file"], DataFrame)
    phases.phase = uppercase.(phases.phase)
    unique!(phases)
    filter!(:evid => in(Set(cat_assoc.evid)), phases)
    return phases
end

function locate_events(params, cat_assoc::DataFrame, phases::DataFrame, stations::DataFrame, phase_unc::Float32, method; outfile=nothing)
    model_state = JLD2.load(params["model_file"], "model_state");
    scaler = data_scaler(params)
    eikonet = EikoNet1D(build_model(), scaler.scale)
    Flux.loadmodel!(eikonet, model_state);

    # Loop over events
    origin_df = DataFrame(time=DateTime[], evid=Int[], latitude=Float32[], longitude=Float32[], depth=Float32[],
                          magnitude=Float32[], unc_h_max=Float32[], unc_h_min=Float32[], unc_z=Float32[], X=Float32[], Y=Float32[],
                          rmse=Float32[], mae=Float32[])
    resid_df = DataFrame(arid=Int[], evid=Int[], network=String[], station=String[], phase=String[], residual=Float32[])

    println("Begin HypoSVI")
    results = []
    results = @showprogress @distributed (append!) for phase_sub in groupby(phases, :evid)
    # for phase_sub in groupby(phases, :evid)
        X_inp, T_obs, T_ref, phase_key = format_arrivals(params, DataFrame(phase_sub), stations, method)
        origin, resid = locate(params, phase_sub.evid[1], X_inp, T_obs, eikonet, T_ref, phase_unc, method)
        mag = filter(row -> row.evid == phase_sub.evid[1], cat_assoc).mag[1]
        temp_origin_df = DataFrame(time=origin.time, evid=phase_sub.evid[1], latitude=origin.lat, longitude=origin.lon,
                                   depth=origin.depth, magnitude=mag, unc_h_max=origin.unc_h_max, unc_h_min=origin.unc_h_min,
                                   unc_z=origin.unc_z, X=origin.X, Y=origin.Y, rmse=std(resid), mae=mean(abs.(resid)))
        temp_resid_df = DataFrame(arid=Int[], evid=Int[], network=String[], station=String[], phase=String[], residual=Float32[])
        for (i, row) in enumerate(eachrow(phase_key))
            push!(temp_resid_df, (row.arid, phase_sub.evid[1], row.network, row.station, row.phase, resid[i]))
        end
        [(temp_origin_df, temp_resid_df)]
        # push!(results, (temp_origin_df, temp_resid_df))
    end

    for (local_origin_df, local_resid_df) in results
        append!(origin_df, local_origin_df)
        append!(resid_df, local_resid_df)
    end

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

function locate_events_ssst_shrinking(pfile; stop=nothing)
    # NOT FULLY TESTED
    params = JSON.parsefile(pfile)
    outfile = params["catalog_outfile"]
    resid_file = params["resid_outfile"]

    cat_assoc = CSV.read(params["catalog_infile"], DataFrame)
    if ~isnothing(stop)
        cat_assoc = cat_assoc[1:stop,:]
    end
    if params["verbose"]
        println("Read in ", nrow(cat_assoc), " events")
    end
    filter_catalog!(params, cat_assoc)
    if params["verbose"]
        println("After filtering ", nrow(cat_assoc), " events remaining")
    end
    phases = read_phases(params, cat_assoc)
    select!(phases, [:network, :station, :phase, :time, :evid])
    phases[!, :arid] = collect(1:nrow(phases))

    stations = get_stations(params)
    unique!(stations)

    if params["max_k-NN_dist"] == params["min_k-NN_dist"]
        ssst_radius = fill(Float32(params["min_k-NN_dist"]), params["n_ssst_iter"])
    else
        ssst_radius = logrange(Float32(params["min_k-NN_dist"]), Float32(params["max_k-NN_dist"]), params["n_ssst_iter"])
        ssst_radius = reverse(collect(ssst_radius))
    end

    # Initial removal of duplicate picks
    origins, residuals = locate_events(params, cat_assoc, phases, stations, outfile="$(outfile)_iter_a")
    println("MAD residual before duplicate removal ", mad(residuals.residual))
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

    # Removal of outlier picks
    origins, residuals = locate_events(params, cat_assoc, phases, stations, outfile="$(outfile)_iter_b")
    println("MAD residual before outlier removal ", mad(residuals.residual))
    # println("Removing picks with ", Float32(params["outlier_ndev"]) * mad(residuals.residual))
    # select!(phases, Not(:residual))
    # phases = innerjoin(phases, residuals, on=[:arid, :evid, :network, :station, :phase])
    # remove_outlier_picks!(params, phases, cat_assoc, Float32(params["outlier_ndev"]) * mad(residuals.residual))

    # origins, residuals = locate_events(params, cat_assoc, phases, stations, outfile="$(outfile)_iter_c")
    # select!(phases, Not(:residual))
    # phases = innerjoin(phases, residuals, on=[:arid, :evid, :network, :station, :phase])
    # println("MAD residual after outlier removal ", mad(residuals.residual))

    phases0 = deepcopy(phases)

    ssst = init_ssst(phases, residuals)
    origins, residuals = nothing, nothing
    for i in 1:3
        if i != 1
            new_ssst = init_ssst(phases, residuals)
            ssst = update_ssst(new_ssst, ssst, origins)
        end
        phases = apply_ssst(phases0, ssst)
        CSV.write("$(resid_file)_iter_static.csv", ssst)
        origins, residuals = locate_events(params, cat_assoc, phases, stations, outfile="$(outfile)_iter_static")
        println("MAD residual after static correction $i ", mad(residuals.residual))
    end

    # SSST iterations
    for k in 1:params["n_ssst_iter"]
        new_ssst = compute_ssst(ssst, origins, residuals, params, ssst_radius[k])
        ssst = update_ssst(new_ssst, ssst, origins)
        phases = apply_ssst(phases0, ssst)
        CSV.write("$(resid_file)_iter_$(k).csv", ssst)
        origins, residuals = locate_events(params, cat_assoc, phases, stations, outfile="$(outfile)_iter_$(k)")
        println("MAD residual for iter $(k) radius ",  ssst_radius[k], ": ", mad(residuals.residual))
    end

    # # Final locations after outlier removal
    # phases = innerjoin(phases, residuals, on=[:evid, :network, :station, :phase])
    # remove_outlier_picks!(params, phases, cat_assoc, Float32(params["outlier_ndev"]) * mad(residuals.residual))
    # origins, residuals = locate_events(params, cat_assoc, phases, stations, outfile="$(outfile)_iter_final")
    # println("MAD residual after outlier removal ", mad(residuals.residual))

end

function locate_events_ssst_knn(pfile; stop=nothing, remove_outliers=false, initial_screening=false)
    params = JSON.parsefile(pfile)
    outfile = params["catalog_outfile"]
    resid_file = params["resid_outfile"]

    cat_assoc = CSV.read(params["catalog_infile"], DataFrame)
    if ~isnothing(stop)
        cat_assoc = cat_assoc[1:stop,:]
    end
    if params["verbose"]
        println("Read in ", nrow(cat_assoc), " events")
    end
    filter_catalog!(params, cat_assoc)
    if params["verbose"]
        println("After filtering ", nrow(cat_assoc), " events remaining")
    end
    phases = read_phases(params, cat_assoc)
    select!(phases, [:network, :station, :phase, :time, :evid])
    phases[!, :arid] = collect(1:nrow(phases))

    stations = get_stations(params)
    unique!(stations)

    phase_unc = Float32(params["phase_unc"])

    # Initial event locations
    origins, residuals = locate_events(params, cat_assoc, phases, stations, phase_unc, MAP4p, outfile="$(outfile)_iter_a")
    println("MAD residual initial ", mad(residuals.residual))

    if initial_screening
        phases, cat_assoc = remove_duplicate_picks(params, phases, cat_assoc, residuals)

        max_mad = Float32(params["max_resid"])
        println("Removing picks with ", max_mad)
        if "residual" in Set(names(phases))
            select!(phases, Not(:residual))
        end
        phases = innerjoin(phases, residuals, on=[:arid, :evid, :network, :station, :phase])
        phases, cat_assoc = remove_outlier_picks(params, phases, cat_assoc, max_mad)

        origins, residuals = locate_events(params, cat_assoc, phases, stations, phase_unc, MAP4p, outfile="$(outfile)_iter_b")
        if "residual" in Set(names(phases))
            select!(phases, Not(:residual))
        end
        phases_keys = names(phases)
        phases = innerjoin(phases, residuals, on=[:arid, :evid, :network, :station, :phase])
        select!(phases, phases_keys)
        println("MAD residual after outlier removal ", mad(residuals.residual))
    end

    phase_unc = 1f0 * mad(residuals.residual)
    phases0 = deepcopy(phases)
    ssst = init_ssst(phases, residuals)
    for i in 1:params["n_static_iter"]
        if i != 1
            new_ssst = init_ssst(phases, residuals)
            ssst = update_ssst(new_ssst, ssst, origins)
        end
        phases = apply_ssst(phases0, ssst)
        CSV.write("$(resid_file)_iter_static.csv", ssst)
        origins, residuals = locate_events(params, cat_assoc, phases, stations, phase_unc, MAP4p, outfile="$(outfile)_iter_static")
        println("MAD residual after static correction $i ", mad(residuals.residual), " ", std(residuals.residual))
        phase_unc = 1f0 * mad(residuals.residual)
    end

    if remove_outliers
        phases, cat_assoc = remove_duplicate_picks(params, phases, cat_assoc, residuals)

        max_mad = min(Float32(params["outlier_ndev"]) * mad(residuals.residual), Float32(params["max_resid"]))
        println("Removing picks with ", max_mad)
        if "residual" in Set(names(phases))
            select!(phases, Not(:residual))
        end
        phases = innerjoin(phases, residuals, on=[:arid, :evid, :network, :station, :phase])
        phases, cat_assoc = remove_outlier_picks(params, phases, cat_assoc, max_mad)

        origins, residuals = locate_events(params, cat_assoc, phases, stations, phase_unc, MAP4p, outfile="$(outfile)_iter_b")
        if "residual" in Set(names(phases))
            select!(phases, Not(:residual))
        end
        ssst_keys = names(ssst)
        ssst = innerjoin(phases, ssst, on=[:evid, :network, :station, :phase])
        select!(ssst, ssst_keys)
        phases_keys = names(phases0)
        phases0 = innerjoin(phases, residuals, on=[:arid, :evid, :network, :station, :phase])
        select!(phases0, phases_keys)
        println("MAD residual after outlier removal ", mad(residuals.residual))
        phase_unc = 1f0 * mad(residuals.residual)
    end

    # SSST iterations
    for k in 1:params["n_ssst_iter"]
        new_ssst = compute_ssst(ssst, origins, residuals, params)
        ssst = update_ssst(new_ssst, ssst, origins)
        phases = apply_ssst(phases0, ssst)
        CSV.write("$(resid_file)_iter_$(k).csv", ssst)
        origins, residuals = locate_events(params, cat_assoc, phases, stations, phase_unc, MAP4p, outfile="$(outfile)_iter_$(k)")
        println("MAD residual for ssst iter $(k) : ", mad(residuals.residual), " ", std(residuals.residual))
        phase_unc = 1f0 * mad(residuals.residual)
    end

    origins, residuals = locate_events(params, cat_assoc, phases, stations, phase_unc, SVI, outfile="$(outfile)_svi")
    return
end

function locate_events_svi(pfile; stop=nothing)
    params = JSON.parsefile(pfile)
    outfile = params["catalog_outfile"]
    resid_file = params["resid_outfile"]

    cat_assoc = CSV.read(params["catalog_infile"], DataFrame)
    if ~isnothing(stop)
        cat_assoc = cat_assoc[1:stop,:]
    end
    if params["verbose"]
        println("Read in ", nrow(cat_assoc), " events")
    end
    filter_catalog!(params, cat_assoc)
    if params["verbose"]
        println("After filtering ", nrow(cat_assoc), " events remaining")
    end
    phases = read_phases(params, cat_assoc)
    select!(phases, [:network, :station, :phase, :time, :evid])
    phases[!, :arid] = collect(1:nrow(phases))

    stations = get_stations(params)
    unique!(stations)

    phase_unc = Float32(params["phase_unc"])

    origins, residuals = locate_events(params, cat_assoc, phases, stations, phase_unc, SVI, outfile="$(outfile)_svi")
    return
end

end
