module SuperGamma
__precompile__

using DataFrames
using CSV
using JSON
using Zygote
using Flux
using Geodesy
using BSON
using Dates
using StatsBase
using Plots
using Combinatorics
using LinearAlgebra
using Distributions
using NearestNeighbors
using Clustering
using Optim

include("./Input.jl")
include("./Eikonet.jl")
include("./Adam.jl")

abstract type InversionMethod end
abstract type EM <: InversionMethod end
abstract type SGD <: InversionMethod end

function get_origin_time(X::AbstractArray{Float32}, eikonet, scaler::MinmaxScaler,
                         T_obs::AbstractArray{Float32})
    ###########
    # Note this function is different from the one in HypoSVI
    ###########
    T_pred = Eikonet.solve(X, eikonet, scaler)
    T_pred = dropdims(T_pred, dims=1)
    T_obs = reshape(T_obs, :, 1)
    T_obs = repeat(T_obs, 1, size(X, 3))
    resid = T_obs - T_pred
    origin_offset = median(resid, dims=1)
    return origin_offset
end

function huber(x)
    x_abs = abs(x)
    if x_abs <= 1f0
        return 5f-1 * x^2
    else
        return x_abs - 5f-1
    end
end

function huber(x::Float32, δ::Float32)
    x_abs = abs(x)
    if x_abs <= δ
        return 5f-1 * x^2
    else
        return δ*x_abs - 5f-1 * δ^2
    end
end

function huber_pdf(x::Float32)
    # Assumes δ = 1
    huber_pdf(x, 1f0)
end

function huber_pdf(x::Float32, δ::Float32)
    Φ = cdf(Normal(0f0, 1f0), δ)
    ϕ = pdf(Normal(0f0, 1f0), δ)
    exp(-huber(x, δ)) / (2f0 * sqrt(2f0 * π) * (Φ - ϕ/δ - 5f-1))
end

function plot_events(origins::DataFrame)
    scatter(origins[!,:lon], origins[!,:lat], left_margin = 20Plots.mm)
    origins = CSV.read("/scratch/zross/oak_ridge/scsn_cat.csv", DataFrame)
    scatter!(origins[!,:lon], origins[!,:lat], left_margin = 20Plots.mm)
    savefig("events.png")
end

function trace_const_vel(X_t::Array{Float32}, scaler::MinmaxScaler)
    X = cat(map(x->inverse(x, scaler), eachslice(X_t, dims=3))..., dims=3)
    vel = [Float32(6000.0), Float32(6000/sqrt(3))]
    v = vel[1 .+ Int.(X[7,:,:])]
    T_pred = sqrt.((X[1,:,:] - X[4,:,:]).^2 + (X[2,:,:] - X[5,:,:]).^2 + (X[3,:,:] - X[6,:,:]).^2) ./ v
    return T_pred
end

function k_means(params::Dict,
                 X_phase::Array{Float32},
                 T_obs::Array{Float32},
                 eikonet,
                 scaler::MinmaxScaler,
                 T_ref,
                 picks::DataFrame)

    X = init_X(params, X_phase)

    for (i, slice) in enumerate(eachslice(X[2:end,:,:], dims=3))
        X[2:end,:,i] = forward(slice, scaler)
    end

    b = Float32(params["phase_unc"])
    K = params["n_particles"]
    n_obs = size(X_phase, 1)
    dist = Laplace(0f0, b)
    # dist = Normal(0f0, b)

    ϕ = ones(Float32, K) ./ K
    γ = zeros(Float32, n_obs, K) ./ K
    γ_best = γ
    idx_best = 0
    log_L_best = Inf

    nn = Int(n_obs/2)
    γ[1:nn,1] .= 1.0
    if K > 1
        γ[nn+1:end,2] .= 1.0
    end

    X_best = X
    for iter in 1:500
        T_pred = dropdims(Eikonet.solve(X[2:end,:,:], eikonet, scaler), dims=1) + X[1,:,:]
        # T_pred = trace_const_vel(X[2:end,:,:]) + X[1,:,:]
        γ = assign_γ(T_pred, T_obs)
        Nₖ = sum(γ, dims=1)
        ϕ .= vec(Nₖ / n_obs)
        X_src, log_L = locate_events(params, X, T_obs, γ, scaler, eikonet)
        # X_src, log_L = locate_events(params, X, T_obs, γ, scaler)
        X[1:4,:,:] .= X_src
        println("iter $iter $log_L $log_L_best ", ϕ)
        if log_L < log_L_best
            X_best .= X
            idx_best = iter
            log_L_best = log_L
            γ_best .= γ
        end
    end

    X = X_best
    γ = γ_best
    Nₖ = vec(sum(γ, dims=1))
    ϕ = vec(Nₖ / size(T_obs, 1))

    for (i, slice) in enumerate(eachslice(X[2:end,:,:], dims=3))
        X[2:end,:,i] = inverse(slice, scaler)
    end

    # Reduce X over arrivals
    X = mean(X, dims=2)[1:4,1,:]

    γ = argmax(γ, dims=2)
    γ = vec([x[2] for x in γ])
    #display(round.(Nₖ))
    # display(sort(ϕ))
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    inv_trans = LLAfromENU(origin, wgs84)
    idx = findall(Nₖ .>= params["n_det"])
    results = []
    for i in idx
        arrivals = picks[findall(γ .== i), [:network, :station, :phase, :time]]
        hypo_lla = inv_trans(ENU(X[2,i], X[3,i], 0f0))
        hypo = Origin(Float32(hypo_lla.lat), Float32(hypo_lla.lon), X[4,i]/1f3,
                      T_ref + sec2date(X[1,i]), NaN, X[2,i]/1f3, X[3,i]/1f3)
        push!(results, (hypo, arrivals))
    end
    return results
end

function init_X(params::Dict, X_phase::Array{Float32})
    lat0 = 0.5(params["lat_min"] + params["lat_max"])
    lon0 = 0.5(params["lon_min"] + params["lon_max"])
    dlat = 0.05(params["lat_max"] - params["lat_min"])
    dlon = 0.05(params["lon_max"] - params["lon_min"])
    z0 = params["z_max"]
    K = params["n_particles"]
    n_obs = size(X_phase, 1)
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)

    X_src = zeros(Float32, 4, n_obs, K)
    for i in 1:K
        # lat1 = rand(Uniform(lat0-dlat, lat0+dlat))
        # lon1 = rand(Uniform(lon0-dlon, lon0+dlon))
        lat1 = lat0
        lon1 = lon0
        z1 = 0.5(params["z_min"] + params["z_max"])
        point_enu = trans(LLA(lat=lat1, lon=lon1))
        X_src[2,:,i] .= point_enu.e
        X_src[3,:,i] .= point_enu.n
        X_src[4,:,i] .= z1*1f3
        # X_src[4,:,i] .= rand(Uniform(params["z_min"], z0)).*1f3
    end
    if K > 1
        X_src[1,:,:] .= reshape(collect(range(-30.0, 20.0, length=K)), 1, K)
    end

    X_phase = reshape(X_phase', 4, n_obs, 1)
    X_phase = repeat(X_phase, 1, 1, K)
    X = cat(X_src, X_phase, dims=1)
    return X
end

function locate_events(params::Dict, X::Array{Float32}, T_obs::Array{Float32}, γ::Array{Float32}, scaler::MinmaxScaler, eikonet) #tracer::Function)
    # Loop for one M-step
    z_min_tr = (params["z_min"]*1f3 - scaler.min)/scaler.scale
    z_max_tr = (params["z_max"]*1f3 - scaler.min)/scaler.scale
    η = Float32(params["lr"])
    opt = Adam(mean(X[1:4,:,:], dims=2), η)
    X_last = mean(X[1:4,:,:], dims=2)
    ll = 0f0
    dist = Laplace(0f0, Float32(params["phase_unc"]))
    for i in 1:params["n_epochs"]
        function loss(X::AbstractArray{Float32})
            T_pred = dropdims(Eikonet.solve(X[2:end,:,:], eikonet, scaler), dims=1) .+ X[1,:,:]
            Q_θ = -logpdf.(dist, T_pred .- T_obs') .* γ
            #Q_θ = huber.((T_pred .- T_obs')./3f-1) .* γ
            return mean(Q_θ)
        end
        ll, ∇Q_θ = withgradient(loss, X)
        ∇Q_θ = sum(∇Q_θ[1][1:4,:,:], dims=2)

        step!(opt, ∇Q_θ)
        opt.theta[2:3,:,:] = clamp.(opt.theta[2:3,:,:], 0f0, 1f0)
        opt.theta[4,:,:] = clamp.(opt.theta[4,:,:], z_min_tr, z_max_tr)
        X[1:4,:,:] .= opt.theta

        ℓ² = sqrt.(sum((opt.theta[2:4,:,:] - X_last[2:4,:,:]).^2))
        Δr = ℓ² * scaler.scale / 1f3
        if Δr < params["iter_tol"] && i > 50
            break
        end
        X_last = opt.theta
    end
    return X_last, ll
end


function locate_events(params::Dict, X::Array{Float32}, T_obs::Array{Float32}, ν::Array{Float32}, γ::Array{Float32},
                       scaler::MinmaxScaler, eikonet, dist::Distribution)
    # Loop for one M-step
    z_min_tr = (params["z_min"]*1f3 - scaler.min)/scaler.scale
    z_max_tr = (params["z_max"]*1f3 - scaler.min)/scaler.scale
    η = Float32(params["lr"])
    opt = Adam(mean(X[1:4,:,:], dims=2), η)
    X_best = mean(X[1:4,:,:], dims=2)
    ℓ_best = Inf

    function loss(X_in::AbstractArray{Float32})
        T_pred = dropdims(Eikonet.solve(X_in[2:end,:,:], eikonet, scaler), dims=1) .+ X_in[1,:,:]
        # Q_θ = -logpdf.(dist, T_pred .- T_obs') .* γ .* ν
        Q_θ = huber.((T_pred .- T_obs')./Float32(params["phase_unc"])) .* γ .* ν
        return sum(Q_θ)
    end

    for i in 1:params["n_epochs"]
        ℓ, ∇Q_θ = withgradient(loss, X)
        ∇Q_θ = sum(∇Q_θ[1][1:4,:,:], dims=2)
        step!(opt, ∇Q_θ)
        opt.theta[2:3,:,:] = clamp.(opt.theta[2:3,:,:], 0f0, 1f0)
        opt.theta[4,:,:] = clamp.(opt.theta[4,:,:], z_min_tr, z_max_tr)
        X[1:4,:,:] .= opt.theta

        if ℓ < ℓ_best
            X_best = opt.theta
            ℓ_best = ℓ
        end
        # println("$i $ℓ $ℓ_best")
    end
    return X_best, ℓ_best
end

function locate_events(params::Dict, X::Array{Float32}, T_obs::Array{Float32}, ν::Array{Float32}, γ::Array{Float32},
                       scaler::MinmaxScaler, dist::Distribution)
    # Loop for one M-step
    z_min_tr = (params["z_min"]*1f3 - scaler.min)/scaler.scale
    z_max_tr = (params["z_max"]*1f3 - scaler.min)/scaler.scale
    η = Float32(params["lr"])
    opt = Adam(mean(X[1:4,:,:], dims=2), η)
    X_best = mean(X[1:4,:,:], dims=2)
    ℓ_best = Inf
    function loss(X::AbstractArray{Float32})
        T_pred = trace_const_vel(X[2:end,:,:], scaler) + X[1,:,:]
        # Q_θ = -logpdf.(dist, T_pred .- T_obs') .* γ .* ν
        Q_θ = huber.((T_pred .- T_obs')./Float32(params["phase_unc"])) .* γ .* ν
        return sum(Q_θ)
    end
    for i in 1:params["n_epochs"]
        ℓ, ∇Q_θ = withgradient(loss, X)
        ∇Q_θ = sum(∇Q_θ[1][1:4,:,:], dims=2)

        step!(opt, ∇Q_θ)
        opt.theta[2:3,:,:] = clamp.(opt.theta[2:3,:,:], 0f0, 1f0)
        opt.theta[4,:,:] = clamp.(opt.theta[4,:,:], z_min_tr, z_max_tr)
        X[1:4,:,:] .= opt.theta
        if ℓ < ℓ_best
            X_best = opt.theta
            ℓ_best = ℓ
        end
    end
    return X_best, ℓ_best
end

function assign_γ(T_pred::Array{Float32}, T_obs::Array{Float32})
    γ = zeros(Float32, size(T_pred))
    resid = abs.(T_pred .- T_obs')
    idx = argmin(resid, dims=2)
    γ[idx] .= 1f0
    return γ
end

function assign_γ(T_pred::Array{Float32}, T_obs::Array{Float32}, ϕ::Array{Float32}, dist::Distribution)
    pr = pdf.(dist, T_pred .- T_obs') .* reshape(ϕ, 1, :)  .+ 1f-8
    denom = sum(pr, dims=2)
    γ = pr ./ denom
    return γ
end

function assign_ν(T_pred::Array{Float32}, T_obs::Array{Float32}, dist::Distribution, ϕ::Array{Float32}, w::Float32)
    unif_pdf = Float32(1.0 / (maximum(T_obs) - minimum(T_obs)))
    pr = pdf.(dist, T_pred .- T_obs') .+ 1f-8
    pr = pr .* reshape(ϕ, 1, :)
    pr = sum(pr, dims=2)
    p_x_Gi = sum(pr, dims=2)
    ν = w * p_x_Gi ./ (w * p_x_Gi .+ (1f0 - w) .* unif_pdf)
    return vec(ν)
end

function GMM(params, X_phase::Array{Float32}, T_obs::Array{Float32}, eikonet, scaler::MinmaxScaler, T_ref, picks::DataFrame)

    X = init_X(params, X_phase)
    for (i, slice) in enumerate(eachslice(X[2:end,:,:], dims=3))
        X[2:end,:,i] = forward(slice, scaler)
    end

    K = params["n_particles"]
    n_obs = size(X_phase, 1)
    dist = Laplace(0f0, Float32(params["phase_unc"]))

    ϕ = ones(Float32, K) ./ K
    γ_best = γ = ones(Float32, n_obs, K) ./ K
    ν_best = ν = ones(Float32, n_obs)
    w = sum(ν_best) / n_obs
    ℓ_best = -Inf
    i_best = 0

    nn = Int(n_obs/2)
    γ[1:nn,1] .= 1.0
    if K > 1
        γ[nn+1:end,2] .= 1.0
    end

    X_best = X
    unif_pdf = Float32(1.0 / (maximum(T_obs) - minimum(T_obs)))

    for iter in 1:params["n_warmup_iter"]
        T_pred = dropdims(Eikonet.solve(X[2:end,:,:], eikonet, scaler), dims=1) + X[1,:,:]
        pr = pdf.(dist, T_pred .- T_obs') .* reshape(ϕ, 1, :)
        ℓ = sum(log.(w .* sum(pr, dims=2) .+ (1f0-w) * unif_pdf .+ 1f-8))
        γ = assign_γ(T_pred, T_obs)
        X_src, loss = locate_events(params, X, T_obs, ν, γ, scaler, dist)
        X[1:4,:,:] .= X_src
        if ℓ > ℓ_best
            ℓ_best = ℓ
            X_best .= X
            γ_best .= γ
            ν_best .= ν
        end
        w = sum(ν) / n_obs
        Nₖ = sum(γ, dims=1)
        ϕ .= vec(Nₖ / n_obs)
        println("warmup iter $iter $ℓ $ℓ_best $w ", vec(Nₖ))            
    end
    w = 2f-1
    ℓ_best = -Inf

    for iter in 1:50
        T_pred = dropdims(Eikonet.solve(X[2:end,:,:], eikonet, scaler), dims=1) + X[1,:,:]
        pr = pdf.(dist, T_pred .- T_obs') .* reshape(ϕ, 1, :)
        ℓ = sum(log.(w .* sum(pr, dims=2) .+ (1f0-w) * unif_pdf .+ 1f-8))
        ν = assign_ν(T_pred, T_obs, dist, ϕ, w) 
        γ = assign_γ(T_pred, T_obs, ϕ, dist)
        X_src, loss = locate_events(params, X, T_obs, ν, γ, scaler, eikonet, dist)
        X[1:4,:,:] .= X_src
        if ℓ > ℓ_best
            ℓ_best = ℓ
            X_best .= X
            γ_best .= γ
            ν_best .= ν
            i_best = iter
        elseif (ℓ_best-ℓ) > params["iter_tol"]
            break
        end
        w = sum(ν) / n_obs
        Nₖ = sum(γ, dims=1)
        ϕ .= vec(Nₖ / n_obs)
        println("full GMM iter $iter $ℓ $ℓ_best $w ", vec(Nₖ))
    end

    X .= X_best
    γ .= γ_best
    ν .= ν_best
    Nₖ = vec(sum(γ, dims=1))
    ϕ = vec(Nₖ / size(T_obs, 1))
    w = sum(ν) / n_obs

    for (i, slice) in enumerate(eachslice(X[2:end,:,:], dims=3))
        X[2:end,:,i] = inverse(slice, scaler)
    end

    γ = argmax(γ, dims=2)
    γ = vec([x[2] for x in γ])

    plot_clusters(params, X[:,:,1], vec(T_obs), γ, ν)
    # Reduce X over arrivals
    X = mean(X, dims=2)[1:4,1,:]

    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    inv_trans = LLAfromENU(origin, wgs84)
    results = []
    for i in sort(unique(γ))
        # Now filter based on ν
        arrivals = picks[findall((γ .== i) .&& (ν .>= 0.5)), [:arid, :network, :station, :phase, :time]]
        if nrow(arrivals) < params["n_det"]
            continue
        end
        hypo_lla = inv_trans(ENU(X[2,i], X[3,i], 0f0))
        hypo = Origin(Float32(hypo_lla.lat), Float32(hypo_lla.lon), X[4,i]/1f3,
                      T_ref + sec2date(X[1,i]), NaN, X[2,i]/1f3, X[3,i]/1f3)
        push!(results, (hypo, arrivals))
    end
    return results
end

function evaluate_results(pred_arrivals, true_arrivals)
    # First do Jaccard precision
    J_p = 0.
    for group in groupby(pred_arrivals, :evid)
        J_pk = []
        for group2 in groupby(true_arrivals, :evid)
            common = length(intersect(group.arid, group2.arid))
            total = length(group.arid)
            #total = length(union(group.arid, group2.arid))
            push!(J_pk, common/total)
        end
        J_p += maximum(J_pk)
    end
    J_p /= length(unique(pred_arrivals.evid))

    J_r = 0.
    for group in groupby(true_arrivals, :evid)
        J_rk = []
        for group2 in groupby(pred_arrivals, :evid)
            common = length(intersect(group.arid, group2.arid))
            total = length(group.arid)
            #total = length(union(group.arid, group2.arid))
            push!(J_rk, common/total)
        end
        J_r += maximum(J_rk)
    end
    J_r /= length(unique(true_arrivals.evid))
    return J_p, J_r
end

function plot_clusters(params, X, T, γ, ν)
    l = @layout [a b]
    p1 = scatter(X[5,:]/1000., T, color=:black)
    p2 = scatter(X[6,:]/1000., T, color=:black)
    for idx in sort(unique(γ))
        idx2 = findall((γ .== idx) .&& (ν .>= 0.5))
        if length(idx2) < params["n_det"]
            continue
        end
        scatter!(p1, X[5,idx2]/1000., T[idx2], label=idx)
        scatter!(p2, X[6,idx2]/1000., T[idx2], label=idx)
    end
    plot(p1, p2, layout = l)
    savefig("test.png")
end

function softmax(f)
    return exp.(f) / sum(exp.(f))
end

function detect(params,
                X_phase::Array{Float32},
                T_obs::Array{Float32},
                eikonet,
                scaler::MinmaxScaler,
                T_ref,
                picks::DataFrame,
                ::Type{SGD})
    # Main function to locate events
    lat0 = 0.5(params["lat_min"] + params["lat_max"])
    lon0 = 0.5(params["lon_min"] + params["lon_max"])
    dlat = 0.05(params["lat_max"] - params["lat_min"])
    dlon = 0.05(params["lon_max"] - params["lon_min"])
    z0 = params["z_max"]
    K = params["n_particles"]
    η = Float32(params["lr"])
    n_obs = size(X_phase, 1)
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)

    X_src = zeros(Float32, 4, n_obs, K)
    for i in 1:K
        lat1 = rand(Uniform(lat0-dlat, lat0+dlat))
        lon1 = rand(Uniform(lon0-dlon, lon0+dlon))
        point_enu = trans(LLA(lat=lat1, lon=lon1))
        X_src[2,:,i] .= point_enu.e
        X_src[3,:,i] .= point_enu.n
        X_src[4,:,i] .= rand(Uniform(params["z_min"], z0)).*1f3
    end
    X_src[1,:,:] .= reshape(collect(range(-30.0, 10.0, length=K)), 1, K)

    T_obs = repeat(T_obs, K, 1)'
    b = Float32(params["phase_unc"])
    #b = fill(Float32(params["phase_unc"]), size(T_obs))

    X_phase = reshape(X_phase', 4, n_obs, 1)
    X_phase = repeat(X_phase, 1, 1, K)
    X = cat(X_src, X_phase, dims=1)

    z_min_tr = (params["z_min"]*1f3 - scaler.min)/scaler.scale
    z_max_tr = (params["z_max"]*1f3 - scaler.min)/scaler.scale

    for (i, slice) in enumerate(eachslice(X[2:end,:,:], dims=3))
        X[2:end,:,i] = forward(slice, scaler)
    end

    dist = Laplace(0f0, b)
    # dist = Normal(0.0, 0.1)

    ϕ = ones(Float32, 1, K) ./ K
    γ = ones(Float32, size(T_obs, 1), K) ./ K
    γ_best = γ
    idx_best = 0
    log_L_best = -Inf

    X_best = X
    println(size(X))
    θ = zeros(Float32, 5, 1, K)
    θ[1:4,:,:] .= mean(X[1:4,:,:], dims=2)
    θ[5,:,:] .= ϕ
    opt = Adam(θ, η)
    for i in 1:params["n_epochs"]
        T_pred = dropdims(Eikonet.solve(X[2:end,:,:], eikonet, scaler), dims=1) .+ X[1,:,:]
        L_data = pdf.(dist, T_pred-T_obs) .* ϕ .+ 1f-8
        γ .= L_data ./ (sum(L_data, dims=2))

        function loss(θ::AbstractArray{Float32})
            X_src = repeat(θ[1:4,:,:], 1, n_obs, 1)
            X_local = cat(X_src, X[5:end,:,:], dims=1)
            ϕ = softmax(θ[5,:,:])
            T_pred = dropdims(Eikonet.solve(X_local[2:end,:,:], eikonet, scaler), dims=1) .+ X_local[1,:,:]
            L_data = pdf.(dist, T_pred-T_obs) .+ 1f-8
            log_p_x = sum(log.(sum(L_data .* ϕ .* γ, dims=2) .+ 1f-8))
            return -log_p_x
        end

        nll, ∇θ = withgradient(loss, θ)
        println("$i $nll ")
        ∇θ = sum(∇θ[1], dims=2)
        step!(opt, ∇θ)
        opt.theta[2:3,:,:] = clamp.(opt.theta[2:3,:,:], 0f0, 1f0)
        opt.theta[4,:,:] = clamp.(opt.theta[4,:,:], z_min_tr, z_max_tr)
        X[1:4,:,:] .= opt.theta[1:4,:,:]
        ϕ .= softmax(opt.theta[5,:,:])
        # println(ϕ)

        # ℓ² = sqrt.(sum((opt.theta[2:4,:,:] - X_last[2:4,:,:]).^2))
        # Δr = ℓ² * scaler.scale / 1f3
        # if Δr < params["iter_tol"] && i > 50
        #     break
        # end
        # X_last = opt.theta
    end
    # X = X_best
    # γ = γ_best
    Nₖ = vec(sum(γ, dims=1))
    ϕ = vec(Nₖ / size(T_obs, 1))

    for (i, slice) in enumerate(eachslice(X[2:end,:,:], dims=3))
        X[2:end,:,i] = inverse(slice, scaler)
    end

    # Reduce X over arrivals
    X = mean(X, dims=2)[1:4,1,:]

    γ = argmax(γ, dims=2)
    γ = vec([x[2] for x in γ])
    #display(round.(Nₖ))
    # display(sort(ϕ))

    inv_trans = LLAfromENU(origin, wgs84)
    idx = findall(Nₖ .>= params["n_det"])
    results = []
    for i in idx
        arrivals = picks[findall(γ .== i), [:network, :station, :phase, :time]]
        hypo_lla = inv_trans(ENU(X[2,i], X[3,i], 0f0))
        hypo = Origin(Float32(hypo_lla.lat), Float32(hypo_lla.lon), X[4,i]/1f3,
                      T_ref + sec2date(X[1,i]), NaN, X[2,i]/1f3, X[3,i]/1f3)
        push!(results, (hypo, arrivals))
    end
    return results
end

function add_fake_picks(X_phase, T_obs, picks, scaler)
    n_fake = 100
    T_fake = rand(Uniform(-60f0, 60f0), 1, n_fake)
    X_fake = rand(Uniform(0f0, scaler.scale), n_fake, 3)
    X_fake[:,3] .= 0f0
    phase_fake = round.(rand(Float32, n_fake))
    X_fake = hcat(X_fake, phase_fake)
    T_obs = cat(T_obs, T_fake, dims=2)
    X_phase = vcat(X_phase, X_fake)
    arid_idx = maximum(picks.arid) + 1
    for i in 1:n_fake
        push!(picks, (arid_idx, "ZR", "FAKE", "P", DateTime("1989-01-16T06:48:40.380"),
                      34.4233, -118.835, -0.205, 94556.2, 118629.0, -205.0))
        arid_idx += 1
    end
    return Float32.(X_phase), Float32.(T_obs), picks
end

function generate_syn_dataset(params::Dict, stations::DataFrame, eikonet, scaler::MinmaxScaler,)
    T0 = DateTime("1986-11-20T00:00:00.0")
    phases = DataFrame(arid=Int[], network=String[], station=String[], phase=String[], time=DateTime[])
    offset = [0f0, 20f0, 30f0]
    lat0 = 0.25*(params["lat_max"] - params["lat_min"]) + params["lat_min"]
    lon0 = 0.25*(params["lon_max"] - params["lon_min"]) + params["lon_min"]
    z0 = params["z_max"]
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)
    point_enu = trans(LLA(lat=lat0, lon=lon0))
    arid_idx = 1
    event_idx = Vector{Int}()
    for i in 1:3
        origin_time = T0 + sec2date(offset[i])
        input = zeros(Float32, 7, 1)
        input[1:3] = forward_point([point_enu.e, point_enu.n, params["z_min"]*1f3], scaler)
        for row in eachrow(stations)
            input[4] = row.X / scaler.scale
            input[5] = row.Y / scaler.scale
            input[6] = row.Z / scaler.scale
            if (input[4] < 0f0) || (input[4] > 1f0)
                continue
            elseif (input[5] < 0f0) || (input[5] > 1f0)
                continue
            end
            for phase_label in ["P", "S"]
                if phase_label == "P"
                    input[7] = 0f0
                else
                    input[7] = 1f0
                end
                T_pred = Eikonet.solve(input, eikonet, scaler)[1]
                if T_pred >= 10
                    continue
                end
                arrival_time = origin_time + sec2date(T_pred)
                push!(phases, (arid_idx, row.network, row.station, phase_label, arrival_time))
                push!(event_idx, i)
                arid_idx += 1
            end
        end
    end
    return phases, event_idx
end

function detect_syn_events(pfile, outfile)
    # params = JSON.parsefile(pfile)
    params = build_supergamma_syn_params()

    if params["inversion_method"] isa String
        params["inversion_method"] = eval(Meta.parse(params["inversion_method"]))
    end
 
    model = BSON.load(params["model_file"], @__MODULE__)[:model]
    scaler = data_scaler(params)

    # phases, stations, true_events = read_syn_dataset(params)
    stations = get_stations(params)
    phases, event_idx = generate_syn_dataset(params, stations, model, scaler)
    phases = sort(phases, "time")
    # println(phases)

    # Loop over events
    origins = DataFrame(time=DateTime[], evid=Int[], lat=Float32[], lon=Float32[], depth=Float32[],
                        X=Float32[], Y=Float32[])
    assoc = DataFrame(arid=Int[], network=String[], station=String[], phase=String[], time=DateTime[], evid=Int[])
    evid = 1000000

    T_rel = map(x -> (x.time - phases[1, "time"]).value, eachrow(phases)) ./ 1000.0
    T_rel = reshape(T_rel, 1, :)
    clusters = dbscan(T_rel, 5, min_neighbors=1, min_cluster_size=params["n_det"])
    println("Begin association with $(length(clusters)) clusters")

    #for cluster in clusters
    for dummy in 1:1
        phase_sub = phases
        X_phase, T_obs, T_ref, picks = format_arrivals(DataFrame(phase_sub), stations)
        dist = Laplace(0f0, Float32(params["phase_unc"]))
        T_obs .+= rand(dist, size(T_obs))
        X_phase, T_obs, picks = add_fake_picks(X_phase, T_obs, picks, scaler)
        results = GMM(params, X_phase, T_obs, model, scaler, T_ref, picks)
        for i in eachindex(results)
            hypo, arrivals = results[i]
            insertcols!(arrivals, :evid => fill(evid, nrow(arrivals)))
            assoc = vcat(assoc, arrivals)
            push!(origins, (hypo.time, evid, hypo.lat, hypo.lon, hypo.depth, hypo.X, hypo.Y))
            println(arrivals)
            println(last(origins, 1))
            println(nrow(arrivals))
            println()
            evid += 1
        end
        if params["verbose"]
            println(last(origins, length(results)))
            println()
        end
    end

    insertcols!(phases, :evid => event_idx)
    J_p, J_r = evaluate_results(assoc, phases)
    println("Jp: $J_p")
    println("Jr: $J_r")

    if params["verbose"]
        println(first(origins, 100))
    end
    plot_events(origins)
    CSV.write(outfile, assoc)

    return origins
end


function detect_events(pfile, outfile)
    # params = JSON.parsefile(pfile)
    params = build_supergamma_params()

    if params["inversion_method"] isa String
        params["inversion_method"] = eval(Meta.parse(params["inversion_method"]))
    end
 
    # Read in a pick file in csv format (e.g. from Gamma or PhaseLink)
    phases = CSV.read(params["phase_file"], DataFrame)

    println(first(phases, 5), "\n")

    stations = get_stations(params)
    unique!(stations, [:network, :station])
    println(first(stations, 5), "\n")

    model = BSON.load(params["model_file"], @__MODULE__)[:model]
    scaler = data_scaler(params)

    # Loop over events
    origins = DataFrame(time=DateTime[], evid=Int[], lat=Float32[], lon=Float32[], depth=Float32[],
                        X=Float32[], Y=Float32[])
    assoc = DataFrame(network=String[], station=String[], phase=String[], time=DateTime[], evid=Int64)
    evid = 1000000
    ##### This iterator needs to be changed later since events are not known
    for phase_sub in groupby(phases, :evid)
        if nrow(phase_sub) < 8
            continue
        end
        # println(phase_sub)
        phase_sub2 = phase_sub
        X_phase, T_obs, T_ref, picks = format_arrivals(DataFrame(phase_sub), stations)
        X_phase, T_obs, picks = add_fake_picks(X_phase, T_obs, picks, scaler)
        results = detect(params, X_phase, T_obs, model, scaler, T_ref, picks, params["inversion_method"])
        for i in eachindex(results)
            hypo, arrivals = results[i]
            insertcols!(arrivals, :evid => fill(evid, nrow(arrivals)))
            println(arrivals)
            println(antijoin(phase_sub, arrivals, on=[:network, :station, :phase, :time]))
            assoc = vcat(assoc, arrivals)
            push!(origins, (hypo.time, evid, hypo.lat, hypo.lon, hypo.depth, hypo.X, hypo.Y))
            evid += 1
        end
        if params["verbose"]
            println(last(origins, length(results)))
        end
        println()
    end

    if params["verbose"]
        println(first(origins, 100))
    end
    plot_events(origins)
    CSV.write(outfile, assoc)

    return origins
end

end