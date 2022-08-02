module HypoSVI
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
using Distributions: Uniform
using ChainRulesCore: ignore_derivatives

include("./Input.jl")
include("./Eikonet.jl")

struct Origin
    lat::Float32
    lon::Float32
    depth::Float32
    time::DateTime
    unc_z::Float32
end

abstract type InversionMethod end
abstract type MAP <: InversionMethod end
abstract type SVI <: InversionMethod end

function sec2date(s::AbstractFloat)
    sec_sign = Int32(sign(s))
    s = abs(s)
    sec = Int32(floor(s))
    msec = Int32(floor(1000*(s - sec)))
    return Dates.Second(sec * sec_sign) + Dates.Millisecond(msec * sec_sign)
end

function locate(params, X_inp, T_obs, eikonet, scaler, T_ref, ::Type{MAP})
    # Main function to locate events
    lat0 = 0.5(params["lat_min"] + params["lat_max"])
    lon0 = 0.5(params["lon_min"] + params["lon_max"])
    z0 = params["z_min"]
    η = params["lr"]
    n_phase = size(X_inp, 1)
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)

    point_enu = trans(LLA(lat=lat0, lon=lon0))
    ipairs = collect(combinations(collect(1:n_phase), 2))
    ipairs = permutedims(hcat(ipairs...))
    ΔT_obs = T_obs[ipairs[:,1]] - T_obs[ipairs[:,2]]
    X_src = [point_enu.e, point_enu.n, z0*1e3]
    X = cat(repeat(X_src[1:3], 1, n_phase)', X_inp, dims=2)'
    X = forward(X, scaler)

    # First determine hypocenter with dtimes
    for i in 1:params["n_epochs"]
        function loss(X::AbstractArray)      
            T_pred = Eikonet.solve(X, eikonet, scaler)
            ΔT_pred = T_pred[ipairs[:,1]] - T_pred[ipairs[:,2]]
            Flux.mae(ΔT_pred, ΔT_obs, agg=mean)
        end
        LL, ∇LL = withgradient(loss, X)
        ∇LL = ∇LL[1]
        ∇LL = mean(∇LL[1:3,:], dims=2)
        X[1:3,:] .-= η .* ∇LL

        # Clip gradient at region boundary
        X[findall(X[1:3,:] .< 0.0)] .= 0f0
        X[findall(X[1:3,:] .> 1.0)] .= 1f0
    end

    # Then determine origin time given hypocenter
    T_pred = Eikonet.solve(X, eikonet, scaler)
    T_src = median(T_obs - T_pred)

    X = inverse(X, scaler)'
    X_src = X[1,1:3]
    inv_trans = LLAfromENU(origin, wgs84)
    hypo_lla = inv_trans(ENU(X_src[1], X_src[2], 0f0))
    return Origin(hypo_lla.lat, hypo_lla.lon, X_src[3]/1f3, T_ref + sec2date(T_src), NaN)
end

function compute_kernel(X::AbstractArray{Float32})
    n = size(X, 1)
    d² = zeros(Float32, n, n)
    # first compute K
    for i in 1:n
        for j in 1:n
            for k in 1:3
                d²[i,j] += (X[i,k] - X[j,k])^2
            end
        end
    end
    if length(d²[d² .> 0.0]) < 1
        κ = Inf
    else
        κ = sqrt(median(d²[d² .> 0.0]) / 2f0)
    end
    K = exp.(-d² ./ κ^2)

    # # Now compute grad K
    ∇K = zeros(Float32, 3, n, n)
    for i in 1:n
        for j in 1:n
            for k in 1:3
                ∇K[k,i,j] = 2f0 * (X[i,k] - X[j,k]) * K[i,j] / κ
            end
        end
    end
    return K, ∇K
end

function compute_kernel!(X::AbstractArray{Float32}, K::AbstractArray{Float32}, ∇K::AbstractArray{Float32})
    n = size(X, 1)
    # first compute K
    for i in 1:n
        for j in 1:n
            K[i,j] = 0f0
            for k in 1:3
                K[i,j] += (X[i,k] - X[j,k])^2
            end
        end
    end
    if n <= 1
        κ = Inf32
    else
        κ = median(K[K .> 0f0]) / 2f0
    end

    # # Now compute grad K
    for i in 1:n
        for j in 1:n
            K[i,j] = exp.(-K[i,j]/κ)
            for k in 1:3
                ∇K[k,i,j] = 2f0 * (X[i,k] - X[j,k]) * K[i,j] / κ
            end
        end
    end
end

mutable struct Adam
    theta::AbstractArray{Float32} # Parameter array
    m::AbstractArray{Float32}     # First moment
    v::AbstractArray{Float32}     # Second moment
    b1::Float32                   # Exp. decay first moment
    b2::Float32                   # Exp. decay second moment
    η::Float32                    # Step size
    eps::Float32                  # Epsilon for stability
    t::Int                        # Time step (iteration)
end
  
# Outer constructor
function Adam(theta::AbstractArray{Float32}, η=1e-3)
    m   = zeros(Float32, size(theta))
    v   = zeros(Float32, size(theta))
    b1  = 0.9f0
    b2  = 0.999f0
    eps = 1e-8
    t   = 0f0
    Adam(theta, m, v, b1, b2, η, eps, t)
end
  
function step!(opt::Adam, grads::AbstractArray{Float32})
    opt.t += 1
    gt    = grads
    opt.m = opt.b1 .* opt.m + (1 - opt.b1) .* gt
    opt.v = opt.b2 .* opt.v + (1 - opt.b2) .* gt .^ 2
    mhat = opt.m ./ (1 - opt.b1^opt.t)
    vhat = opt.v ./ (1 - opt.b2^opt.t)
    opt.theta -= opt.η .* (mhat ./ (sqrt.(vhat) .+ opt.eps))
end

function locate(params, X_inp::Array{Float32}, T_obs::Array{Float32}, eikonet, scaler, T_ref, ::Type{SVI})
    # Main function to locate events
    lat0 = 0.5(params["lat_min"] + params["lat_max"])
    lon0 = 0.5(params["lon_min"] + params["lon_max"])
    dlat = 0.05(params["lat_max"] - params["lat_min"])
    dlon = 0.05(params["lon_max"] - params["lon_min"])
    z0 = params["z_max"]
    N = params["n_particles"]
    η = Float32(params["lr"])
    n_phase = size(X_inp, 1)
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)

    X_src = zeros(Float32, 3, n_phase, N)
    for i in 1:N
        lat1 = rand(Uniform(lat0-dlat, lat0+dlat))
        lon1 = rand(Uniform(lon0-dlon, lon0+dlon))
        point_enu = trans(LLA(lat=lat1, lon=lon1))
        X_src[1,:,i] .= point_enu.e
        X_src[2,:,i] .= point_enu.n
        X_src[3,:,i] .= z0*1e3
    end

    ipairs = collect(combinations(collect(1:n_phase), 2))
    ipairs = permutedims(hcat(ipairs...))
    ΔT_obs = T_obs[ipairs[:,1]] - T_obs[ipairs[:,2]]

    X_inp = reshape(X_inp', 4, n_phase, 1)
    X_inp = repeat(X_inp, 1, 1, N)
    X = cat(X_src, X_inp, dims=1)

    for (i, b) in enumerate(eachslice(X, dims=3))
        X[:,:,i] = forward(b, scaler)
    end

    ΔT_obs = reshape(ΔT_obs, :, 1)
    ΔT_obs = repeat(ΔT_obs, 1, N)
    K = zeros(Float32, N, N)
    ∇K = zeros(Float32, 3, N, N)
    ϕ = zeros(Float32, 3, N)

    opt = Adam(mean(X[1:3,:,:], dims=2), η)

    for i in 1:params["n_epochs"]
        function loss(X::AbstractArray)
            T_pred = Eikonet.solve(X, eikonet, scaler)
            T_pred = dropdims(T_pred, dims=1)
            ΔT_pred = T_pred[ipairs[:,1],:] - T_pred[ipairs[:,2],:]
            Flux.mse(ΔT_pred, ΔT_obs, agg=sum)
        end
        ∇LL = gradient(loss, X)[1]
        ∇LL = dropdims(sum(∇LL[1:3,:,:], dims=2), dims=2)'

        X_src = dropdims(mean(X[1:3,:,:], dims=2), dims=2)'
        K, ∇K = compute_kernel(X_src)
        # compute_kernel!(X_src, K, ∇K)

        for j in 1:N
            for l in 1:3
                ϕ[l,j] = 0f0
                for k in 1:N
                    ϕ[l,j] += K[k,j] * ∇LL[k,l] - ∇K[l,k,j]
                end
                ϕ[l,j] /= N
            end
        end

        step!(opt, Flux.unsqueeze(ϕ, 2), )
        X[1:3,:,:] .= opt.theta
        #X[1:3,:,:] .-= η * Flux.unsqueeze(ϕ, 2)

        X[findall(X[1:3,:,:] .< 0.0)] .= 0f0
        X[findall(X[1:3,:,:] .> 1.0)] .= 1f0
    end

    # Then determine origin time given hypocenter
    T_pred = Eikonet.solve(X, eikonet, scaler)
    T_pred = dropdims(T_pred, dims=1)
    T_obs = reshape(T_obs, :, 1)
    T_obs = repeat(T_obs, 1, N)
    T_src = median(T_obs - T_pred)
    for (i, b) in enumerate(eachslice(X, dims=3))
        X[:,:,i] = inverse(b, scaler)
    end

    X = mean(X, dims=2)[1:3,1,:]
    X_mean = dropdims(mean(X, dims=2), dims=2)
    z_unc = StatsBase.std(X[3,:,:]) / 1000f0

    inv_trans = LLAfromENU(origin, wgs84)
    hypo_lla = inv_trans(ENU(X_mean[1], X_mean[2], 0f0))
    return Origin(Float32(hypo_lla.lat), Float32(hypo_lla.lon), X_mean[3]/1f3, T_ref + sec2date(T_src), z_unc)
end

function timedelta(t1::DateTime, t2::DateTime)
    # returns total seconds between t1,t2
    (t1-t2) / Millisecond(1000)
end

function prepare_event_data(phases::DataFrame, stations::DataFrame)
    phase_sta = innerjoin(phases, stations, on = [:network, :station])
    X_inp = zeros(Float32, size(phase_sta, 1), 4)
    X_inp[:,1] .= phase_sta.X
    X_inp[:,2] .= phase_sta.Y
    X_inp[:,3] .= phase_sta.Z
    arrival_times = DateTime.(phase_sta[!, "time"])
    T_obs = zeros(Float32, 1, length(arrival_times))
    for (i, row) in enumerate(eachrow(phase_sta))
        if row.phase == "P"
            X_inp[i,4] = 0
        else
            X_inp[i,4] = 1
        end
        T_obs[i] = timedelta(arrival_times[i], minimum(arrival_times))
    end
    T_ref = minimum(arrival_times)
    return X_inp, T_obs, T_ref
end

function update!(ssst::DataFrame, origins::DataFrame, stations::DataFrame, phases::DataFrame)

end

function plot_events(origins::DataFrame)
    scatter(origins[!,:lon], origins[!,:lat], left_margin = 20Plots.mm)
    origins = CSV.read("/scratch/zross/oak_ridge/scsn_cat.csv", DataFrame)
    scatter!(origins[!,:lon], origins[!,:lat], left_margin = 20Plots.mm)
    savefig("events.png")
end

function locate_events(pfile, outfile)
    # params = JSON.parsefile(pfile)
    params = build_hyposvi_params()

    if params["inversion_method"] isa String
        params["inversion_method"] = eval(Meta.parse(params["inversion_method"]))
    end
 
    # Read in a pick file in csv format (e.g. from Gamma or PhaseLink)
    phases = CSV.read(params["phase_file"], DataFrame)
    println(first(phases, 5), "\n")

    stations = get_stations(params)
    println(first(stations, 5), "\n")

    model = BSON.load(params["model_file"], @__MODULE__)[:model]
    scaler = data_scaler(params)

    # Loop over events
    origins = DataFrame(time=DateTime[], evid=Int[], lat=Float32[], lon=Float32[], depth=Float32[], z_unc=Float32[])
    for phase_sub in groupby(phases, :evid)
        if params["verbose"]
            println(phase_sub, "\n")
        end
        X_inp, T_obs, T_ref = prepare_event_data(DataFrame(phase_sub), stations)
        @time origin = locate(params, X_inp, T_obs, model, scaler, T_ref, params["inversion_method"])

        push!(origins, (origin.time, phase_sub.evid[1], origin.lat, origin.lon, origin.depth, origin.unc_z))
        println(last(origins, 1), "\n")
    end

    println(first(origins, 100))
    CSV.write(outfile, origins)
    plot_events(origins)
    return origins
end

function locate_events_ssst(pfile, outfile)
    params = build_hyposvi_params()

    # Read in a pick file in csv format (e.g. from Gamma or PhaseLink)
    phases = CSV.read(params["phase_file"], DataFrame)
    println(first(phases, 5), "\n")

    stations = get_stations(params)
    println(first(stations, 5), "\n")

    ssst = DataFrame()
    for k in 1:pfile["ssst_iterations"]
        origins = locate_events(pfile, "$(outfile)_$k")
        update!(ssst, origins, stations, phases)
    end
end

end