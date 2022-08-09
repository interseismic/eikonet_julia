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
using ChainRulesCore: ignore_derivatives
using NearestNeighbors

include("./Input.jl")
include("./Eikonet.jl")
include("./Adam.jl")

abstract type InversionMethod end
abstract type EM <: InversionMethod end

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

function plot_events(origins::DataFrame)
    scatter(origins[!,:lon], origins[!,:lat], left_margin = 20Plots.mm)
    origins = CSV.read("/scratch/zross/oak_ridge/scsn_cat.csv", DataFrame)
    scatter!(origins[!,:lon], origins[!,:lat], left_margin = 20Plots.mm)
    savefig("events.png")
end

function detect(params,
                X_inp::Array{Float32},
                T_obs::Array{Float32},
                eikonet,
                scaler::MinmaxScaler,
                T_ref,
                picks::DataFrame,
                ::Type{EM})
    # Main function to locate events
    lat0 = 0.5(params["lat_min"] + params["lat_max"])
    lon0 = 0.5(params["lon_min"] + params["lon_max"])
    dlat = 0.05(params["lat_max"] - params["lat_min"])
    dlon = 0.05(params["lon_max"] - params["lon_min"])
    z0 = params["z_max"]
    K = params["n_particles"]
    η = Float32(params["lr"])
    n_obs = size(X_inp, 1)
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)
    b = Float32(params["phase_unc"])

    X_src = zeros(Float32, 4, n_obs, K)
    for i in 1:K
        lat1 = rand(Uniform(lat0-dlat, lat0+dlat))
        lon1 = rand(Uniform(lon0-dlon, lon0+dlon))
        point_enu = trans(LLA(lat=lat1, lon=lon1))
        X_src[2,:,i] .= point_enu.e
        X_src[3,:,i] .= point_enu.n
        X_src[4,:,i] .= rand(Uniform(params["z_min"], z0)).*1e3
    end
    X_src[1,:,:] .= reshape(collect(range(-30.0, 10.0, length=K)), 1, K)

    T_obs = repeat(T_obs, K, 1)'

    X_inp = reshape(X_inp', 4, n_obs, 1)
    X_inp = repeat(X_inp, 1, 1, K)
    X = cat(X_src, X_inp, dims=1)

    z_min_tr = (params["z_min"]*1e3 - scaler.min)/scaler.scale
    z_max_tr = (params["z_max"]*1e3 - scaler.min)/scaler.scale

    for (i, slice) in enumerate(eachslice(X[2:end,:,:], dims=3))
        X[2:end,:,i] = forward(slice, scaler)
    end

    ϕ = ones(Float32, K) ./ K
    γ = ones(Float32, size(T_obs, 1), K) ./ K
    log_L0 = -Inf

    for iter in 1:500
        # Loop for one E-step
        T_pred = Eikonet.solve(X[2:end,:,:], eikonet, scaler)
        T_pred = dropdims(T_pred, dims=1) .+ X[1,:,:]
        pdf = exp.(-1f0 .* huber.((T_pred-T_obs)/b))
        log_L = sum(log.(sum(reshape(ϕ, 1, :) .* pdf, dims=2)))
        for i in axes(γ, 1)
            denom = sum(ϕ .* pdf[i,:])
            for j in axes(γ, 2)
                γ[i,j] = ϕ[j] * pdf[i,j] / denom
            end
        end
        Nₖ = sum(γ, dims=1)
        ϕ .= vec(Nₖ / size(T_obs, 1))

        # println("iter $iter $log_L")
        if log_L-log_L0 < 1e-3
            break
        end 
        log_L0 = log_L

        # Loop for one M-step
        opt = Adam(mean(X[1:4,:,:], dims=2), η)
        X_last0 = mean(X[1:4,:,:], dims=2)
        X_last = X_last0
        for i in 1:params["n_epochs"]
            function loss(X::AbstractArray{Float32})
                T_pred = dropdims(Eikonet.solve(X[2:end,:,:], eikonet, scaler), dims=1) .+ X[1,:,:]
                Qz = huber.((T_pred-T_obs)/b) .* γ # This is the expectation of log L under Z
                return sum(Qz)
            end
            ∇Qz = gradient(loss, X)
            ∇Qz = ∇Qz[1][1:4,:,:]
            ∇Qz = sum(∇Qz[1:4,:,:], dims=2)

            step!(opt, ∇Qz)
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
    end
    Nₖ = vec(sum(γ, dims=1))
    ϕ = vec(Nₖ / size(T_obs, 1))
    display(ϕ')

    for (i, slice) in enumerate(eachslice(X[2:end,:,:], dims=3))
        X[2:end,:,i] = inverse(slice, scaler)
    end

    # Reduce X over arrivals
    X = mean(X, dims=2)[1:4,1,:]

    γ = argmax(γ, dims=2)
    γ = vec([x[2] for x in γ])

    origins = []
    inv_trans = LLAfromENU(origin, wgs84)
    idx = findall(Nₖ .>= params["n_det"])
    
    arrivals = Vector{DataFrame}()
    for i in idx
        push!(arrivals, picks[findall(γ .== i), [:network, :station, :phase, :time]])

        hypo_lla = inv_trans(ENU(X[2,i], X[3,i], 0f0))
        hypo = Origin(Float32(hypo_lla.lat), Float32(hypo_lla.lon), X[4,i]/1f3,
                      T_ref + sec2date(X[1,i]), NaN, X[2,i]/1f3, X[3,i]/1f3)
        push!(origins, hypo)
    end
    return origins, arrivals
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
    println(first(stations, 5), "\n")

    model = BSON.load(params["model_file"], @__MODULE__)[:model]
    scaler = data_scaler(params)

    # Loop over events
    origins = DataFrame(time=DateTime[], evid=Int[], lat=Float32[], lon=Float32[], depth=Float32[],
                        z_unc=Float32[], X=Float32[], Y=Float32[])
    assoc = DataFrame(network=String[], station=String[], phase=String[], time=DateTime[], evid=Int64)
    evid = 1000000
    for phase_sub in groupby(phases, :evid)
        X_inp, T_obs, T_ref, picks = format_arrivals(DataFrame(phase_sub), stations)
        result = detect(params, X_inp, T_obs, model, scaler, T_ref, picks, params["inversion_method"])
        detections, arrivals = result

        for j in eachindex(arrivals)
            insertcols!(arrivals[j], :evid => fill(evid, nrow(arrivals[j])))
            assoc = vcat(assoc, arrivals[j])
            evid += 1
        end

        for origin in detections
            push!(origins, (origin.time, phase_sub.evid[1], origin.lat, origin.lon,
                            origin.depth, origin.unc_z, origin.X, origin.Y))
        end

        if params["verbose"]
            println(last(origins, length(detections)))
            println()
        end
    end

    if params["verbose"]
        println(first(origins, 100))
    end
    plot_events(origins)
    CSV.write(outfile, assoc)

    return origins
end

end