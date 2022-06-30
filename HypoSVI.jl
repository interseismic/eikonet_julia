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

include("./Input.jl")
include("./Eikonet.jl")

struct Origin
    lat::Float32
    lon::Float32
    depth::Float32
end

function locate(params, X_inp, T_obs, eikonet, scaler)
    # Main function to locate events
    X_src = Vector{Float32}([0.5, 0.5, 0.5, 0.0])
    n_phase = size(X_inp, 1)
    point_lla = LLA(lat=34.3420, lon=-118.9090)
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)
    point_enu = trans(point_lla)
    X_src = [point_enu.e, point_enu.n, 17.74*1e3, 0.0]
    for i in 1:params["n_epochs"]
        function loss(X_src)
            X = cat(repeat(X_src[1:3], 1, n_phase)', X_inp, dims=2)'
            StatsBase.transform!(scaler, X)
            Flux.mse(Eikonet.solve(X, eikonet).+X_src[4], T_obs)
        end
        println(i, " ", loss(X_src))
        X = cat(repeat(X_src[1:3], 1, n_phase)', X_inp, dims=2)'
        StatsBase.transform!(scaler, X)
        display(X)
        println(Eikonet.solve(X, eikonet) .+ X_src[4])
        #display(Eikonet.solve(cat(repeat(X_src[1:3], 1, n_phase)', X_inp, dims=2)'.+X_src[4], eikonet))
        display(T_obs)
        println()
        return
        grads = gradient(loss, X_src)[1]
        X_src .-= params["lr"] * grads
    end
    hypo_lla = LLAfromENU(trans.origin_lla, wgs84)
    hypo_lla = hypo_lla(ENU(X_src[1]*trans.normalizer*1f3, X_src[2]*trans.normalizer*1f3, 0f0))
    return Origin(hypo_lla.lat, hypo_lla.lon, X_src[3]*trans.normalizer)
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

function main()
    # pfile = "params.json"
    # params = JSON.parsefile(pfile)
    params = build_params()

    # Read in a pick file in csv format (e.g. from Gamma or PhaseLink)
    phases = CSV.read(params["phase_file"], DataFrame)
    println(first(phases, 5), "\n")

    stations = get_stations(params)
    println(first(stations, 5), "\n")

    model = BSON.load(params["model_file"], @__MODULE__)[:model]
    scaler = data_scaler(params)

    ######
    velmod = Eikonet.initialize_velmod(params, Eikonet.VelMod1D)
    train_loader, test_loader = Eikonet.build_linear_dataset(params, velmod, 8192*4, 1024, 128)
    Eikonet.plot_solution(params, test_loader, model)
    #####

    # Loop over events
    origins = []
    for phase_sub in groupby(phases, :evid)
        println(phase_sub, "\n")
        X_inp, T_obs, T_ref = prepare_event_data(DataFrame(phase_sub), stations)
        @time origin = locate(params, X_inp, T_obs, model, scaler)
        println(origin)
        push!(origins, (phase_sub.evid, origin))
        return
    end
    #origins = DataFrame(origins, )
end


end