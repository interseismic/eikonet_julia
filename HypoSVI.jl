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
    time::DateTime
end

function sec2date(s::AbstractFloat)
    sec_sign = Int32(sign(s))
    s = abs(s)
    sec = Int32(floor(s))
    msec = Int32(floor(1000*(s - sec)))
    return Dates.Second(sec * sec_sign) + Dates.Millisecond(msec * sec_sign)
end

function locate(params, X_inp, T_obs, eikonet, scaler, T_ref)
    # Main function to locate events
    lat0 = 0.5(params["lat_min"] + params["lat_max"])
    lon0 = 0.5(params["lon_min"] + params["lon_max"])
    z0 = params["z_min"]
    
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)

    point_enu = trans(LLA(lat=lat0, lon=lon0))
    X_src = [point_enu.e, point_enu.n, z0*1e3, 0.0]

    grad_scale = [scaler.scale, scaler.scale, scaler.scale*1e3, 1.0]

    for i in 1:params["n_epochs"]
        function loss(X_src)
            X = cat(repeat(X_src[1:3], 1, size(X_inp, 1))', X_inp, dims=2)'
            T_src = X_src[4]
            X = forward(X, scaler)
            Flux.mae(Eikonet.solve(X, eikonet, scaler) .+ T_src, T_obs)
        end
        grads = gradient(loss, X_src)[1]
        X_src .-= params["lr"] * grad_scale .* grads 
    end
    T_src = X_src[4]
    X_src = reshape(X_src[1:3], :, 1)
    inv_trans = LLAfromENU(origin, wgs84)
    hypo_lla = inv_trans(ENU(X_src[1], X_src[2], 0f0))
    return Origin(hypo_lla.lat, hypo_lla.lon, X_src[3]/1f3, T_ref + sec2date(T_src))
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
    params = build_hyposvi_params()

    # Read in a pick file in csv format (e.g. from Gamma or PhaseLink)
    phases = CSV.read(params["phase_file"], DataFrame)
    println(first(phases, 5), "\n")

    stations = get_stations(params)
    println(first(stations, 5), "\n")

    model = BSON.load(params["model_file"], @__MODULE__)[:model]
    scaler = data_scaler(params)

    # Loop over events
    origins = DataFrame(time=DateTime[], evid=Int[], lat=Float32[], lon=Float32[], depth=Float32[])
    for phase_sub in groupby(phases, :evid)
        if params["verbose"]
            println(phase_sub, "\n")
        end
        X_inp, T_obs, T_ref = prepare_event_data(DataFrame(phase_sub), stations)
        origin = locate(params, X_inp, T_obs, model, scaler, T_ref)

        push!(origins, (origin.time, phase_sub.evid[1], origin.lat, origin.lon, origin.depth))
        println(last(origins, 1), "\n")
    end

    println(first(origins, 100))
    CSV.write(params["catalog_outfile"], origins)
end


end