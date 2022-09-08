using Dates
using JSON

function convert_gamma_to_hyposvi(fname, outfile)
    picks = CSV.read(fname, DataFrame)
    nets = []
    stas = []
    chans = []
    evids = []
    for row in eachrow(picks)
        push!(evids, row.event_index)
        net, sta, loc, chan = split(row.station_id, ".")
        push!(nets, net)
        push!(stas, sta)
    end
    phases = DataFrame(evid=evids, network=nets, station=stas, phase=picks[:,:phase_type], time=picks[:,:phase_time])
    unique!(phases)
    filter!(row -> row.evid >= 0.0, phases)
    unique!(phases, [:evid, :network, :station, :phase])
    CSV.write(outfile, phases)
end

function read_syn_dataset(params)
    sta_dict = JSON.parsefile("synthetic_stations.json")
    stations = DataFrame(network=String[], station=String[], latitude=Float64[], longitude=Float64[], elevation=Float64[])
    for key in keys(sta_dict)
        net, sta, loc, chan = split(key, ".")
        elev = sta_dict[key]["elevation(m)"]
        lat = sta_dict[key]["latitude"]/1000.
        lon = sta_dict[key]["longitude"]
        push!(stations, (net, sta, lat, lon, elev))
    end
    events = CSV.read("synthetic_events.csv", DataFrame)
    picks = CSV.read("synthetic_picks.csv", DataFrame)
    nets = []
    stas = []
    chans = []
    for row in eachrow(picks)
        net, sta, loc, chan = split(row.station_id, ".")
        push!(nets, net)
        push!(stas, sta)
    end
    phases = DataFrame(network=nets, station=stas, phase=picks[:,:phase_type], time=picks[:,:phase_time])
    
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)
    X = Vector{Float32}()
    Y = Vector{Float32}()
    Z = Vector{Float32}()
    for row in eachrow(stations)
        xyz = trans(LLA(lat=row.latitude, lon=row.longitude))
        push!(X, xyz.e)
        push!(Y, xyz.n)
        push!(Z, row.elevation*1000.0)
    end
    stations= hcat(stations, DataFrame(X=X, Y=Y, Z=Z))


    return phases, stations, events
end

function sec2date(s::AbstractFloat)
    sec_sign = Int32(sign(s))
    s = abs(s)
    sec = Int32(floor(s))
    msec = Int32(floor(1000*(s - sec)))
    return Dates.Second(sec * sec_sign) + Dates.Millisecond(msec * sec_sign)
end

struct Origin
    lat::Float32
    lon::Float32
    depth::Float32
    time::Dates.DateTime
    unc_z::Float32
    X::Float32
    Y::Float32
end

function build_eikonet_params()
    params = Dict()
    params["phase_file"] = "/scratch/zross/oak_ridge/scsn_oak_ridge.csv"
    params["station_file"] = "/scratch/zross/oak_ridge/scsn_stations.csv"
    params["velmod_file"] = "/scratch/zross/oak_ridge/vz_socal.csv"
    params["lon_min"] = -119.8640
    params["lon_max"] = -117.8640
    params["lat_min"] = 33.3580
    params["lat_max"] = 35.3580
    params["z_min"] = 0.0
    params["z_max"] = 60.0
    params["model_file"] = "/scratch/zross/oak_ridge/model.bson"
    params["n_epochs"] = 200
    params["lr"] = 1e-3
    return params
end

function build_eikonet_syn_params()
    params = Dict()
    params["phase_file"] = "/scratch/zross/oak_ridge/scsn_oak_ridge.csv"
    params["station_file"] = "/scratch/zross/oak_ridge/scsn_stations.csv"
    params["velmod_file"] = "/scratch/zross/oak_ridge/vz_socal.csv"
    params["lon_min"] = -119.8640
    params["lon_max"] = -117.8640
    params["lat_min"] = 33.3580
    params["lat_max"] = 35.3580
    params["z_min"] = 0.0
    params["z_max"] = 60.0
    params["model_file"] = "/home/zross/git/eikonet_julia/syntest/model.bson"
    params["n_epochs"] = 200
    params["lr"] = 1e-3
    return params
end

function build_hyposvi_params()
    params = Dict()
    params["phase_file"] = "/scratch/zross/oak_ridge/scsn_oak_ridge.csv"
    params["station_file"] = "/scratch/zross/oak_ridge/scsn_stations.csv"
    params["velmod_file"] = "/scratch/zross/oak_ridge/vz_socal.csv"
    params["catalog_outfile"] = "/scratch/zross/oak_ridge/catalog_svi.csv"
    params["lon_min"] = -119.8640
    params["lon_max"] = -117.8640
    params["lat_min"] = 33.3580
    params["lat_max"] = 35.3580
    params["z_min"] = 0.0
    params["z_max"] = 60.0
    params["model_file"] = "/scratch/zross/oak_ridge/model.bson"
    params["n_epochs"] = 1000
    params["n_particles"] = 1
    params["lr"] = 1e-3
    params["phase_unc"] = 0.20
    params["verbose"] = true
    params["k-NN"] = 500
    params["iter_tol"] = 1f-2
    params["max_k-NN_dist"] = 50
    params["n_ssst_iter"] = 3
    params["inversion_method"] = "SVI"
    return params
end

function build_supergamma_params()
    params = Dict()
    params["phase_file"] = "/scratch/zross/oak_ridge/scsn_oak_ridge.csv"
    params["station_file"] = "/scratch/zross/oak_ridge/scsn_stations.csv"
    params["velmod_file"] = "/scratch/zross/oak_ridge/vz_socal.csv"
    params["catalog_outfile"] = "/scratch/zross/oak_ridge/catalog_svi.csv"
    params["lon_min"] = -119.8640
    params["lon_max"] = -117.8640
    params["lat_min"] = 33.3580
    params["lat_max"] = 35.3580
    params["z_min"] = 0.0
    params["z_max"] = 60.0
    params["model_file"] = "/scratch/zross/oak_ridge/model.bson"
    params["n_epochs"] = 500
    params["n_particles"] = 10
    params["lr"] = 1e-3
    params["phase_unc"] = 0.5
    params["verbose"] = true
    params["k-NN"] = 500
    params["iter_tol"] = 1f-2
    params["max_k-NN_dist"] = 50
    params["n_ssst_iter"] = 1
    params["n_det"] = 4
    params["inversion_method"] = "EM"
    return params
end

function build_supergamma_syn_params()
    params = Dict()
    params["phase_file"] = "/scratch/zross/oak_ridge/scsn_oak_ridge.csv"
    params["station_file"] = "/scratch/zross/oak_ridge/scsn_stations.csv"
    params["velmod_file"] = "/scratch/zross/oak_ridge/vz_socal.csv"
    params["catalog_outfile"] = "/scratch/zross/oak_ridge/catalog_svi.csv"
    params["lon_min"] = -118.004
    params["lon_max"] = -117.004
    params["lat_min"] = 35.205
    params["lat_max"] = 36.205
    params["z_min"] = 0.0
    params["z_max"] = 60.0
    params["model_file"] = "/home/zross/git/eikonet_julia/syntest/model.bson"
    params["n_epochs"] = 100
    params["n_particles"] = 10
    params["lr"] = 1e-2
    params["phase_unc"] = 0.05
    params["verbose"] = true
    params["k-NN"] = 500
    params["iter_tol"] = 0.1
    params["n_det"] = 8
    params["n_warmup_iter"] = 20
    params["inversion_method"] = "EM"
    return params
end

function fwd_transform(point::Geodesy.LLA, origin::Geodesy.LLA)
    enu = ENU(point, origin, wgs84) ./ 1f3
    return enu
end

function fwd_transform!(X::Array, origin::Geodesy.LLA)
    for i in 1:size(X, 2)
        x_tmp = fwd_transform(LLA(lon=X[1], lat=X[1]), origin)
        X[1,i] = x_tmp.e
        X[2,i] = x_tmp.n
        x_tmp = fwd_transform(LLA(lon=X[4], lat=X[4]), origin)
        X[4,i] =  x_tmp.e
        X[5,i] = x_tmp.n
    end
end

struct MinmaxScaler
    min::Float32
    scale::Float32
end

function fit(X::AbstractArray, ::Type{MinmaxScaler})
    mins = minimum(X[1:6,:])
    maxs = maximum(X[1:6,:])
    return MinmaxScaler(mins, maxs-mins)
end

function forward!(X::AbstractArray, scaler::MinmaxScaler)
    X[1:6,:] .= (X[1:6,:] .- scaler.min) ./ scaler.scale
end

function forward(X::AbstractArray, scaler::MinmaxScaler)
    X_new = (X[1:6,:] .- scaler.min) ./ scaler.scale
    return cat(X_new, reshape(X[7,:], 1, :), dims=1)
end

function forward_point(X::AbstractArray, scaler::MinmaxScaler)
    return (X .- scaler.min) ./ scaler.scale
end

function inverse_point(Y::AbstractArray, scaler::MinmaxScaler)
    return (Y .* scaler.scale) .+ scaler.min
end

function inverse!(Y::AbstractArray, scaler::MinmaxScaler)
    Y[1:6,:] .= (Y[1:6,:] .* scaler.scale) .+ scaler.min
end

function inverse(Y::AbstractArray, scaler::MinmaxScaler)
    Y_new = (Y[1:6,:] .* scaler.scale) .+ scaler.min
    return cat(Y_new, reshape(Y[7,:], 1, :), dims=1)
end

function get_stations(params)
    # Note: origin_lla must match what the Eikonet was trained with.
    stations = CSV.read(params["station_file"], DataFrame)
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)
    X = Vector{Float32}()
    Y = Vector{Float32}()
    Z = Vector{Float32}()
    for row in eachrow(stations)
        xyz = trans(LLA(lat=row.latitude, lon=row.longitude))
        push!(X, xyz.e)
        push!(Y, xyz.n)
        push!(Z, row.elevation*1000.0)
    end
    return unique(hcat(stations, DataFrame(X=X, Y=Y, Z=Z)), [:network, :station])
end

function data_scaler(params)
    x = zeros(Float32, 7, 2)
    min_lla = LLA(lat=params["lat_min"], lon=params["lon_min"])
    max_lla = LLA(lat=params["lat_max"], lon=params["lon_max"])
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)
    min_enu = trans(min_lla)
    max_enu = trans(max_lla)
    x[1,1] = min_enu.e
    x[2,1] = min_enu.n
    x[3,1] = params["z_min"] * 1e3
    x[4,1] = min_enu.e
    x[5,1] = min_enu.n
    x[6,1] = params["z_min"] * 1e3
    x[7,1] = 0.0

    x[1,2] = max_enu.e
    x[2,2] = max_enu.n
    x[3,2] = params["z_max"] * 1e3
    x[4,2] = max_enu.e
    x[5,2] = max_enu.n
    x[6,2] = params["z_max"] * 1e3
    x[7,2] = 1.0

    scaler = fit(x, MinmaxScaler)
    return scaler
end

function timedelta(t1::DateTime, t2::DateTime)
    # returns total seconds between t1,t2
    (t1-t2) / Millisecond(1000)
end

function format_arrivals(phases::DataFrame, stations::DataFrame)
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
    return X_inp, T_obs, T_ref, phase_sta
end