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
    params["n_epochs"] = 500
    params["lr"] = 1e-1
    params["verbose"] = false
    return params
end

# function fwd_transform(point::Geodesy.LLA, origin::Geodesy.LLA, scaler::UnitRangeTransform)
#     enu = ENU(point, origin, wgs84) ./ 1f3
#     return transform(scaler, [enu.e, enu.n, enu.u])
# end

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
    return hcat(stations, DataFrame(X=X, Y=Y, Z=Z))
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
    #scaler = fit(UnitRangeTransform, x, dims=2)
    return scaler
end