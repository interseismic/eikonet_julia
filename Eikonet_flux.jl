module Eikonet
__precompile__
using CSV
using DataFrames
using CUDA
using Flux
using Flux.Data: DataLoader
using Plots
using LinearAlgebra
using JLD2
using Random
using Interpolations: LinearInterpolation, Extrapolation
using Geodesy
using Distributions
using Base.Iterators: flatten
using StatsBase: mean

include("./Input.jl")
export MinmaxScaler, inverse, Origin, sec2date
export forward, data_scaler, get_stations, format_arrivals, generate_syn_dataset
export forward_point, inverse_point, EikoNet, EikoNet1D, solve, build_model

abstract type EikoNet end

struct EikoNet1D <: EikoNet
    model::Flux.Chain
    scale::Float32
    lat_min::Float32
    lat_max::Float32
    lon_min::Float32
    lon_max::Float32
    z_min::Float32
    z_max::Float32
    cartesian_mins::Vector{Float32}
end

struct EikoNet1DSymm <: EikoNet
    encoder::Flux.Chain
    decoder::Flux.Chain
    scale::Float32
end

function τ0(x::AbstractArray)
    if ndims(x) == 2
        return sqrt.(sum((x[4:6,:] - x[1:3,:]).^2, dims=1))
    elseif ndims(x) == 3
        return sqrt.(sum((x[4:6,:,:] - x[1:3,:,:]).^2, dims=1))
    elseif ndims(x) == 1
        return sqrt(sum((x[4:6] - x[1:3])^2))
    end
end

function τ1(eikonet::EikoNet1DSymm, x::Array)
    x_new = eikonet.encoder(x[1:3,:]) #+ eikonet.encoder(x[4:6,:])
    # x_new = cat(z, reshape(x[7,:], 1, size(x,2)), dims=1)
    # x_new = cat(z, x[7:7,:], dims=1)
    return eikonet.decoder(x_new)
end

function ∇τ0(x::AbstractArray)
    return (x[4:6,:] .- x[1:3,:]) ./ τ0(x)
end

function (eikonet::EikoNet1D)(x::AbstractArray)
    return τ0(x) .* eikonet.model(x)
end

# function (eikonet::EikoNet1DSymm)(x::AbstractArray)
#     z = eikonet.encoder(x[1:3,:]) + eikonet.encoder(x[4:6,:])
#     x_new = cat(z, x[7:7,:], dims=1)
#     τ1 = eikonet.decoder(x_new)
#     return eikonet.scale * τ1 .* τ0(x)
# end

function EikonalPDE(x::AbstractArray, eikonet::EikoNet1D)
    τ1 = eikonet.model(x)
    ∇τ1 = gradient(y -> sum(eikonet.model(y)), x)[1][4:6,:] 
    ∇τ = τ1 .* ∇τ0(x) + τ0(x) .* ∇τ1
    s = sqrt.(sum(∇τ.^2, dims=1))
end

function EikonalPDE(x::Array, eikonet::EikoNet1DSymm)
    τ1 = eikonet.decoder(x) + eikonet.decoder(x)
    # τ1 = eikonet.decoder(x[1:3,:]) #+ eikonet.decoder(x[1:3,:])
    ∇τ1 = grad(central_fdm(5, 1), z -> sum(eikonet.decoder(z)), x)[1][4:6,:]
    # ∇τ1 = Zygote.gradient(y -> sum(eikonet.decoder(y)), x)[1][4:6,:]
    ∇τ = τ1 .* ∇τ0(x) + τ0(x) .* ∇τ1
    s = sqrt.(sum(∇τ.^2, dims=1))
end

function build_model()
    return Chain(
        Dense(7, 64, gelu),
        SkipConnection(Dense(64, 64, gelu), +),
        SkipConnection(Dense(64, 64, gelu), +),
        SkipConnection(Dense(64, 64, gelu), +),
        SkipConnection(Dense(64, 64, gelu), +),
        Dense(64, 1, abs),
    )
end

function build_symm_model()
    encoder = Chain(
        Dense(3, 16, elu),
        # Dense(16, 16, elu),
        # Dense(16, 16, elu),
        Dense(16, 7, elu),
    )
    decoder = Chain(
        Dense(7, 16, elu),
        # Dense(16, 16, elu),
        # SkipConnection(Dense(64, 64, elu), +),
        # SkipConnection(Dense(64, 64, elu), +),
        # SkipConnection(Dense(64, 64, elu), +),
        # SkipConnection(Dense(64, 64, elu), +),
        Dense(16, 1, abs),
    )
    return encoder, decoder
end

Flux.@functor EikoNet1D (model,)
Flux.@functor EikoNet1DSymm (encoder, decoder,)

abstract type VelocityModel end

function forward(X::AbstractArray{Float32}, eikonet::EikoNet)
    return forward(X, eikonet.scale)
end

function forward(X::AbstractArray, eikonet::EikoNet)
    return forward(X, eikonet.scale)
end

function inverse(Y::AbstractArray, eikonet::EikoNet)
    return inverse(Y, eikonet.scale)
end

struct VelMod1D <: VelocityModel
    df::DataFrame
    int_p::Extrapolation
    int_s::Extrapolation
end

function build_linear_dataset(params, velmod::VelMod1D, n_train::Int, n_test::Int, batch_size::Tuple, device)
    n_tot = n_train + n_test
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)

    src_lat = rand(Uniform(params["lat_min"], params["lat_max"]), n_tot)
    src_lon = rand(Uniform(params["lon_min"], params["lon_max"]), n_tot)
    src_depth = rand(Uniform(params["z_min"], params["z_max"]), n_tot) .* 1f3
    sources = trans.([LLA(lat=src_lat[i], lon=src_lon[i]) for i in 1:n_tot])

    rec_lat = rand(Uniform(params["lat_min"], params["lat_max"]), n_tot)
    rec_lon = rand(Uniform(params["lon_min"], params["lon_max"]), n_tot)
    rec_depth = rand(Uniform(params["z_min"], params["z_max"]), n_tot) .* 1f3
    recs = trans.([LLA(lat=rec_lat[i], lon=rec_lon[i]) for i in 1:n_tot])    

    v = zeros(Float32, n_tot)
    x = zeros(Float32, 7, n_tot)
    phase_labels = collect(1:n_tot) .% 2
 
    for i in 1:n_tot
        x[1,i] = sources[i].e
        x[2,i] = sources[i].n
        x[3,i] = src_depth[i]

        x[4,i] = recs[i].e
        x[5,i] = recs[i].n
        x[6,i] = rec_depth[i]
        x[7,i] = phase_labels[i]

        if phase_labels[i] == 0
            v[i] = 1f0/velmod.int_p(rec_depth[i]/Float32(1000))
        elseif phase_labels[i] == 1
            v[i] = 1f0/velmod.int_s(rec_depth[i]/Float32(1000))
        else
            println(phase_labels[i])
            println("Error phase label not binary")
            return
        end
    end

    x[1:6,:] ./= 1f3
    scaler = data_scaler(params)
    x = Array{Float32}(forward(x, scaler))

    x_train = x[:,1:n_train] |> device
    x_test = x[:,end-n_test+1:end] |> device
    y_train = reshape(v[1:n_train], 1, :) |> device
    y_test = reshape(v[end-n_test+1:end], 1, :) |> device

    train_data = DataLoader((x_train, y_train), batchsize=batch_size[1], shuffle=true)
    test_data = DataLoader((x_test, y_test), batchsize=batch_size[2], shuffle=true)
    
    return train_data, test_data
end

function EikonalLoss(x::AbstractArray, s::AbstractArray, eikonet::EikoNet)
    ŝ = EikonalPDE(x, eikonet)
    return Flux.mse(ŝ, s)
end

function plot_solution(params, test_loader, eikonet::EikoNet)
    x_test, s = test_loader.data
    scaler = data_scaler(params)
    ŝ = EikonalPDE(x_test, eikonet)
    x_test = inverse(x_test, scaler)
    v̂ = 1f0 ./ ŝ
    v = 1f0 ./ s
    if length(v) > 1000
        nmax = 1000
    else
        nmax = length(v)
    end
    # x_cart = inverse(x_test, scaler)
    scatter(x_test[6,1:nmax], v̂[1,1:nmax], label="v̂", left_margin = 20Plots.mm)
    scatter!(x_test[6,1:nmax], v[1,1:nmax], label="v", left_margin = 20Plots.mm)
    ylims!((0f0, 10.0))
    savefig("test_v.pdf")

    # x_test = zeros(Float32, 7, 100)
    # x_test[3,:] .= Float32(15.0)
    # x_test[4,:] = collect(range(0f0, 200f0, length=100))
    # x_test = forward(x_test, scaler)
    # T̂ = eikonet(x_test)
    # x_test = inverse(x_test, scaler)
    # scatter(x_test[4,:], T̂[1,:], label="T̂", left_margin = 20Plots.mm)
    # savefig("test_t.pdf")
end

function solve(x::AbstractArray, model::Chain, scaler::MinmaxScaler)
    if ndims(x) == 2
        x_const = forward(x, scaler)
        τ0 = sqrt.(sum((x[4:6,:] - x[1:3,:]).^2, dims=1))
    elseif ndims(x) == 3
        x_const = forward(x, scaler)
        τ0 = sqrt.(sum((x[4:6,:,:] - x[1:3,:,:]).^2, dims=1))
    end
    return τ0 .* model(x_const)
end

function solve(x::AbstractArray, eikonet::EikoNet)
    return solve(x, eikonet.model, eikonet.scaler)
end

function initialize_velmod(params, ::Type{VelMod1D})
    df = CSV.read(params["velmod_file"], DataFrame)
    int_p = LinearInterpolation(df.depth, df.vp)
    int_s = LinearInterpolation(df.depth, df.vs)
    return VelMod1D(df, int_p, int_s)
end

function train(pfile; kws...)
    params = JSON.parsefile(pfile)

    device = cpu
    velmod = initialize_velmod(params, VelMod1D)

    scaler = data_scaler(params)

    eikonet = EikoNet1D(build_model(), scaler.scale,
                        params["lat_min"], params["lat_max"],
                        params["lon_min"], params["lon_max"],
                        params["z_min"], params["z_max"],
                        scaler.min) |> device
    # eikonet = EikoNet1DSymm(build_symm_model()[1], build_symm_model()[2], scaler.scale)
    opt_state = Flux.setup(ADAM(params["lr"]), eikonet)

    println("Compiling model...")
    n_train = params["n_train"]
    n_test = params["n_test"]
    dummy_train, dummy_test = build_linear_dataset(params, velmod, 2, 2, (2, 2), device)

    @time for data in dummy_train
        x, s = data
        println(EikonalLoss(x, s, eikonet))
        grad = Flux.gradient(m -> EikonalLoss(x, s, m), eikonet)
    end
    println("Finished compiling.")

    return
    println("Begin training Eikonet")
    loss_best = Inf
    for i in 1:params["n_epochs"]
        train_loss = 0f0
        test_loss = 0f0
        train_loader, test_loader = build_linear_dataset(params, velmod, n_train, n_test,
                                                        (params["batch_size"], 1024), device)
        for data in train_loader
            x, s = data
            val, grads = Flux.withgradient(eikonet) do m
                EikonalLoss(x, s, m)
            end
            Flux.update!(opt_state, eikonet, grads[1])
            train_loss += val * length(s)
        end
        train_loss /= n_train
        for data in test_loader
            x, s = data
            val = EikonalLoss(x, s, eikonet)
            test_loss += val * length(s)
        end
        test_loss /= n_test
        if test_loss < loss_best
            model_state = Flux.state(eikonet)
            jldsave(params["model_file"]; model_state)
            loss_best = test_loss
        end
        println("Epoch $i train $train_loss test $test_loss best $loss_best")
        plot_solution(params, test_loader, eikonet)
    end

    plot_solution(params, test_loader, eikonet)

end

function plot_results(pfile)
    params = JSON.parsefile(pfile)
    eikonet = BSON.load(params["model_file"], @__MODULE__)[:eikonet]
    velmod = initialize_velmod(params, VelMod1D)
    train_loader, test_loader = build_linear_dataset(params, velmod, 2000, 2000, (64, 64), cpu)
    scaler = data_scaler(params)
    plot_solution(params, test_loader, eikonet)
end

end
