module Eikonet

using Lux, LuxCUDA, Random, MLUtils, Optimisers, Zygote
using DataFrames
using Interpolations: LinearInterpolation, Extrapolation
using Plots
using LinearAlgebra
using JLD2
using Random
using Geodesy
using CSV
using JSON
using Distributions
using Base.Iterators: flatten
using StatsBase: mean
using BenchmarkTools

include("./Input.jl")

export EikoNet, get_stations, sec2date

abstract type VelocityModel end

struct VelMod1D <: VelocityModel
    df::DataFrame
    int_p::Extrapolation
    int_s::Extrapolation
end

struct EikoNet{C, T} <: Lux.AbstractExplicitContainerLayer{(:τ1,)}
    τ1::C
    scale::T
    # ω::AbstractArray{T}
end

function (e::EikoNet)(x::AbstractArray{T}, ps::NamedTuple, st::NamedTuple) where {T}
    return τ0(x) .* e.τ1(x, ps, st)[1]
end

# Commented out for now. Will need this when the Sine encoding layer is added
# function EikoNet(model::Lux.AbstractExplicitLayer, scale::T, n_freqs::Int) where {T}
#     x = range(0f0, 1f0, length=2*n)
#     dx = x[2] - x[1]
#     f_nyquist = 5f-1 * 1f0 / dx
#     ω = 2f0 .* Float32(π) .* collect(range(0f0, f_nyquist, length=n))
#     return EikoNet(model, scale)
# end

function plot_solution(params, test_loader, EikoNet, ps::NamedTuple, st::NamedTuple)
    x_test, s = test_loader.data
    ŝ = EikonalPDE(x_test, EikoNet, ps, st)
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
end

function build_1d_velmod(params, velmod::VelMod1D, n_train::Int, n_test::Int, batch_size::Tuple, device)
    n_tot = n_train + n_test
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)

    x = rand(Float32, 7, n_tot)
    x[3,:] = rand(Uniform(params["z_min"], params["z_max"]), n_tot) ./ Float32(params["scale"])
    x[6,:] = rand(Uniform(params["z_min"], params["z_max"]), n_tot) ./ Float32(params["scale"])
    x[7,:] = collect(1:n_tot) .% 2

    s = zeros(Float32, n_tot)
    for i in 1:n_tot
        if x[7,i] <= 5f-1
            s[i] = 1f0 ./ velmod.int_p(x[6,i] * Float32(params["scale"]))
        else
            s[i] = 1f0 ./ velmod.int_s(x[6,i] * Float32(params["scale"]))
        end
    end

    x_train = x[:,1:n_train] |> device
    x_test = x[:,end-n_test+1:end] |> device
    y_train = reshape(s[1:n_train], 1, :) |> device
    y_test = reshape(s[end-n_test+1:end], 1, :) |> device

    train_data = DataLoader((x_train, y_train), batchsize=batch_size[1], shuffle=true)
    test_data = DataLoader((x_test, y_test), batchsize=batch_size[2], shuffle=true)
    
    return train_data, test_data
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

function ∇τ0(x::AbstractArray)
    return (x[4:6,:] .- x[1:3,:]) ./ τ0(x)
end

function finite_diff(x::AbstractArray{Float32}, ϵ::Float32, EikoNet::Lux.AbstractExplicitContainerLayer, ps::NamedTuple, st::NamedTuple)
    dx = Array{Float32}([0 0 0; 0 0 0; 0 0 0; 1 0 0; 0 1 0; 0 0 1; 0 0 0]) * 0.5f0 * ϵ
    f(x) = Lux.apply(EikoNet, x, ps, st)[1]
    Δf(x) = (f(x .+ dx) - f(x .- dx)) / ϵ
    ∇f = Δf.(collect(eachcol(x)))
    return vcat(∇f...)'
end

function EikonalPDE(x::AbstractArray, model::EikoNet, ps::NamedTuple, st::NamedTuple)
    τ1 = model.τ1(x, ps, st)[1]
    ∇τ1 = finite_diff(x, 1f-3, model.τ1, ps, st)
    ∇τ = (τ1 .* ∇τ0(x) + τ0(x) .* ∇τ1) ./ model.scale
    ŝ = sqrt.(sum(∇τ.^2, dims=1))
    return ŝ
end

function EikonalLoss(x::AbstractArray, s::AbstractArray, EikoNet::L, ps::NamedTuple, st::NamedTuple) where {L}
    ŝ = EikonalPDE(x, EikoNet, ps, st)
    return sum(((s-ŝ)).^2)
end

function initialize_velmod(params, ::Type{VelMod1D})
    df = CSV.read(params["velmod_file"], DataFrame)
    int_p = LinearInterpolation(df.depth, df.vp)
    int_s = LinearInterpolation(df.depth, df.vs)
    return VelMod1D(df, int_p, int_s)
end

function train(pfile; kws...)
    params = JSON.parsefile(pfile)

    # dev = gpu_device()
    dev = cpu_device()
    velmod = initialize_velmod(params, VelMod1D)

    println("Compiling model...")
    n_train = params["n_train"]
    n_test = params["n_test"]

    τ = Lux.Chain(
        Dense(7, 16, Lux.elu),
        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
        Dense(16, 1))

    rng = MersenneTwister()
    Random.seed!(rng, 12345)
    model = EikoNet(τ, params["scale"])
    ps, st = Lux.setup(rng, model)

    println(extrema(model(rand(Float32, 7, 1000), ps, st)))
    println(extrema(EikonalPDE(rand(Float32, 7, 1000),model,  ps, st)))

    ps = ps |> dev
    st = st |> dev

    opt = Optimisers.ADAM(1f-3)
    opt_state = Optimisers.setup(opt, ps)

    loss_best = Inf

    println("Begin training EikoNet")
    for epoch in 1:params["n_epochs"]
        train_loss = 0f0
        test_loss = 0f0
        train_loader, test_loader = build_1d_velmod(params, velmod, n_train, n_test, (params["batch_size"], 1024), dev)

        # Train the model
        for (x, s) in train_loader
            loss, ∇_loss = withgradient(p -> EikonalLoss(x, s, model, p, st), ps)
            opt_state, ps = Optimisers.update(opt_state, ps, ∇_loss[1])
            train_loss += loss
        end
        train_loss /= n_train

        # Validate the model
        test_loss = 0f0
        for (x, s) in test_loader
            test_loss += EikonalLoss(x, s, model, ps, st)
        end
        test_loss /= n_test

        if test_loss < loss_best
            @save params["model_file"] {compress = true} ps st model
            loss_best = test_loss
        end
        println("Epoch $epoch train $train_loss test $test_loss best $loss_best")
        plot_solution(params, test_loader, model, ps, st)
    end

end

end