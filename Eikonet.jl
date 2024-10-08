module Eikonet

using Lux, LuxCUDA, LuxDeviceUtils, Random, MLUtils, Optimisers, Zygote
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

export EikoNet, get_stations, sec2date, CylindricalSymmetry

abstract type VelocityModel end

struct VelMod1D <: VelocityModel
    df::DataFrame
    int_p::Extrapolation
    int_s::Extrapolation
end

struct EikoNet{C, T} <: Lux.AbstractExplicitContainerLayer{(:τ1,)}
    τ1::C
    scale::T
end

struct CylindricalSymmetry <: Lux.AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
end

struct PosEncoding <: Lux.AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
end

Lux.initialparameters(::AbstractRNG, ::PosEncoding) = return ()
Lux.initialstates(::AbstractRNG, ::PosEncoding) = NamedTuple()

Lux.initialparameters(::AbstractRNG, ::CylindricalSymmetry) = return ()
Lux.initialstates(::AbstractRNG, ::CylindricalSymmetry) = NamedTuple()

function (l::PosEncoding)(x::AbstractArray{Float32}, ps, st::NamedTuple)
    γ = cat(cos.(2π .* x[1:6,:]), sin.(2π .* x[1:6,:]), x[7:7,:], dims=1)
    return γ, st
end

function (l::CylindricalSymmetry)(x::AbstractArray{T, 2}, ps, st::NamedTuple) where {T}
    radial_offset = sqrt.((x[1:1,:]-x[4:4,:]).^2 + (x[2:2,:]-x[5:5,:]).^2)
    x_new = cat(radial_offset, x[3:3,:], x[6:7,:], dims=1)
    return x_new, st
end

function (l::CylindricalSymmetry)(x::AbstractArray{T, 3}, ps, st::NamedTuple) where {T}
    radial_offset = sqrt.((x[1:1,:,:]-x[4:4,:,:]).^2 + (x[2:2,:,:]-x[5:5,:,:]).^2)
    x_new = cat(radial_offset, x[3:3,:,:], x[6:7,:,:], dims=1)
    return x_new, st
end

function (e::EikoNet)(x::AbstractArray{T}, ps::NamedTuple, st::NamedTuple) where {T}
    return τ0(x) .* e.τ1(x, ps, st)[1]
end

function plot_solution(params, test_loader, EikoNet, ps::NamedTuple, st::NamedTuple)
    x_test, s = test_loader.data
    ŝ = EikonalPDE(x_test, EikoNet, ps, st)
    v̂ = 1f0 ./ ŝ
    v = 1f0 ./ s
    resid = vec(v - v̂)
    idx = sortperm(resid, rev=true)
    if length(v) > 1000
        nmax = 1000
    else
        nmax = length(v)
    end
    # x_cart = inverse(x_test, scaler)
    scatter(x_test[6,idx[1:nmax]] * params["scale"], v̂[1,idx[1:nmax]], label="v̂", left_margin = 20Plots.mm, markersize = 2, markerstrokewidth=0)
    scatter!(x_test[6,1:nmax] * params["scale"], v[1,1:nmax], label="v", left_margin = 20Plots.mm, markersize = 2, markerstrokewidth=0)
    ylims!((0f0, 10.0))
    savefig("test_v.pdf")

    XX = x_test[6,idx] * params["scale"]

    idx = findall(x_test[7,:] .<= 0.5)
    YY = abs.(v[1,idx] - v̂[1,idx]) ./ v[1,idx]
    # scatter(XX, YY, color = :red, left_margin = 20Plots.mm, markersize = 2)
    histogram(log10.(YY), bins=100)
    savefig("test_error_p.png")

    idx = findall(x_test[7,:] .> 0.5)
    YY = abs.(v[1,idx] - v̂[1,idx]) ./ v[1,idx]
    # scatter(XX, YY, color = :blue, left_margin = 20Plots.mm, markersize = 2)
    histogram(log10.(YY), bins=100)
    
    # ylims!((0f0, 10.0))
    savefig("test_error_s.png")

    histogram()
end

function build_1d_velmod(params, velmod::VelMod1D, n_train::Int, n_test::Int, batch_size::Tuple, device)
    n_tot = n_train + n_test
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)

    x = rand(Float32, 7, n_tot)
    z_min = Float32(params["z_min"]) / Float32(params["scale"])
    z_max = Float32(params["z_max"]) / Float32(params["scale"])
    vz_max = Float32(maximum(velmod.df.depth)) / Float32(params["scale"])
    x[3,:] = rand(Uniform(z_min, vz_max), n_tot)
    x[6,:] = rand(Uniform(z_min, z_max), n_tot)
    x[7,:] = collect(1:n_tot) .% 2

    s = zeros(Float32, n_tot)
    for i in 1:n_tot
        ϵ = 1f-2
        effective_depth = clamp(x[6,i] * Float32(params["scale"]), Float32(minimum(velmod.df.depth))+ϵ, Float32(maximum(velmod.df.depth))-ϵ)
        if x[7,i] <= 5f-1
            s[i] = 1f0 ./ velmod.int_p(effective_depth)
        else
            s[i] = 1f0 ./ velmod.int_s(effective_depth)
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


function build_1d_velmod(params, eikonet, ps::NamedTuple, st::NamedTuple, velmod::VelMod1D, n_tot0::Int, fraction::Float32, batch_size::Tuple, device)
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)
    n_tot = Int(floor(n_tot0 / fraction))

    x = rand(Float32, 7, n_tot)
    z_min = Float32(params["z_min"]) / Float32(params["scale"])
    z_max = Float32(params["z_max"]) / Float32(params["scale"])
    vz_max = Float32(maximum(velmod.df.depth)) / Float32(params["scale"])
    x[3,:] = rand(Uniform(z_min, vz_max), n_tot)
    x[6,:] = rand(Uniform(z_min, z_max), n_tot)
    x[7,:] = collect(1:n_tot) .% 2

    y = zeros(Float32, n_tot)
    for i in 1:n_tot
        ϵ = 1f-2
        effective_depth = clamp(x[6,i] * Float32(params["scale"]), Float32(minimum(velmod.df.depth))+ϵ, Float32(maximum(velmod.df.depth))-ϵ)
        if x[7,i] <= 5f-1
            y[i] = 1f0 ./ velmod.int_p(effective_depth)
        else
            y[i] = 1f0 ./ velmod.int_s(effective_depth)
        end
    end

    x = x |> device
    y = reshape(y, 1, :) |> device

    loader = DataLoader((x, y), batchsize=batch_size[1], shuffle=true)
    
    residuals = []
    for (x, s) in loader
        ŝ = EikonalPDE(x, eikonet, ps, st)
        resid = abs.(ŝ - s)
        append!(residuals, resid)
    end
    idx = sortperm(residuals, rev=true)
    idx = idx[1:n_tot0]
    x = x[:,idx]
    y = y[:,idx]
    loader = DataLoader((x, y), batchsize=batch_size[2], shuffle=true)
    return loader

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
    return mean(abs2, (s - ŝ) ./ s)
    # return mean(abs.(s - ŝ).^3)
end

function RelativeEikonalLoss(x::AbstractArray, s::AbstractArray, EikoNet::L, ps::NamedTuple, st::NamedTuple) where {L}
    ŝ = EikonalPDE(x, EikoNet, ps, st)
    return mean(abs, (ŝ .- s) ./ s)
end

function initialize_velmod(params, ::Type{VelMod1D})
    df = CSV.read(params["velmod_file"], DataFrame)
    int_p = LinearInterpolation(df.depth, df.vp)
    int_s = LinearInterpolation(df.depth, df.vs)
    return VelMod1D(df, int_p, int_s)
end


function init_eikonet(params)

    τ = Lux.Chain(
        CylindricalSymmetry(6+7, 3+1),
        Dense(4, 16, Lux.elu),
    	SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
    	SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
    	SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
    	Dense(16, 1, Lux.relu))

    return EikoNet(τ, Float32(params["scale"]))
end

function train(pfile; kws...)
    params = JSON.parsefile(pfile)

    # dev = gpu_device()
    dev = cpu_device()
    velmod = initialize_velmod(params, VelMod1D)

    n_train = params["n_train"]
    n_test = params["n_test"]

    # τ = Lux.Chain(
    #     CylindricalSymmetry(6+7, 3+1),
    #     Dense(3+1, 64, Lux.elu),
    #     Dense(64, 64, Lux.elu),
    #     Dense(64, 64, Lux.elu),
    #     Dense(64, 1, Lux.relu))

    τ = init_eikonet(params)

    rng = MersenneTwister()
    Random.seed!(rng, 12345)
    model = EikoNet(τ, params["scale"])
    ps, st = Lux.setup(rng, model)

    println("Compiling model...")
    println(extrema(model(rand(Float32, 6+1, 1000), ps, st)))
    println(extrema(EikonalPDE(rand(Float32, 6+1, 1000),model,  ps, st)))

    ps = ps |> dev
    st = st |> dev

    η = Float32(params["lr"])
    opt = Optimisers.ADAM(η)
    opt_state = Optimisers.setup(opt, ps)

    loss_best = Inf

    println("Begin training EikoNet")

    for epoch in 1:params["n_epochs"]
        train_loss = 0f0
        test_loss = 0f0

        if epoch < 15
            train_loader, test_loader = build_1d_velmod(params, velmod, n_train, n_test, (params["batch_size"], 1024), dev)
        else
            train_loader = build_1d_velmod(params, model, ps, st, velmod, n_train, Float32(0.1), (1024, params["batch_size"]), dev)
            test_loader = build_1d_velmod(params, model, ps, st, velmod, n_test, Float32(1.0), (1024, 1024), dev)
        end

        # Train the model
        for (x, s) in train_loader
            loss, ∇_loss = withgradient(p -> EikonalLoss(x, s, model, p, st), ps)
            opt_state, ps = Optimisers.update(opt_state, ps, ∇_loss[1])
            train_loss += loss * length(s)
        end
        train_loss /= n_train

        # Validate the model
        test_loss = 0f0
        for (x, s) in test_loader
            test_loss += EikonalLoss(x, s, model, ps, st) * length(s)
        end
        test_loss /= n_test

        if test_loss < loss_best
            @save params["model_file"] {compress = true} ps st model
            loss_best = test_loss
            plot_solution(params, test_loader, model, ps, st)
        end
        println("Epoch $epoch train $train_loss test $test_loss best $loss_best")

        if (epoch % 30) == 0
            η /= 1f1
            Optimisers.adjust!(opt_state, η)
        end
    end

end

end