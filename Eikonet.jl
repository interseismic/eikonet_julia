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
    γ = cat(cos.(Float32(2π) .* x[1:6,:]), sin.(Float32(2π) .* x[1:6,:]), x[7:7,:], dims=1)
    return γ, st
end

function (l::CylindricalSymmetry)(x::AbstractArray, ps, st::NamedTuple)
    radial_offset = sqrt.((x[1:1,:]-x[4:4,:]).^2 + (x[2:2,:]-x[5:5,:]).^2)
    x_new = cat(radial_offset, x[3:3,:], x[6:7,:], dims=1)
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

    x_src = rand(Float32, 3, n_tot)
    x_src[3,:] = rand(Uniform(params["z_min"], params["z_max"]), n_tot) ./ Float32(params["scale"])
    
    min_z_scaled = Float32(params["z_min"] / params["scale"])
    max_z_scaled = Float32(params["z_max"] / params["scale"])
    
    ## random distance to prevent source-receiver distance bias in training data
    x_rec = zeros(Float32, 3, n_tot)
    points_outside = 1:n_tot
    max_dist = Float32(sqrt(3)) ## space diagonal of unit cube
    while length(points_outside) > 0
        vect = rand(Float32, 3, n_tot) .- 5f-1 ## random vectors - source-receiver distances
        vect = vect ./ sqrt.(sum(vect.^2, dims=1)) ## normalize vectors
        dist = rand(Float32, n_tot) .* max_dist ## random distances from 0 to max_dist
        clamp!(dist, 1f-5, max_dist) ## need to clamp or you end up with singularity and NaN (2 hours of debugging - NG&T)
        x_rec_random = (dist' .* vect) .+ x_src ## receiver positions
    
        x_rec[:, points_outside] = x_rec_random[:, points_outside]
    
        outside_x = findall((x_rec[1,:] .< 0f0) .| (x_rec[1,:] .> 1f0))
        outside_y = findall((x_rec[2,:] .< 0f0) .| (x_rec[2,:] .> 1f0))
        outside_z = findall((x_rec[3,:] .< min_z_scaled) .| (x_rec[3,:] .> max_z_scaled))
        
        points_outside = unique(cat(outside_x, outside_y, outside_z, dims=1))
    end

    x_pha = collect(1:n_tot) .% 2 ;
    
    x = cat(x_src, x_rec, x_pha', dims=1)

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

function EikonalLoss(x::AbstractArray, s::AbstractArray, EikoNet::L, ps::NamedTuple, st::NamedTuple; reduce=true) where {L}
    ŝ = EikonalPDE(x, EikoNet, ps, st)
    return Float32(0.95) * sum(abs.(ŝ - s)) / sum(abs.(s)) + Float32(0.05) * sqrt(sum((ŝ - s).^2) / sum(s.^2))
    # return mean(abs2, ŝ .- s)
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
        CylindricalSymmetry(6+7, 3+1),
        Dense(4, 16, Lux.elu),
    	SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
    	SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
    	SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
        SkipConnection(Chain(Dense(16, 32, Lux.elu), Dense(32, 16, Lux.elu)), +),
    	Dense(16, 1, Lux.relu))

    rng = MersenneTwister()
    Random.seed!(rng, 12345)
    model = EikoNet(τ, params["scale"])
    ps, st = Lux.setup(rng, model)

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

        train_loader, test_loader = build_1d_velmod(params, velmod, n_train, n_test, (params["batch_size"], 1024), dev)

        # Train the model
        train_loss = []
        for (x, s) in train_loader
            loss, ∇_loss = withgradient(p -> EikonalLoss(x, s, model, p, st), ps)
            opt_state, ps = Optimisers.update(opt_state, ps, ∇_loss[1])
            push!(train_loss, loss)
        end
        train_loss = mean(train_loss)

        # Validate the model
        test_loss = []
        for (x, s) in test_loader
            loss = EikonalLoss(x, s, model, ps, st)
            push!(test_loss, loss)
        end
        test_loss = mean(test_loss)

        if test_loss < loss_best
            @save params["model_file"] {compress = true} ps st model
            loss_best = test_loss
            plot_solution(params, test_loader, model, ps, st)
        end
        println("Epoch $epoch train $train_loss test $test_loss best $loss_best")

        # if (epoch % 30) == 0
        #     η /= 1f1
        #     Optimisers.adjust!(opt_state, η)
        # end
    end

end

end
