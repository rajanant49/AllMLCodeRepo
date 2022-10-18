
include("models.jl")

module Altmin

import Flux
using Statistics: mean
using Flux: gpu
import Main.Models
using CUDA

function compute_code_loss(codes, nmod, lin, loss_fn, codes_target, μ, λ_c)
    output = lin(nmod(codes))
    loss = (1 / μ) * loss_fn(output) + Flux.mse(codes, codes_target)
    if λ_c > 0.0
        loss += (λ_c / μ) * mean(abs.(codes))
    end
    return loss
end

function mark_code_mods!(model)
    module_types = [Models.ConvMod, Models.DenseMod, Models.BatchNormMod, Models.LinMod]
    for mod in Iterators.flatten((model.features_vec, [Models.Flatten()], model.classifier_vec))
        if any(t -> mod isa t, module_types)
            mod.has_codes = true
        end
    end
end

struct AltMinModel{C}
    model_mods::Vector{Any}
    model::C
    n_inputs::Union{Int32, Nothing}
end

Flux.@functor AltMinModel
(m::AltMinModel)(x) = m.model(x)

function get_mods(model, optimizer, scheduler)
    mark_code_mods!(model)

    model_mods::Vector{Any} = []
    nmod::Vector{Any} = []
    lmod::Vector{Any} = []

    for m in Iterators.flatten((model.features_vec, [Models.Flatten()], model.classifier_vec))
        if m.has_codes !== nothing && m.has_codes
            nmod = insert_mod!(model_mods, nmod, false)
            push!(lmod, m |> gpu)
        else
            lmod = insert_mod!(model_mods, lmod, true)
            push!(nmod, m |> gpu)
        end
    end

    insert_mod!(model_mods, nmod, false)
    insert_mod!(model_mods, lmod, true)

    id_codes = [i for (i, m) in Iterators.enumerate(model_mods) 
                  if m.has_codes !== nothing && m.has_codes]

    model_tmp = model_mods[1:id_codes[end - 1]]
    push!(model_tmp, Models.Chain(model_mods[id_codes[end - 1]+1:end]))
    model_tmp[end].has_codes = false
    model_mods = model_tmp

    for m in model_mods
        if m.has_codes !== nothing && m.has_codes
            m.optimizer = optimizer
            m.scheduler = scheduler
        end
    end
    model_mods[end].optimizer = optimizer
    model_mods[end].scheduler = scheduler

    # Data parallel?

    if :n_inputs ∈ fieldnames(typeof(model))
        n_inputs = (model.n_inputs)::Union{Int32, Nothing}
    else
        n_inputs = nothing::Union{Int32, Nothing}
    end
    return AltMinModel(model_mods, Flux.Chain(model_mods...) |> gpu, n_inputs)
end

function insert_mod!(model_mods, mods, has_codes)
    if length(mods) == 1
        push!(model_mods, mods[1] |> gpu)
        model_mods[end].has_codes = has_codes
    elseif length(mods) > 1
        push!(model_mods, Models.Chain(mods) |> gpu)
        model_mods[end].has_codes = has_codes
    end
    return []
end

function get_codes(model, inputs)
    if model.n_inputs !== nothing
        x = reshape(inputs, Int(model.n_inputs), :)
    else
        x = inputs
    end

    codes = []
    for m in model.model_mods
        x = m(x)
        if m.has_codes !== nothing && m.has_codes
            push!(codes, copy(x))
        end
    end
    return x, codes
end

function update_codes(codes, model, targets, criterion, μ, λ_c, n_iter, lr)
    model_mods = model.model_mods

    id_codes = [i for (i, m) in Iterators.enumerate(model_mods) 
                if m.has_codes !== nothing && m.has_codes]
    for l = 0:(length(codes) - 1)
        idx = id_codes[end - l]

        optimizer = Flux.Optimise.Nesterov(lr, 0.9)
        codes_initial = copy(codes[end - l])

        if (idx + 1) ∈ id_codes
            nmod = identity
            lin = model_mods[idx + 1]
        else
            if idx + 1 <= length(model_mods)
                nmod = model_mods[idx + 1]
            else
                nmod = identity
            end
            if idx + 2 <= length(model_mods)
                lin = model_mods[idx + 2]
            else
                lin = identity
            end
        end

        criterion_loss(x) = criterion((x, targets))
        mse_loss(x) = μ * Flux.Losses.mse(x, codes[end - l + 1])

        if l == 0
            loss_fn = criterion_loss
        else
            loss_fn = mse_loss
        end

        for it = 1:n_iter
            gs = Flux.gradient(Flux.params(codes[end - l])) do
                compute_code_loss(codes[end - l], nmod, lin, loss_fn, codes_initial, μ, λ_c)
            end
            Flux.Optimise.update!(optimizer, Flux.params(codes[end - l]), gs)
        end
    end

    return codes
end 

function update_last_layer(mod_out, inputs, targets, criterion, n_iter)
    for it = 1:n_iter
        outputs = mod_out(inputs)
        gs = Flux.gradient(Flux.params(mod_out)) do
            criterion((outputs, targets)) 
        end
        Flux.Optimise.update!(mod_out.optimizer, Flux.params(mod_out), gs)
    end
end

function update_hidden_weights_adam(model, inputs, codes, λ_w, n_iter)
    model_mods = model.model_mods

    id_codes = [i for (i, m) in Iterators.enumerate(model_mods) 
                if m.has_codes !== nothing && m.has_codes]

    if model.n_inputs !== nothing
        x = reshape(inputs, Int(model.n_inputs), :)
    else
        x = inputs
    end

    cins = [x]
    append!(cins, codes[1:end-1])
    for (idx, c_in, c_out) in Iterators.zip(id_codes, cins, codes)
        lin = model_mods[idx]
        if idx >= 2 && (idx - 1) ∉ id_codes
            nmod = model_mods[idx - 1]
        else
            nmod = identity
        end

        for it = 1:n_iter
            gs = Flux.gradient(Flux.params(lin)) do
                Flux.mse(lin(nmod(c_in)), copy(c_out))
                # Ignore λ_w
            end
            Flux.Optimise.update!(lin.optimizer, Flux.params(lin), gs)
        end
    end
end

function scheduler_step(model, epoch)
    model_mods = model.model_mods
    for m in model_mods
        if :scheduler ∈ fieldnames(typeof(m)) && m.scheduler !== nothing
            m.optimizer.eta = m.scheduler(epoch)
        end
    end
end

end