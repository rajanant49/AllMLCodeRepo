module Models

import Flux
using Flux: cpu, gpu
using Printf
using CUDA

mutable struct DenseMod{C}
    classifier::C
    has_codes::Union{Bool, Nothing}
    optimizer::Union{Any, Nothing}
    scheduler::Union{Any, Nothing}
end

DenseMod(n_inputs, n_outputs, bias=true) = DenseMod(
    Flux.Dense(n_inputs, n_outputs, bias=bias),
    nothing, nothing, nothing
)

Flux.@functor DenseMod
(m::DenseMod)(x) = m.classifier(x)

mutable struct BatchNormMod{C}
    classifier::C
    has_codes::Union{Bool, Nothing}
    optimizer::Union{Any, Nothing}
    scheduler::Union{Any, Nothing}
end

BatchNormMod(n_outputs) = BatchNormMod(
    Flux.BatchNorm(Int(n_outputs)),
    nothing, nothing, nothing
)

Flux.@functor BatchNormMod
(m::BatchNormMod)(x) = m.classifier(x)

mutable struct LinMod{C}
    classifier::C
    n_inputs::Int32
    has_codes::Union{Bool, Nothing}
    optimizer::Union{Any, Nothing}
    scheduler::Union{Any, Nothing}
end

function LinMod(n_inputs, n_outputs; bias=false, batchnorm=false)
    layers::Vector{Any} = [DenseMod(n_inputs, n_outputs, bias)]
    if batchnorm
        push!(layers, BatchNormMod(Int(n_outputs)))
    end
    return LinMod(Flux.Chain(layers...) |> gpu, n_inputs,
                nothing, nothing, nothing)
end

Flux.@functor LinMod
(m::LinMod)(x) = m.classifier(x)

mutable struct Relu
    has_codes::Union{Bool, Nothing}
    optimizer::Union{Any, Nothing}
    scheduler::Union{Any, Nothing}
end

Relu() = Relu(nothing, nothing, nothing)

Flux.@functor Relu
(m::Relu)(x) = Flux.relu.(x)

struct FFNet{C}
    features_vec::Vector{Any}
    classifier_vec::Vector{Any}
    classifier::C
    n_inputs::Int32
end

function FFNet(;n_inputs, n_hiddens, n_hidden_layers=Int32(2), n_outputs=Int32(10), bias=false, batchnorm=false)
    layers::Vector{Any} = [LinMod(n_inputs, n_hiddens, bias=bias, batchnorm=batchnorm), Relu()]
    for i = 1:n_hidden_layers
        push!(layers, LinMod(n_hiddens, n_hiddens, bias=bias, batchnorm=batchnorm), Relu())
    end
    push!(layers, DenseMod(n_hiddens, n_outputs))
    return FFNet([], layers, Flux.Chain(layers...) |> gpu, n_inputs)
end

Flux.@functor FFNet
(m::FFNet)(x) = m.classifier(x)

struct LeNet{F, C}
    features_vec::Vector{Any}
    classifier_vec::Vector{Any}
    features::F
    classifier::C
end

mutable struct ConvMod{C}
    classifier::C
    has_codes::Union{Bool, Nothing}
    optimizer::Union{Any, Nothing}
    scheduler::Union{Any, Nothing}
end

ConvMod(size, in, out, bias) = ConvMod(
    Flux.Conv(size, in => out; bias=bias),
    nothing, nothing, nothing
)

Flux.@functor ConvMod
(m::ConvMod)(x) = m.classifier(x)

mutable struct MaxPoolMod{C}
    classifier::C
    has_codes::Union{Bool, Nothing}
    optimizer::Union{Any, Nothing}
    scheduler::Union{Any, Nothing}
end

MaxPoolMod(size) = MaxPoolMod(
    Flux.MaxPool(size),
    nothing, nothing, nothing
)

Flux.@functor MaxPoolMod
(m::MaxPoolMod)(x) = m.classifier(x)

function LeNet(;num_input_channels=3, num_classes=10, window_size=32, bias=true)
    features_vec::Vector{Any} = [
        ConvMod((5, 5), num_input_channels, 6, bias),
        Relu(),
        MaxPoolMod((2, 2)),

        ConvMod((5, 5), 6, 16, bias),
        Relu(),
        MaxPoolMod((2, 2))
    ]
    features = Flux.Chain(features_vec...) |> gpu

    inp_size = 16 * (((window_size - 4) ÷ 2 - 4) ÷ 2) ^ 2
    classifier_vec::Vector{Any} = [
        DenseMod(inp_size, 120, bias),
        Relu(),
        DenseMod(120, 84, bias),
        Relu(),
        DenseMod(84, num_classes, bias)
    ]
    classifier = Flux.Chain(classifier_vec...) |> gpu

    return LeNet(features_vec, classifier_vec, features, classifier)
end

Flux.@functor LeNet
function (m::LeNet)(x) 
    interm = m.features(x)
    interm = reshape(interm, :, size(interm, ndims(interm)))
    return m.classifier(interm)
end

mutable struct Flatten
    has_codes::Union{Bool, Nothing}
    optimizer::Union{Any, Nothing}
    scheduler::Union{Any, Nothing}
end

Flatten() = Flatten(nothing, nothing, nothing)

Flux.@functor Flatten
(m::Flatten)(x) = reshape(x, :, size(x, ndims(x)))

function test(model, data_loader; criterion=Flux.Losses.logitcrossentropy, label="")
    test_loss, correct = 0.0, 0
    no_mini_batches = 0
    no_datapoints = 0
    for (data, targ) in data_loader
        output = model(data)
        # Check if output is a tuple or not?
        test_loss += criterion(output, targ)
        correct += count(((x, y),) -> x == y, zip(argmax(targ |> cpu, dims=1), argmax(output |> cpu, dims=1)))
        no_mini_batches += 1
        no_datapoints += size(data, ndims(data))
    end
    avg_test_loss = test_loss / no_mini_batches
    accuracy = Float32(correct) / no_datapoints
    if length(label) > 0
        loss_str = @sprintf "%.4f" avg_test_loss
        acc_str = @sprintf "%.2f" 100.0 * accuracy
        println("$label: Average loss: $loss_str, Accuracy: $correct/$no_datapoints ($acc_str%)")
    end
    return accuracy
end

mutable struct Chain{C}
    chain::C
    has_codes::Union{Bool, Nothing}
    optimizer::Union{Any, Nothing}
    scheduler::Union{Any, Nothing}
end

Chain(c) = Chain(Flux.Chain(c...), nothing, nothing, nothing)

Flux.@functor Chain
(m::Chain)(x) = m.chain(x)

end