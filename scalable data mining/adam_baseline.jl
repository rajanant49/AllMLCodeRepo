using Flux
using ArgParse
using Statistics
import Printf: @sprintf
import ParameterSchedulers: Stateful, Step, next!
import Serialization: serialize
import Base.Filesystem: pwd, joinpath

include("utils/utils.jl")
include("models.jl")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--model"
            help = "name of model: \"feedforward\", \"binary\" or \"LeNet\""
            arg_type = String
            default = "feedforward"
        "--n_hidden_layers"
            help = "number of hidden layers (ignored for LeNet)"
            arg_type = Int32
            default = Int32(2)
        "--n_hiddens"
            help = "number of hidden units (ignored for LeNet)"
            arg_type = Int32
            default = Int32(100)
        "--dataset"
            help = "name of dataset"
            arg_type = String
            default = "mnist"
        "--data_augmentation"
            help = "enables data augmentation"
            action = :store_true
        "--batch_size"
            help = "input batch size for training"
            arg_type = Int32
            default = Int32(200)
        "--epochs"
            help = "number of epochs to train for"
            arg_type = Int32
            default = Int32(50)
        "--lr"
            help = "learning rate"
            arg_type = Float32
            default = Float32(0.003)
        "--lr_decay"
            help = "learning rate decay factor per epoch"
            arg_type = Float32
            default = Float32(1.0)
        "--no_batchnorm"
            help = "disables batch-normalisation"
            action = :store_true
        "--seed"
            help = "random seed"
            arg_type = Int32
            default = Int32(1)
        "--log_interval"
            help = "how many batches to wait before logging training status"
            arg_type = Int32
            default = Int32(100)
        "--save_interval"
            help = "how many batches to wait before saving test performance (if set to zero, it does not save)"
            arg_type = Int32
            default = Int32(1000)
        "--log_first_epoch"
            help = "whether or not it should test and log after every mini-batch in first epoch"
            action = :store_true
        "--no_cuda"
            help = "disables CUDA training"
            action = :store_true
    end

    return parse_args(s, as_symbols=true)
end

function main()
    args = parse_commandline()

    device, num_gpus = Utils.get_devices(0)
    num_workers = num_gpus > 0 ? num_gpus : 1

    model_name = lowercase(args[:model])
    if model_name == "feedforward" || model_name == "binary"
        model_name = "$(model_name)_$(args[:n_hidden_layers])x$(args[:n_hiddens])"
    end
    file_name = "save_adam_baseline_$(model_name)_$(args[:dataset])_$(args[:seed]).pt"

    println("\nSGD Baseline")
    println("* Loading dataset: $(args[:dataset])")
    println("* Loading model: $(model_name)")
    println("      BatchNorm: $(!args[:no_batchnorm])")
    
    if lowercase(args[:model]) == "feedforward"
        trainloader, testloader, n_inputs = Utils.load_dataset(;namedataset=args[:dataset], batch_size=args[:batch_size])
        model = Models.FFNet(n_inputs=n_inputs, n_hiddens=args[:n_hiddens], n_hidden_layers=args[:n_hidden_layers],
                batchnorm=!args[:no_batchnorm], bias=true)
    elseif lowercase(args[:model]) == "lenet"
        trainloader, testloader, n_inputs = Utils.load_dataset(;namedataset=args[:dataset], batch_size=args[:batch_size], 
            conv_net=true, data_aug=args[:data_augmentation])
        if args[:data_augmentation]
            println("    data augmentation")
        end

        first_data = first(trainloader).data
        window_size = size(first_data, 1)
        if ndims(first_data) == 3
            num_input_channels = size(first_data, 3)
        else
            num_input_channels = 1
        end

        model = Models.LeNet(;num_input_channels=num_input_channels, window_size=window_size, bias=true)
    end

    # Multi-GPU?

    loss((x, y)) = Flux.Losses.logitcrossentropy(model(x), y)
    optimiser = Flux.Optimise.ADAM(args[:lr])
    lr_sched = Stateful(Step(args[:lr], args[:lr_decay], 1))

    perf = Utils.Performance()

    # Train, IG
    for epoch in 1:args[:epochs]
        println("\nEpoch $epoch of $(args[:epochs])")
        optimiser.eta = next!(lr_sched)
        for (batchidx, (x, y)) in enumerate(trainloader)
            train_loss = loss((x, y))

            gs = gradient(params(model)) do 
                loss((x, y)) 
            end
            Flux.Optimise.update!(optimiser, params(model), gs)

            if epoch == 1 && args[:log_first_epoch]
                push!(perf.first_epoch, Models.test(model, testloader, label=" - Test"))
            end

            if (batchidx - 1) % args[:log_interval] == 0
                train_loss_str = @sprintf "%.6f" train_loss
                println(" Train Epoch $epoch, Minibatch $batchidx: Train-loss = $train_loss_str")
            end

            if args[:save_interval] > 0 && (batchidx - 1) % args[:save_interval] == 0 && batchidx > 1
                acc = Models.test(model, test_loader, label=" - Test")
                push!(perf.te_vs_iter, acc)
            end
        end
        push!(perf.tr, Models.test(model, trainloader, label="Training"))
        push!(perf.te, Models.test(model, testloader, label="Test"))
    end

    println("\n - Training performance after each epoch: $(perf.tr)")
    println(" - Test performance after each epoch: $(perf.te)")

    out_path = joinpath(pwd(), "outputs", file_name)
    serialize(out_path, perf)
end

main()