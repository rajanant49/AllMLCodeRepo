using MLDatasets
using Flux: gpu, Data.DataLoader, onehotbatch

function load_dataset(;namedataset="mnist", batch_size=200, data_aug=false, conv_net=false, has_gpu=true)
    if namedataset == "mnist"
        train_x, train_y = MNIST.traindata(Float32)
        test_x, test_y = MNIST.testdata(Float32)

        mean, dev = Float32(0.1307), Float32(0.3081)

        normalize!(train_x, mean, dev)
        normalize!(test_x, mean, dev)

        if !conv_net
            train_x = linearize_tensor(train_x)
            test_x = linearize_tensor(test_x)
        else
            train_x = add_channel_dim(train_x)
            test_x = add_channel_dim(test_x)
        end

        train_y, test_y = onehotbatch(train_y, 0:9), onehotbatch(test_y, 0:9)

        train_x = train_x |> gpu
        train_y = train_y |> gpu
        test_x = test_x |> gpu
        test_y = test_y |> gpu

        train_loader = DataLoader((data=train_x, label=train_y), batchsize=Int64(batch_size), shuffle=true)
        test_loader = DataLoader((data=test_x, label=test_y), batchsize=200, shuffle=true)
        n_inputs = 28 * 28
    elseif namedataset == "cifar10"
        train_x, train_y = CIFAR10.traindata(Float32)
        test_x, test_y = CIFAR10.testdata(Float32)

        mean, dev = [Float32(0.4914), Float32(0.4822), Float32(0.4465)], [Float32(0.2470), Float32(0.2435), Float32(0.2616)]
        #std dev value are different according to https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data
        # Transform test
        normalize!(test_x, mean, dev)
        if !conv_net
            test_x = linearize_tensor(test_x)
        end

        # Transform train
        if data_aug
            train_x = pad_img(train_x, 2)
            random_horizontal_flip!(train_x)
        end

        normalize!(train_x, mean, dev)
        if !conv_net
            train_x = linearize_tensor(train_x)
        end

        train_y, test_y = onehotbatch(train_y, 0:9), onehotbatch(test_y, 0:9)

        train_x = train_x |> gpu
        train_y = train_y |> gpu
        test_x = test_x |> gpu
        test_y = test_y |> gpu

        train_loader = DataLoader((data=train_x, label=train_y), batchsize=Int64(batch_size), shuffle=true)
        test_loader = DataLoader((data=test_x, label=test_y), batchsize=200, shuffle=true)
        n_inputs = 3 * 32 * 32
    end

    return train_loader, test_loader, Int32(n_inputs)
end

function normalize!(inp_tensor, means, deviations)
    if ndims(inp_tensor) == 4
        nchannels = size(inp_tensor, 3)
        for ch = 1:nchannels
            inp_tensor[:, :, ch, :] = broadcast(-, inp_tensor[:, :, ch, :], means[ch])
            inp_tensor[:, :, ch, :] = broadcast(/, inp_tensor[:, :, ch, :], deviations[ch])
        end
    else
        broadcast!(-, inp_tensor, inp_tensor, means)
        broadcast!(/, inp_tensor, inp_tensor, deviations)
    end
end

function linearize_tensor(tensor)
    dims = size(tensor)
    compressed_len = reduce(*, dims[1:ndims(tensor)-1]; init=1)
    reshape(tensor, (compressed_len, size(tensor, ndims(tensor))))
end

function pad_img(tensor, pad)
    height = size(tensor, 1)
    width = size(tensor, 2)
    out = zeros(Float32, 2*pad + height, 2*pad + width, size(tensor)[3:ndims(tensor)]...)
    out[pad+1:pad+height, pad+1:pad+width, :, :] = tensor
    return out
end

function random_horizontal_flip!(tensor, prob=Float32(0.5))
    if rand(Float32) < prob
        reverse!(tensor; dims=2)
    end
end

add_channel_dim(tensor) = reshape(tensor, (size(tensor, 1), size(tensor, 2), 1, size(tensor, 3)))
