module Utils

import CUDA

function get_devices(device_num::Int=0)
    device = CUDA.CuDevice(device_num) 
    num_gpus = length(CUDA.devices())
    return device, num_gpus
end

struct Performance
    first_epoch::Vector{Float32}
    te_vs_iter::Vector{Float32}
    tr::Vector{Float32}
    te::Vector{Float32}
end

Performance() = Performance([], [], [], [])

include("datasets.jl")

end