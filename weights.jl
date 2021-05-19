using JLD
using Distributions

#initializes random weights and biases
function init()
    save(random_weights(input_neurons, hidden_neurons), "data/hidden_weights.jld")
    save(random_weights(hidden_neurons, output_neurons), "data/output_weights.jld")
    save(random_biases(hidden_neurons), "data/hidden_biases.jld")
    save(random_biases(output_neurons), "data/output_biases.jld")
end

#creates weight array with random values
function random_weights(n_inputs, n_neurons)
    rand(Uniform(0.005, 0.05), n_neurons, n_inputs)
end

#creates bias array with random values
function random_biases(n_neurons)
    rand(n_neurons)
end

#saves data at path
function save(data, path)
    @save path data
end

#loads data from path
function load(path)
    f = jldopen(path)
    o = read(f["data"])
    close(f)
    o
end