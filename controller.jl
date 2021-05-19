include("network.jl")
include("input.jl")
include("weights.jl")
include("activation.jl")

global input_neurons = length(get_inputs("train")[1])
global hidden_neurons = 75
global output_neurons = 10
global weight_l_rate = 0.2
global bias_l_rate = 0.1

#runs training on all inputs of dataset
function run_training()
    init()

    parameters = [load("data/hidden_weights.jld"), load("data/output_weights.jld"), load("data/hidden_biases.jld"), load("data/output_biases.jld")]
    inputs = get_inputs("train")
    targets = get_targets("train")

    println("Running training...")
    println("")

    for i = 1 : length(inputs)
        parameters = train(inputs[i], [parameters[1], parameters[2]], [parameters[3], parameters[4]], targets[i])
    end

    println("Done!")

    save(parameters[1], "data/hidden_weights.jld")
    save(parameters[2], "data/output_weights.jld")
    save(parameters[3], "data/hidden_biases.jld")
    save(parameters[4], "data/output_biases.jld")
end

#tests the network on the entire test set
function benchmark()
    parameters = [load("data/hidden_weights.jld"), load("data/output_weights.jld"), load("data/hidden_biases.jld"), load("data/output_biases.jld")]
    inputs = get_inputs("test")
    targets = get_targets("test")
    correct = 0
    events = length(inputs)

    println("Benchmarking...")
    println("")

    for i = 1 : events
        correct += test(inputs[i], [parameters[1], parameters[2]], [parameters[3], parameters[4]], targets[i])
    end
    print("Accuracy: ")
    println(string(round((correct / events) * 100, digits=2)) * "%")
end