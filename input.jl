using MLDatasets

output_layer = 10

#returns array of training inputs if type == "train", testing inputs if type == "test"
function get_inputs(type::String)
    if type == "train"
        data = MNIST.traintensor()
    end
    if type == "test"
        data = MNIST.testtensor()
    end

    inputs = Array{Array}(undef, length(data[1,1,:]))

    for i = 1 : length(data[1,1,:])
        inputs[i] = vec(data[:,:,i])
    end

    inputs
end

#returns array of training labels if type == "train", testing labels if type == "test"
function get_targets(type::String)
    if type == "train"
        labels = MNIST.trainlabels()
    end
    if type == "test"
        labels = MNIST.testlabels()
    end

    targets = Array{Array}(undef, length(labels))

    for i = 1 : length(labels)
        targets[i] = zeros(output_layer)
        targets[i][labels[i] + 1] = 1
    end

    targets
end