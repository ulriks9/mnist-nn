#returns 1 if the network classifies the input correctly, 0 otherwise
function test(input, weights, biases, target)
    hidden_weights = weights[1]
    output_weights = weights[2]
    hidden_biases = biases[1]
    output_biases = biases[2]

    hidden_output = forward(input, hidden_weights, hidden_biases)
    output_output = forward(hidden_output, output_weights, output_biases)

    findmax(output_output)[2] == findmax(target)[2] ? 1 : 0
end

#trains network on input based on target
function train(input, weights, biases, target)
    hidden_weights = weights[1]
    output_weights = weights[2]
    hidden_biases = biases[1]
    output_biases = biases[2]

    #produces output from hidden layer
    hidden_output = forward(input, hidden_weights, hidden_biases)

    #produces output from output layer
    output_output = forward(hidden_output, output_weights, output_biases)
    
    #calculates error at output layer
    output_error = target - output_output

    #calculates error at hidden layer
    hidden_error = transpose(output_weights) * output_error

    #updates weights of output layer
    delta_w = weight_l_rate * output_error .* d_sigmoid.(output_output) * transpose(hidden_output)
    output_weights += delta_w

    #updates weights of hidden layer
    delta_w = weight_l_rate * hidden_error .* d_sigmoid.(hidden_output) * transpose(input)
    hidden_weights += delta_w

    #updates biases of output layer
    delta_b = bias_l_rate * output_error .* d_sigmoid.(output_output)
    output_biases += delta_b

    #updates biases of hidden layer
    delta_b = bias_l_rate * hidden_error .* d_sigmoid.(hidden_output)
    hidden_biases += delta_b

    [hidden_weights, output_weights, hidden_biases, output_biases]
end

#generates outputs based on inputs and weights
function forward(input, weights, bias)
    output = weights * input
    output += bias
    sigmoid.(output)
end