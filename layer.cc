#include "../include/layer.hpp"
#include <chrono>
#include <algorithm>

double custom_tanh(double x) {
    // Asymptotes for large x
    if (x > 10.0) return 1.0;
    if (x < -10.0) return -1.0;

    // Linear approximation for small x
    if (std::abs(x) < 1e-3) return x;

    // Use standard tanh for moderate x
    return (std::exp(2 * x) - 1) / (std::exp(2 * x) + 1);
}

//Constructor: initializing filters with random values
ConvLayer::ConvLayer(int filter_size, int num_filters, double learning_rate)
    : filter_size(filter_size), num_filters(num_filters), learning_rate(learning_rate), convolution_done(false) {
    filters = new std::vector<std::vector<neuron*>>(num_filters);
    conv_output = nullptr;  // Initialize to null to avoid dangling pointers

    for (int i = 0; i < num_filters; i++) {
        (*filters)[i] = std::vector<neuron*>(filter_size * filter_size);
        for (int j = 0; j < filter_size * filter_size; j++) {
            (*filters)[i][j] = new neuron(filter_size * filter_size);
        }
    }
}

ConvLayer::~ConvLayer() {
    // Clean up the filters
    for (auto& filter : *filters) {
        for (neuron* n : filter) {
            delete n;
        }
    }
    delete filters;
}

// Forward Pass: Apply convolution on the input
std::vector<double>* ConvLayer::forward(std::vector<double>* input) {
    if (input == nullptr || input->empty()) {
        throw std::runtime_error("Error: ConvLayer received an empty input vector!");
    }

    // Check input size for squareness
    int image_size = static_cast<int>(std::sqrt(input->size()));
    if (image_size * image_size != input->size()) {
        throw std::runtime_error("Error: ConvLayer received a non-square input vector!");
    }

    // Reshape the input
    auto reshaped_input = new std::vector<std::vector<double>>(image_size, std::vector<double>(image_size));
    for (int i = 0; i < image_size; ++i) {
        for (int j = 0; j < image_size; ++j) {
            (*reshaped_input)[i][j] = (*input)[i * image_size + j];
        }
    }

    // Perform convolution
    auto convolved_output = new std::vector<double>();
    for (int i = 0; i < num_filters; ++i) {
        auto filter_output = convolve(reshaped_input, &((*filters)[i]));
        for (auto& row : *filter_output) {
            convolved_output->insert(convolved_output->end(), row.begin(), row.end());
        }
        delete filter_output;
    }
    delete reshaped_input;

    // Apply pooling
    auto pooled_output = average_pooling(convolved_output, 2);  // Example: pooling size of 2
    delete convolved_output;

    if (pooled_output == nullptr) {
        throw std::runtime_error("Error: pooled_output is null after pooling!");
    }

    if (pooled_output->size() == 0) {
        throw std::runtime_error("Error: pooled_output is empty after pooling!");
    }

    std::cout << "Convolution and pooling completed. Output size: " << pooled_output->size() << std::endl;
    return pooled_output;  // Return the processed output
}

std::vector<double>* ConvLayer::backward(std::vector<double>* gradients) {
    // Assuming d_out is the gradient of the loss w.r.t the output of this layer
    std::vector<double>* d_input = new std::vector<double>(input->size());  // Gradient w.r.t input
    std::vector<std::vector<double>> d_filters(num_filters, std::vector<double>(filter_size * filter_size)); // Gradient w.r.t filters

    // Loop over the filters and inputs to compute gradients
    for (int f = 0; f < num_filters; ++f) {
        for (int i = 0; i < input->size(); ++i) {
            for (int j = 0; j < filter_size * filter_size; ++j) {
                d_filters[f][j] += (*input)[i] * (*gradients)[i];  // Adjust the filters using chain rule
                (*d_input)[i] += (*filters)[f][j]->weights->at(j) * (*gradients)[i];  // Propagate gradient back to input
            }
        }
    }

    // Update the filters using gradients (gradient descent)
    for (int f = 0; f < num_filters; ++f) {
        for (int j = 0; j < filter_size * filter_size; ++j) {
            for (auto &n : (*filters)[f]) {
                n->weights->at(j) -= learning_rate * d_filters[f][j];  // Update weights using the learning rate
            }
        }
    }

    return d_input;  // Return the gradient w.r.t input
}

std::vector<std::vector<double>>* ConvLayer::convolve(std::vector<std::vector<double>>* input, std::vector<neuron*>* filter) {
    int input_size = input->size();  // Input size should match the reshaped 2D matrix
    int output_size = input_size - filter_size + 1;  // Calculate output size

    /*std::cout << "Starting convolution process..." << std::endl;
    std::cout << "Input size: " << input_size << ", Filter size: " << filter_size 
              << ", Output size: " << output_size << std::endl;
    std::cout << "Filter neuron count: " << filter->size() << std::endl;*/

    // Check for valid output size before proceeding
    if (output_size <= 0) {
        std::cerr << "Error: Output size is zero or negative. Input size: " << input_size 
                  << ", Filter size: " << filter_size 
                  << ", Calculated output size: " << output_size 
                  << ". Skipping this convolution layer." << std::endl;
        return nullptr;  // Skip the convolution if output size is invalid
    }

    // Proceed with convolution if sizes are valid
    auto result = new std::vector<std::vector<double>>(output_size, std::vector<double>(output_size));

    // Perform 2D convolution
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            double convolved_value = 0;
            int filter_idx = 0;
            for (int fi = 0; fi < filter_size; fi++) {
                for (int fj = 0; fj < filter_size; fj++) {
                    if (filter_idx >= filter->size()) {
                        std::cerr << "Error: Filter neuron out of bounds at index " << filter_idx << std::endl;
                        return nullptr;
                    }

                    neuron* n = (*filter)[filter_idx++];
                    if (!n) {
                        std::cerr << "Error: Null neuron at filter index " << filter_idx - 1 << std::endl;
                        return nullptr;
                    }

                    double input_value = (*input)[i + fi][j + fj];
                    double weight_value = n->weights->at(fi * filter_size + fj);

                    convolved_value += input_value * weight_value;
                }
            }
            (*result)[i][j] = convolved_value;
        }
    }

    //std::cout << "\nConvolution process completed!" << std::endl;
    return result;
}

std::vector<double>* ConvLayer::average_pooling(std::vector<double>* input, int pooling_size) {
    int input_size = static_cast<int>(std::sqrt(input->size()));
    int output_size = input_size / pooling_size;
    auto output = new std::vector<double>(output_size * output_size, 0.0);

    //std::cout << "Pooling input size: " << input->size() << " | Pooling output size: " << output->size() << std::endl;

    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            double sum = 0.0;
            for (int x = 0; x < pooling_size; x++) {
                for (int y = 0; y < pooling_size; y++) {
                    sum += (*input)[(i * pooling_size + x) * input_size + (j * pooling_size + y)];
                }
            }
            (*output)[i * output_size + j] = sum / (pooling_size * pooling_size);
        }
    }

    // Log pooled data for inspection
    return output;
}

int ConvLayer::get_pooled_output_size() const {
    return this->input->size();  // Return the size after pooling
}

// RNNLayer (LSTM/GRU) Implementation

//Initialize the hidden and cell states for usage
RNNLayer::RNNLayer(int input_size, int hidden_size, double learning_rate)
    : input_size(input_size), hidden_size(hidden_size), learning_rate(learning_rate), chunk_size(0), num_chunks(0) {
    cd->initialize_matrix(W_f, hidden_size, input_size + hidden_size);
    cd->initialize_matrix(W_i, hidden_size, input_size + hidden_size);
    cd->initialize_matrix(W_o, hidden_size, input_size + hidden_size);
    cd->initialize_matrix(W_C, hidden_size, input_size + hidden_size);

    hidden_state = new std::vector<double>(hidden_size, 0.0);
    cell_state = new std::vector<double>(hidden_size, 0.0);
    hidden_neurons = new std::vector<neuron*>();

    for (int i = 0; i < hidden_size; ++i) {
        hidden_neurons->push_back(new neuron(input_size + hidden_size));
    }

    std::cout << "Hidden neurons initialized: " << hidden_neurons->size() << " neurons." << std::endl;
}

RNNLayer::~RNNLayer() {
    delete hidden_state;
    delete cell_state;
    for (neuron* n : *hidden_neurons) {
        delete n;
    }
    delete hidden_neurons;
}

// Inside the RNNLayer's forward pass method
std::vector<double>* RNNLayer::forward(std::vector<double>* input) {
    if (!input || input->empty()) {
        throw std::runtime_error("RNNLayer::forward - Invalid input.");
    }

    std::cout << "Processing RNN Layer | Input Size: " << input->size() << std::endl;

    // Calculate chunk size and number of chunks
    chunk_size = 8192;  // Example chunk size
    num_chunks = (input->size() + chunk_size - 1) / chunk_size;  // Round up to cover all elements

    std::cout << "Chunk Size: " << chunk_size << ", Number of Chunks: " << num_chunks << std::endl;

    std::vector<double>* output = new std::vector<double>(hidden_size, 0.0);

    for (int i = 0; i < num_chunks; ++i) {
        int chunk_start = i * chunk_size;
        int chunk_end = std::min(chunk_start + chunk_size, static_cast<int>(input->size()));

        std::vector<double> chunk(input->begin() + chunk_start, input->begin() + chunk_end);
        std::vector<double>* chunk_output = lstm_forward(&chunk);

        for (int j = 0; j < hidden_size; ++j) {
            (*output)[j] += (*chunk_output)[j];  // Aggregate outputs
        }

        delete chunk_output;
    }

    return output;
}


std::vector<double>* RNNLayer::forward_chunk(std::vector<double>* chunk) {
    if (chunk == nullptr || chunk->empty()) {
        throw std::runtime_error("Invalid chunk input. Chunk is null or empty.");
    }

    std::cout << "Processing RNN chunk of size: " << chunk->size() << std::endl;

    // Placeholder for RNN forward logic
    // You can replace this with the actual computation for your RNN layer
    std::vector<double>* output = new std::vector<double>(chunk->size(), 0.0);  // Dummy output
    for (size_t i = 0; i < chunk->size(); ++i) {
        // Example: Copy input directly to output for now
        (*output)[i] = (*chunk)[i] * 0.5;  // Dummy operation
    }

    std::cout << "Chunk processed. Output size: " << output->size() << std::endl;
    return output;
}

std::vector<double>* RNNLayer::backward(std::vector<double>* d_output) {
    if (!hidden_neurons || hidden_neurons->empty()) {
        std::cerr << "[Backward] Error: Hidden neurons are not initialized or empty!" << std::endl;
        return nullptr;
    }

    if (!d_output) { // Check for null pointer only
        std::cerr << "[Backward] Error: d_output is not provided!" << std::endl;
        return nullptr;
    }

    std::vector<double>* d_input = new std::vector<double>(input_size, 0.0);

    for (int i = hidden_neurons->size() - 1; i >= 0; --i) {
        neuron* current_neuron = (*hidden_neurons)[i];
        if (!current_neuron) {
            std::cerr << "[Backward] Error: Null neuron detected at index " << i << std::endl;
            continue;
        }

        if (current_neuron->weights) {
            for (int j = 0; j < input_size; ++j) {
                if (j < current_neuron->weights->size()) {
                    (*d_input)[j] += current_neuron->delta * (*current_neuron->weights)[j];
                } else {
                    std::cerr << "[Backward] Warning: Weight index out of range for neuron " << i << std::endl;
                }
            }
        } else {
            std::cerr << "[Backward] Warning: Weights not initialized for neuron " << i << std::endl;
        }
    }

    return d_input;
}

std::vector<double>* RNNLayer::backward_chunked(std::vector<double>* d_output, int chunk_size) {
    if (chunk_size <= 0) {
        throw std::runtime_error("[RNNLayer] Invalid chunk size!");
    }

    std::vector<double>* d_input = new std::vector<double>(input_size, 0.0);

    for (int chunk_start = 0; chunk_start < d_output->size(); chunk_start += chunk_size) {
        int chunk_end = std::min(chunk_start + chunk_size, static_cast<int>(d_output->size()));
        std::vector<double> chunk_output(d_output->begin() + chunk_start, d_output->begin() + chunk_end);

        std::vector<double> dummy_cell_state(hidden_size, 0.0);
        std::vector<double>* d_chunk_input = lstm_backward(&chunk_output, &dummy_cell_state);

        // Aggregate gradients for the input
        for (size_t i = 0; i < d_chunk_input->size(); ++i) {
            (*d_input)[i] += (*d_chunk_input)[i];
        }
        delete d_chunk_input;
    }

    return d_input;
}

std::vector<double>* RNNLayer::lstm_forward(std::vector<double>* input) {
    std::vector<double>* output = new std::vector<double>(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {
        double forget_sum = 0.0, input_sum = 0.0, output_sum = 0.0, cell_candidate_sum = 0.0;

        // Compute contributions from input
        for (int j = 0; j < input->size(); ++j) {
            forget_sum += W_f[i][j] * (*input)[j];
            input_sum += W_i[i][j] * (*input)[j];
            output_sum += W_o[i][j] * (*input)[j];
            cell_candidate_sum += W_C[i][j] * (*input)[j];
        }

        // Compute contributions from hidden state
        for (int j = 0; j < hidden_state->size(); ++j) {
            forget_sum += W_f[i][input->size() + j] * (*hidden_state)[j];
            input_sum += W_i[i][input->size() + j] * (*hidden_state)[j];
            output_sum += W_o[i][input->size() + j] * (*hidden_state)[j];
            cell_candidate_sum += W_C[i][input->size() + j] * (*hidden_state)[j];
        }

        // Debug: Print summation values before activation
        std::cout << "Neuron " << i << ": Forget Sum: " << forget_sum
                  << ", Input Sum: " << input_sum
                  << ", Output Sum: " << output_sum
                  << ", Cell Candidate Sum: " << cell_candidate_sum << std::endl;

        // Clamp the summations for numerical stability
        forget_sum = std::max(-10.0, std::min(10.0, forget_sum));
        input_sum = std::max(-10.0, std::min(10.0, input_sum));
        output_sum = std::max(-10.0, std::min(10.0, output_sum));
        cell_candidate_sum = std::max(-10.0, std::min(10.0, cell_candidate_sum));

        // Apply activations
        double forget_gate = sigmoid(forget_sum);
        double input_gate = sigmoid(input_sum);
        double output_gate = sigmoid(output_sum);
        double cell_candidate = custom_tanh(cell_candidate_sum);

        // Debug: Print activation outputs
        std::cout << "Neuron " << i << ": Forget Gate = " << forget_gate
                  << ", Input Gate = " << input_gate
                  << ", Output Gate = " << output_gate
                  << ", Cell Candidate = " << cell_candidate
                  << std::endl;

        // Update cell state and hidden state
        (*cell_state)[i] = forget_gate * (*cell_state)[i] + input_gate * cell_candidate;
        (*hidden_state)[i] = output_gate * std::tanh((*cell_state)[i]);

        // Set output for this neuron
        (*output)[i] = (*hidden_state)[i];
    }

    return output;
}

std::vector<double>* RNNLayer::lstm_backward(std::vector<double>* d_output, std::vector<double>* d_next_cell_state) {
    if (!hidden_neurons || hidden_neurons->empty()) {
        throw std::runtime_error("Error: hidden_neurons is not initialized or empty!");
    }

    // Skip validation for the first neuron if d_output is empty for it
    if ((*d_output)[0] == 0) {
        std::cout << "Skipping size validation for the first neuron due to zero d_output." << std::endl;
        std::cerr << "Skipping size validation for the first neuron due to zero d_output." << std::endl;
    } else {
        // Ensure all sizes align
        if (d_output->size() != hidden_size || d_next_cell_state->size() != hidden_size) {
            std::cout << "d_output: " << d_output << std::endl;
            std::cerr << "Error: Gradient size mismatch in lstm_backward!" << std::endl;
            std::cerr << "Expected hidden_size: " << hidden_size 
                      << ", d_output size: " << d_output->size() 
                      << ", d_next_cell_state size: " << d_next_cell_state->size() << std::endl;
            throw std::runtime_error("Gradient size mismatch in lstm_backward!");
        }
    }

    std::vector<double>* d_input = new std::vector<double>(input_size, 0.0);  // Gradients w.r.t input
    std::vector<double>* d_hidden_state = new std::vector<double>(hidden_size, 0.0);  // Gradients w.r.t hidden state
    std::vector<double>* d_cell_state = new std::vector<double>(hidden_size, 0.0);  // Gradients w.r.t cell state

    std::vector<double> d_forget_gate(hidden_size, 0.0);
    std::vector<double> d_input_gate(hidden_size, 0.0);
    std::vector<double> d_output_gate(hidden_size, 0.0);
    std::vector<double> d_cell_candidate(hidden_size, 0.0);

    // Loop through each neuron in the hidden layer to compute gradients
    for (int i = 0; i < hidden_size; ++i) {
        if (!(*hidden_neurons)[i]) {
            throw std::runtime_error("Error: Null neuron at index " + std::to_string(i));
        }

        neuron* current_neuron = (*hidden_neurons)[i];

        // Skip the first neuron if d_output is zero
        if (i == 0 && (*d_output)[i] == 0) {
            std::cerr << "Skipping gradient computation for the first neuron due to zero d_output." << std::endl;
            continue;
        }

        // Gradient computations
        d_output_gate[i] = (*d_output)[i] * std::tanh((*cell_state)[i]);
        double d_tanh_cell_state = (*d_output)[i] * (*hidden_state)[i];

        d_forget_gate[i] = d_tanh_cell_state * (*cell_state)[i];
        d_input_gate[i] = d_tanh_cell_state * (*cell_state)[i];
        d_cell_candidate[i] = d_tanh_cell_state * (1 - std::pow(std::tanh((*cell_state)[i]), 2));

        (*d_cell_state)[i] = (*d_next_cell_state)[i] * d_forget_gate[i];

        for (size_t j = 0; j < current_neuron->weights->size(); ++j) {
            current_neuron->weights->at(j) -= learning_rate * d_forget_gate[i];
            (*d_input)[j] += current_neuron->weights->at(j) * d_output_gate[i];
        }
    }

    // Clean up
    delete d_hidden_state;
    delete d_cell_state;

    return d_input;
}


double RNNLayer::sigmoid(double x) {
    // Clip large values to avoid overflow/underflow
    if (x > 20.0) return 1.0;
    if (x < -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}