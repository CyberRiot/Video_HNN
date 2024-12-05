#include "../include/network.hpp"
#include <algorithm>

network::network(std::vector<int> *spec, int input_size, int num_classes, double learning_rate) {
    this->input_size = input_size;
    this->num_classes = num_classes;
    this->learning_rate = learning_rate;
    layers = new std::vector<layer *>();

    std::cout << "Starting layer initialization..." << std::endl;

    int current_input_size = input_size;  // Track the input size for each layer

    // Initialize the ConvLayer first
    int conv_filter_size = 3;  // Use filter size 3 for convolution
    int conv_num_filters = spec->at(0);  // First element in spec is the number of filters
    layers->push_back(new ConvLayer(conv_filter_size, conv_num_filters, learning_rate));

    std::cout << "ConvLayer initialized with filter size " << conv_filter_size 
              << " and " << conv_num_filters << " filters." << std::endl;

    // Now, get the output size after convolution and pooling (or flattening if pooling is removed)
    ConvLayer* conv_layer = static_cast<ConvLayer*>(layers->back());

    // Run a dummy forward pass to get the size after convolution and pooling/flattening
    std::vector<double> dummy_input(input_size, 0.0);  // Create a dummy input with the original input size
    std::vector<double>* conv_output = conv_layer->forward(&dummy_input);
    std::cout << "constructor under conv_output" << std::endl;
    int pooled_output_size = conv_output->size();  // Get the size after convolution and pooling/flattening

    std::cout << "Pooled output size from ConvLayer: " << pooled_output_size << std::endl;

    // Set up chunking for RNN input size if necessary
    int chunk_size = 8192;  // Adjust chunk size based on available resources
    int num_chunks = pooled_output_size / chunk_size;
    int remaining_elements = pooled_output_size % chunk_size;

    std::cout << "Number of chunks: " << num_chunks << ", Remaining elements: " << remaining_elements << std::endl;

    // Use the pooled output size as the input size for the first RNNLayer
    current_input_size = chunk_size;

    // Initialize RNN layers in chunks
    for (int i = 1; i < spec->size(); i++) {
        int hidden_size = spec->at(i);  // Hidden size for each layer
        /*std::cout << "Initializing RNNLayer with input size (chunked) " << current_input_size 
                  << " and hidden size " << hidden_size << "." << std::endl;*/

        layers->push_back(new RNNLayer(current_input_size, hidden_size, learning_rate));

        current_input_size = hidden_size;  // Update input size for the next RNNLayer
    }

    // Initialize the final output layer (RNNLayer) in chunks
    std::cout << "Initializing final RNNLayer with input size (chunked) " << current_input_size 
              << " and output size " << num_classes << "." << std::endl;
    layers->push_back(new RNNLayer(current_input_size, num_classes, learning_rate));

    std::cout << "Network initialized with " << layers->size() << " layers." << std::endl;

    delete conv_output;  // Clean up the dummy input/output after determining sizes
}

network::~network() {
    for (layer *l : *layers) {
        delete l;
    }
    delete layers;
    close_debug_output();
}

std::vector<double>* network::fprop(data* d) {
    std::vector<double>* input = new std::vector<double>(d->get_feature_vector()->begin(), d->get_feature_vector()->end());

    std::cout << "Forward Pass | Initial Feature Vector Size: " << input->size() << std::endl;

    for (int i = 0; i < layers->size(); ++i) {
        layer* current_layer = (*layers)[i];

        std::cout << "Layer " << i << " - Input Size: " << input->size() << std::endl;

        if (RNNLayer* rnn_layer = dynamic_cast<RNNLayer*>(current_layer)) {
            std::cout << "Processing RNNLayer" << std::endl;

            // Use RNNLayer's input_size to determine chunk size
            int chunk_size = rnn_layer->input_size;
            int calculated_chunks = (input->size() + chunk_size - 1) / chunk_size;

            std::cout << "Calculated chunks: " << calculated_chunks << std::endl;

            // Set num_chunks
            cd->set_num_chunks(&calculated_chunks);

            std::vector<double>* rnn_output = new std::vector<double>();

            for (int chunk_index = 0; chunk_index < cd->get_num_chunks(); ++chunk_index) {
                int chunk_start = chunk_index * chunk_size;
                int chunk_end = std::min(chunk_start + chunk_size, static_cast<int>(input->size()));

                if (chunk_start >= input->size()) {
                    throw std::runtime_error("Invalid chunk start index.");
                }

                std::vector<double> chunk(input->begin() + chunk_start, input->begin() + chunk_end);

                // Pad the last chunk if necessary
                if (chunk.size() < rnn_layer->input_size) {
                    chunk.resize(rnn_layer->input_size, 0.0);  // Pad with zeros
                }

                std::vector<double>* chunk_output = rnn_layer->forward(&chunk);

                if (!chunk_output) {
                    std::cerr << "Error: chunk_output is null!" << std::endl;
                    return nullptr;
                }

                rnn_output->insert(rnn_output->end(), chunk_output->begin(), chunk_output->end());
                delete chunk_output;
            }

            delete input;
            input = rnn_output;

        } else {
            input = current_layer->forward(input);

            if (!input || input->empty()) {
                std::cerr << "Error: Input is null or empty after forward pass in layer " << i << "!" << std::endl;
                return nullptr;
            }

            std::cout << "Layer " << i << " forward pass completed with output size: " << input->size() << std::endl;
        }
    }

    return input;
}

void network::initialize_chunks(common* cd, std::vector<double>* input) {
    int max_chunks = 50;  // Upper limit for number of chunks
    int chunk_size = std::max(1, static_cast<int>(input->size() / max_chunks));  // Ensure a valid chunk size
    int calculated_chunks = (input->size() + chunk_size - 1) / chunk_size;      // Round up for remaining elements

    cd->set_num_chunks(&calculated_chunks);
    std::cout << "Chunks initialized: " << cd->get_num_chunks() << " with chunk size: " << chunk_size << std::endl;
}

void network::bprop(data* d) {
    if (!d) {
        std::cerr << "[bprop] Error: Null data provided!" << std::endl;
        return;
    }

    // Perform forward propagation
    std::vector<double>* output = fprop(d);
    if (!output) {
        std::cerr << "[bprop] Error: Forward pass returned null output!" << std::endl;
        return;
    }

    // Debug: Output the size of the forward pass result
    std::cout << "[bprop] Forward pass output size: " << output->size() << std::endl;

    // Initialize gradients for the final layer
    std::vector<double>* gradients = new std::vector<double>(output->size(), 0.0);

    int class_id = d->get_class_id();
    if (class_id < 0 || class_id >= static_cast<int>(gradients->size())) {
        std::cerr << "[bprop] Error: Invalid class ID!" << std::endl;
        delete gradients;
        delete output;
        return;
    }

    // Calculate the error for the final layer
    for (size_t i = 0; i < output->size(); i++) {
        (*gradients)[i] = (i == static_cast<size_t>(class_id)) ?
            ((*output)[i] - 1.0) * transfer_derivative((*output)[i]) :
            ((*output)[i] - 0.0) * transfer_derivative((*output)[i]);
    }

    // Backpropagate through the layers
    std::vector<double>* d_next = gradients;
    for (int i = layers->size() - 1; i >= 0; i--) {
        layer* current_layer = (*layers)[i];
        std::cout << "[bprop] Backward pass for layer " << i << std::endl;

        d_next = current_layer->backward(d_next);
        if (!d_next) {
            std::cerr << "[bprop] Error: Backward pass failed at layer " << i << "!" << std::endl;
            delete gradients;
            delete output;
            return;
        }
    }

    // Clean up
    delete gradients;
    delete output;

    std::cout << "[bprop] Backpropagation completed successfully." << std::endl;
}

void network::update_weights(data *d) {
    for (int i = 0; i < layers->size(); i++) {
        layer* current_layer = (*layers)[i];

        // Check if current layer is a ConvLayer or RNNLayer
        if (ConvLayer* conv_layer = dynamic_cast<ConvLayer*>(current_layer)) {
            // This is a Convolutional Layer, update weights for ConvLayer
            for (int f = 0; f < conv_layer->num_filters; f++) {
                for (int j = 0; j < conv_layer->filter_size * conv_layer->filter_size; j++) {
                    for (auto& n : (*conv_layer->filters)[f]) {
                        // This is the rule for the simple weight update
                        n->weights->at(j) -= learning_rate * n->delta;
                    }
                }
            }
        }
        else if (RNNLayer* rnn_layer = dynamic_cast<RNNLayer*>(current_layer)) {
            // Update weights for RNNLayer
            for (int h = 0; h < rnn_layer->hidden_size; h++) {
                for (int w = 0; w < rnn_layer->input_size; w++) {
                    (*rnn_layer->hidden_neurons)[h]->weights->at(w) -= learning_rate * (*rnn_layer->hidden_neurons)[h]->delta;
                }
            }
        }
    }
}

double network::transfer(double activation) {
    return 1.0 / (1.0 + std::exp(-activation));
}

double network::transfer_derivative(double output) {
    return output * (1 - output);
}

int network::predict(data* d) {
    std::vector<double>* output = fprop(d);
    int predicted_class = std::distance(output->begin(), std::max_element(output->begin(), output->end()));
    delete output;
    return predicted_class;
}



// Training: Pass data through the network, forward pass only for now
void network::train(int epochs, double validation_threshold) {
    if (common_training_data == nullptr || common_training_data->empty()) {
        std::cerr << "Error: Training data is empty!" << std::endl;
        return;
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;

        // Iterate through each batch
        for (data* d : *common_training_data) {
            std::vector<double>* output = nullptr;
            try {
                output = fprop(d);  // Forward pass
            } catch (const std::exception& e) {
                std::cerr << "An exception occurred during fprop: " << e.what() << std::endl;
                continue;  // Continue processing the next sample
            }

            // Retrieve the class ID
            int class_id = d->get_class_id();  // Replace with your class name key
            if (class_id < 0) {
                std::cerr << "Error: Invalid class ID!" << std::endl;
                delete output;
                continue;
            }

            // Calculate the loss for this batch
            double sample_loss = calculate_loss(output, class_id);
            total_loss += sample_loss;

            // Backpropagate the error and update weights
            bprop(d);
            update_weights(d);

            // Clean up after processing this batch
            delete output;
        }

        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Total Loss: " << total_loss << std::endl;

        // Validate after each epoch to track performance
        double validation_accuracy = 0.0;
        for (data* vd : *common_validation_data) {
            validation_accuracy += validate(vd);
        }

        validation_accuracy /= common_validation_data->size();
        std::cout << "Validation Accuracy: " << validation_accuracy * 100 << "%" << std::endl;

        // Early stopping if accuracy exceeds the validation threshold
        if (validation_accuracy >= validation_threshold) {
            std::cout << "Stopping early due to high validation accuracy." << std::endl;
            break;
        }
    }
}


void network::set_debug_output(const std::string &filename) {
    debug_output.open(filename);
    if (!debug_output.is_open()) {
        std::cerr << "Error opening debug file: " << filename << std::endl;
    }
}

void network::close_debug_output() {
    if (debug_output.is_open()) {
        debug_output.close();
    }
}

void network::save_model(const std::string &filename) {
    std::ofstream outfile(filename);
    if(!outfile.is_open()){
        std::cerr << "Error: Could not open file " << filename << " for saving model." << std::endl;
        exit(1);
    }

    int total_layers = layers->size();
    int processed_layers = 0;

    // Save each layer's weights
    for(layer* l : *layers){
        ConvLayer* conv_layer = dynamic_cast<ConvLayer*>(l);
        RNNLayer* rnn_layer = dynamic_cast<RNNLayer*>(l);

        if(conv_layer != nullptr){
            for(auto& filter : *conv_layer->filters){
                for(neuron* n : filter){
                    for(double weight : *n->weights){
                        outfile << weight << " ";
                    }
                    outfile << std::endl;
                }
            }
        }
        if(rnn_layer != nullptr){
            for(neuron* n : *rnn_layer->hidden_neurons){
                for(double weight : *n->weights){
                    outfile << weight << " ";
                }
                outfile << std::endl;
            }
        }

        processed_layers++;
    }

    outfile.close();
    std::cout << "Model saved to " << filename << std::endl;
}

void network::load_model(const std::string &filename) {
    std::ifstream infile(filename);
    if(!infile.is_open()){
        std::cerr << "Error: could not open file " << filename << " for loading model." << std::endl;
        exit(1);
    }
    for(layer* l : *layers){
        ConvLayer* conv_layer = dynamic_cast<ConvLayer*>(l);
        RNNLayer* rnn_layer = dynamic_cast<RNNLayer*>(l);

        if(conv_layer != nullptr){
            for(auto& filter : *conv_layer->filters){
                for(neuron* n : filter){
                    for(double& weight : *n->weights){
                        infile >> weight;
                    }
                }
            }
        }
        if(rnn_layer != nullptr){
            for(neuron* n : *rnn_layer->hidden_neurons){
                for(double& weight : *n->weights){
                    infile >> weight;
                }
            }
        }
    }
    infile.close();
    std::cout << "Model loaded from " << filename << std::endl;
}

double network::validate(data* d) {
    // Get predicted class using the updated predict function
    int predicted_class = predict(d);

    // Retrieve the actual class ID
    int actual_class_id = d->get_class_id();  // Replace with your class name key

    // Compare predicted class with actual class
    if (predicted_class == actual_class_id) {
        return 1.0;  // Accuracy for this sample (correct prediction)
    } else {
        return 0.0;  // Accuracy for this sample (incorrect prediction)
    }
}


double network::test() {
    if (common_testing_data == nullptr || common_testing_data->empty()) {
        std::cerr << "Error: Testing data is empty!" << std::endl;
        return 0.0;
    }

    int correct = 0;
    int total = 0;

    for (data* d : *common_testing_data) {
        // Get predicted class using the updated predict function
        int predicted_class = predict(d);

        // Retrieve the actual class ID for this data point
        int actual_class_id = d->get_class_id();

        // Check if the prediction is correct
        if (predicted_class == actual_class_id) {
            correct++;
        }
        total++;
    }

    // Return the accuracy as the ratio of correct predictions
    return static_cast<double>(correct) / total;
}

void network::output_predictions(const std::string &filename, data_handler *dh) {
    // Output predictions to CSV or other file formats
}

double network::calculate_loss(std::vector<double>* output, int class_id) {
    if (class_id < 0 || class_id >= output->size()) {
        throw std::runtime_error("Invalid class ID: " + std::to_string(class_id));
    }

    double target = 1.0;  // Assume binary classification for simplicity (adjust for multi-class)
    double prediction = (*output)[class_id];

    // Prevent log(0) by clamping predictions between a small epsilon value and 1 - epsilon
    prediction = std::min(std::max(prediction, 1e-15), 1.0 - 1e-15);

    // Binary Cross-Entropy Loss for the target class
    return -target * std::log(prediction) - (1 - target) * std::log(1 - prediction);
}

// In Main
int main() {
    try {
        // Initialize the data handler
        data_handler *dh = new data_handler();

        std::string data_path = "C:\\Users\\cyber\\Desktop\\Code Fix\\data\\output_binary_file.data";
        std::string labels_path = "C:\\Users\\cyber\\Desktop\\Code Fix\\data\\VOOD_labels.csv";
        std::string delimiter = ",";

        dh->read_data(data_path);
        dh->read_labels(labels_path, delimiter);
        dh->split_data();

        // Initialize the network
        std::vector<int> *spec = new std::vector<int>{128, 64, 3};
        network *net = new network(spec, dh->get_training_data()->at(0)->get_feature_vector()->size(), 3, 0.01);

        // Set debug output file
        net->set_debug_output("debug_output.txt");

        // Assign the split data to the network's inherited common_data members
        net->set_common_training_data(dh->get_training_data());
        net->set_common_testing_data(dh->get_testing_data());
        net->set_common_validation_data(dh->get_validation_data());

        // Train the network
        std::cout << "Starting training..." << std::endl;
        net->train(10, 0.98);  // Train for 10 epochs

        // Save the trained model
        net->save_model("E:\\Code\\VOOD\\data\\saved_model\\trained_model.bin");
        std::cout << "Model saved successfully." << std::endl;

        // Test the model on test data
        double test_accuracy = net->test();
        std::cout << "Test Accuracy: " << test_accuracy << std::endl;

        // Cleanup
        delete dh;
        delete net;
        delete spec;
    } catch (const std::exception &e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "An unknown exception occurred!" << std::endl;
    }

    return 0;
}
