#include "../include/data_handler.hpp"
#include <algorithm>

data_handler::data_handler(){
    data_array = new std::vector<data *>;
    training_data = new std::vector<data *>;
    testing_data = new std::vector<data *>;
    validation_data = new std::vector<data *>;
}

data_handler::~data_handler() {
    delete data_array;
    delete training_data;
    delete testing_data;
    delete validation_data;
}


void data_handler::read_data(const std::string& data_path) {
    //std::cout << "Reading data from: " << data_path << std::endl;

    std::ifstream data_file(data_path, std::ios::binary);
    if (!data_file) {
        std::cerr << "Error: Could not open data file: " << data_path << std::endl;
        exit(1);
    }

    int width = 480;    // Known dimensions from preprocessing
    int height = 270;   // Known dimensions from preprocessing
    int channels = 1;   // Grayscale images
    feature_vector_size = width * height * channels;

    //std::cout << "Width: " << width << ", Height: " << height << ", Channels: " << channels << std::endl;
    //std::cout << "Feature Vector Size: " << feature_vector_size << " bytes" << std::endl;

    data_file.seekg(0, std::ios::end);
    std::streamsize binary_file_size = data_file.tellg();
    data_file.seekg(0, std::ios::beg);

    /*if (binary_file_size % feature_vector_size != 0) {
        std::cerr << "Warning: Binary file size is not a perfect multiple of the feature vector size." << std::endl;
    }*/

    int num_images = binary_file_size / feature_vector_size;
    //std::cout << "Detected " << num_images << " images in the binary file." << std::endl;

    data_array->reserve(num_images);

    for (int i = 0; i < num_images; ++i) {
        std::vector<uint8_t>* feature_vector = new std::vector<uint8_t>(feature_vector_size);
        data_file.read(reinterpret_cast<char*>(feature_vector->data()), feature_vector_size);

        if (data_file.gcount() != feature_vector_size) {
            std::cerr << "Error: Incomplete image read at index " << i << std::endl;
            delete feature_vector;
            break;
        }

        data* d = new data();
        d->set_feature_vector(feature_vector);
        data_array->push_back(d);
    }

    data_file.close();
    std::cout << "Successfully loaded " << data_array->size() << " images into the data array." << std::endl;
}

void data_handler::read_labels(const std::string& label_path, const std::string& delimiter) {
    //std::cout << "Reading labels from: " << label_path << std::endl;

    std::ifstream labels_file(label_path);
    if (!labels_file) {
        std::cerr << "Error: Could not open labels file: " << label_path << std::endl;
        exit(1);
    }

    std::string line;
    bool is_header = true;
    int label_count = 0; // Track the number of labels processed

    while (std::getline(labels_file, line)) {
        if (is_header) {
            is_header = false; // Skip the header line
            continue;
        }

        size_t pos = line.find(delimiter);
        if (pos == std::string::npos) {
            std::cerr << "Error: Malformed line in labels file: " << line << std::endl;
            continue;
        }

        std::string class_name = line.substr(0, pos);
        int class_id = std::stoi(line.substr(pos + 1));

        if (data_array->size() <= label_count) {
            std::cerr << "Warning: More labels than images. Label ignored: " << class_name << std::endl;
            continue;
        }

        data* d = (*data_array)[label_count];
        d->set_class_vector_mapping(class_name, class_id);
        d->set_label(class_id);

        // Debugging Output
        /*std::cout << "Class Name: " << class_name
                  << ", Class ID: " << class_id
                  << ", Label Count: " << label_count << std::endl;*/

        label_count++;
    }

    labels_file.close();
    std::cout << "Successfully loaded " << label_count << " labels with mappings." << std::endl;

    // Calculate unique class counts
    count_classes();
}

void data_handler::split_data(){
    std::unordered_set<int> used_indexes;
    int train_size = data_array->size() * TRAINING_DATA_SET_PERCENTAGE;
    int testing_size = data_array->size() * TESTING_DATA_SET_PERCENTAGE;
    int validation_size = data_array->size() * VALIDATION_DATA_SET_PERCENTAGE;

    std::random_shuffle(data_array->begin(), data_array->end());

    //TRAINING DATA
    int count = 0;
    int index = 0;
    while(count < train_size){
        training_data->push_back(data_array->at(index++));
        count++;
    }

    //TESTING DATA
    count = 0;
    while(count < testing_size){
        testing_data->push_back(data_array->at(index++));
        count++;
    }

    //VALIDATION DATA
    count = 0;
    while(count < validation_size){
        validation_data->push_back(data_array->at(index++));
        count++;
    }
    printf("Training Data Size: %lu.\tTesting Data Size: %lu.\tValidation Data Size: %lu.\n", training_data->size(),testing_data->size(),validation_data->size());
}

void data_handler::count_classes() {
    std::unordered_set<int> unique_classes;

    for (data* d : *data_array) {
        int label = d->get_label();
        unique_classes.insert(label); // Add the label to the set
    }

    class_counts = unique_classes.size(); // Count unique classes

    // Debugging Output
    std::cout << "Successfully counted " << class_counts << " unique classes." << std::endl;
}

int data_handler::get_class_counts(){
    return class_counts;
}

std::vector<data *> *data_handler::get_data_array(){
    return data_array;
}

std::vector<data *> *data_handler::get_training_data(){
    return training_data;
}

std::vector<data *> *data_handler::get_testing_data(){
    return testing_data;
}

std::vector<data *> *data_handler::get_validation_data(){
    return validation_data;
}