#include "../include/data.hpp"

// Constructor
data::data() : feature_vector(new std::vector<uint8_t>()), label(0), enum_label(0), distance(0.0) {}

// Destructor
data::~data() {
    delete feature_vector;
}

// Set and Adjust the feature vector
void data::set_feature_vector(std::vector<uint8_t> *vect) {
    delete feature_vector; // Free previous memory
    feature_vector = vect; // Assign new vector
}

void data::append_to_feature_vector(double val) {
    feature_vector->push_back(static_cast<uint8_t>(val)); // Add value to the vector
}

// Set Vectors, Distance, and Labels
void data::set_label(int lab) {
    label = lab;
}

void data::set_distance(double val) {
    distance = val;
}

// Set mapping between class name and class ID
void data::set_class_vector_mapping(const std::string& class_name, int class_id) {
    class_vector[class_name] = class_id;
}

// Set enum label
void data::set_enum_label(int lab) {
    enum_label = lab;
}

// Get Distance and Labels
double data::get_distance() const {
    return distance;
}

int data::get_label() const {
    return label;
}

int data::get_enum_label() const {
    return enum_label;
}

// Getters for class vector
int data::get_class_id() const {
    if (class_vector.empty()) {
        return -1; // Return -1 if the map is empty (no mapping exists)
    }
    for (const auto& pair : class_vector) {
        return pair.second; // Return the first class ID found
    }
    return -1; // Fallback in case of unexpected behavior
}

std::string data::get_class_vector() const {
    if (class_vector.empty()) {
        return "UNKNOWN"; // Return a default value if the map is empty
    }
    for (const auto& pair : class_vector) {
        return pair.first; // Return the first class name found
    }
    return "UNKNOWN"; // Fallback in case of unexpected behavior
}


// Debugging Class Vector
void data::debug_class_vector() const {
    std::cout << "Class Vector Contents:" << std::endl;
    for (const auto& pair : class_vector) {
        std::cout << "Class Name: " << pair.first << ", Class ID: " << pair.second << std::endl;
    }
}

// Getters for vectors
std::vector<uint8_t>* data::get_feature_vector() {
    return feature_vector;
}
