#ifndef __DATA_HPP
#define __DATA_HPP

#include <vector>
#include <cstdint>
#include <iostream>
#include <map>

class data {
    std::vector<uint8_t> *feature_vector;
    std::map<std::string, int> class_vector;
    int label;
    int enum_label;
    double distance;

public:
    // Constructor and Destructor
    data();
    ~data();

    // Set and Adjust the feature vector
    void set_feature_vector(std::vector<uint8_t> *vect);
    void append_to_feature_vector(double val);

    // Set Vectors, Distance, and Labels
    void set_label(int lab);
    void set_distance(double val);
    void set_class_vector_mapping(const std::string& class_name, int class_id);
    void set_enum_label(int lab);

    // Get Distance and Labels
    double get_distance() const;
    int get_label() const;
    int get_enum_label() const;

    // Getters for class vector
    int get_class_id() const;  // Name → ID
    std::string get_class_vector() const;       // ID → Name
    std::vector<uint8_t>* get_feature_vector();
    void debug_class_vector() const;
};

#endif