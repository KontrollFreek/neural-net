#pragma once

#include "activation.hpp"
#include <vector>

struct Node {
    float* value;
    float* bias;
    std::vector<float>* incoming_weights;
};

struct Layer {
    std::vector<float> values;
    std::vector<float> biases;
    std::vector<std::vector<float>> incoming_weights;

    ActivationFunction activation_function;
    

    inline size_t size() {
        return values.size();
    }
    inline Node getNode(size_t i) {
        return {
            &values[i],
            &biases[i],
            &incoming_weights[i]
        };
    }
    
    static void passLayer(Layer* first_layer, Layer* second_layer);
    static void passLayerSIMD(Layer* first_layer, Layer* second_layer);
};