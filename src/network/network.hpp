#pragma once

#include "layer.hpp"

struct Network {
    std::vector<Layer> layers;

    void initialise(std::initializer_list<size_t> nodes_per_layer);
    void passLayers();
};