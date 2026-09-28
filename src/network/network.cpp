#include "network.hpp"

#include <cstdlib>
#include <ctime>

static inline float rand_float() {
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}

void Network::initialise(std::initializer_list<size_t> nodes_per_layer) {
    std::srand(std::time({}));

    layers.resize(nodes_per_layer.size());
    for (size_t layers_idx = 0; layers_idx < layers.size(); ++layers_idx) {

        auto& layer = layers[layers_idx];
        auto num_nodes = nodes_per_layer.begin() + layers_idx;

        layer.nodes.resize(*num_nodes);
        for (auto& node : layer.nodes) {
            node.bias = rand_float();

            if (layers_idx + 1 >= layers.size()) continue;
            node.outgoingWeights.resize(*(num_nodes + 1));
            for (auto& weight : node.outgoingWeights) {
                weight = rand_float();
            }
        }
    }
}