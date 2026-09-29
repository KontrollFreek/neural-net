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
        // TODO - this feels nasty, please fix
        auto num_nodes = nodes_per_layer.begin() + layers_idx;

        layer.values.resize(*num_nodes);
        layer.biases.resize(*num_nodes);
        layer.incoming_weights.resize(*num_nodes);
        for (size_t node_idx = 0; node_idx < layer.size(); ++node_idx) {
            auto node = layer.getNode(node_idx);

            *node.bias = rand_float();

            if (layers_idx == 0) continue;
            node.incoming_weights->resize(*(num_nodes - 1));
            for (auto& weight : *node.incoming_weights) {
                weight = rand_float();
            }
        }
    }
}