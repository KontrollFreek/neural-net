#include "layer.hpp"

void Layer::passLayer(Layer* first_layer, Layer* second_layer) {
    for (size_t second_idx = 0; second_idx < second_layer->size(); ++second_idx) {
        auto second_node = second_layer->getNode(second_idx);

        float sum = *second_node.bias;

        for (size_t first_idx = 0; first_idx < first_layer->size(); ++first_idx) {
            auto first_node = first_layer->getNode(first_idx);

            sum += *first_node.value * second_node.incoming_weights->at(first_idx);
        }

        *second_node.value = second_layer->activation_function(sum);
    }
}


#include <immintrin.h>
constexpr size_t shorts_per_register = sizeof(__m512) / sizeof(float);

void Layer::passLayerSIMD(Layer* first_layer, Layer* second_layer) {
    for (size_t second_idx = 0; second_idx < second_layer->size(); second_idx += shorts_per_register) {
        auto second_node = second_layer->getNode(second_idx);

        float sum = (*second_node.bias);

        for (size_t first_idx = 0; first_idx < first_layer->size(); first_idx += shorts_per_register) {
            auto first_node = first_layer->getNode(first_idx);

            auto values = _mm512_load_ps(first_node.value);
            // TODO - check if this is actually correct, iI feel a bit fishy about this
            auto weights = _mm512_load_ps(&second_node.incoming_weights->at(first_idx));

            auto mul = _mm512_mul_ps(values, weights);

            sum += _mm512_reduce_add_ps(mul);
        }

        *second_node.value = second_layer->activation_function(sum);
    }
}
