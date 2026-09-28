#include "activation.hpp"

#include <cmath>

static constexpr float s_alpha = 0.01;

namespace ActivationFunctions {
    float linear(float x) {
        return x;
    }

    float elu(float x) {
        return x >= 0 ? x : s_alpha * (std::exp(x) - 1);
    }

    float relu(float x) {
        return x > 0 ? x : 0;
    }

    float leaky_relu(float x) {
        return x > 0 ? x : s_alpha * x;
    }

    float sigmoid(float x) {
        return 1.f / (1.f + std::exp(-x));
    }

    float tanh(float x) {
        float exp = std::exp(x), n_exp = std::exp(-x);
        return (exp - n_exp) / (exp + n_exp);
    }
}