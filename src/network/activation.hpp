#pragma once

#include <functional>

typedef std::function<float(float)> ActivationFunction;

namespace ActivationFunctions {
    float linear(float x);
    float elu(float x);
    float relu(float x);
    float leaky_relu(float x);
    float sigmoid(float x);
    float tanh(float x);
}