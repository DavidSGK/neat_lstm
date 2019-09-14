#include <cmath>
#include <unordered_map>

#include "neat_lstm/activation.h"

namespace activation {

double sigmoid(double x) { return 1 / (1 + exp(-4.9 * x)); }

double tanh(double x) { return std::tanh(x); }

double relu(double x) {
  double g = 0.0001;

  return x > 0 ? x : g * x;
}

const std::unordered_map<ActivationType, activation_t*>& get_activation_map() {
  static const std::unordered_map<ActivationType, activation_t*>
      activation_map = {{SIGMOID, sigmoid}, {RELU, relu}, {TANH, tanh}};

  return activation_map;
}

}  // namespace activation
