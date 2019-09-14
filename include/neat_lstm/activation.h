#ifndef NEAT_LSTM_ACTIVATION_H
#define NEAT_LSTM_ACTIVATION_H

#include <unordered_map>

#include "proto/structures.pb.h"

typedef double activation_t(double);

namespace activation {

double sigmoid(double x);
double tanh(double x);
double relu(double x);

const std::unordered_map<ActivationType, activation_t*>& get_activation_map();

}  // namespace activation

#endif
