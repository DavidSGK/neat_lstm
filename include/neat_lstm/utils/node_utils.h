#ifndef NEAT_LSTM_UTILS_NODE_UTILS_H
#define NEAT_LSTM_UTILS_NODE_UTILS_H

#include "proto/structures.pb.h"

namespace utils {

// Creates a Node instance with the specified fields.
Node create_node(int id, Node_Type type, ActivationType activation_type);

}  // namespace utils

#endif
