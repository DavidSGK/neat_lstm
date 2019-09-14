#include "neat_lstm/utils/node_utils.h"

#include "proto/structures.pb.h"

namespace utils {

Node create_node(int id, Node_Type type, ActivationType activation_type) {
  Node node;
  node.set_id(id);
  node.set_type(type);
  node.set_activation_type(activation_type);

  return node;
}

}  // namespace utils
