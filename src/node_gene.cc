#include "neat_lstm/node_gene.h"
#include "proto/structures.pb.h"

int NodeGene::id() const { return node->id(); }

Node_Type NodeGene::type() const { return node->type(); }

ActivationType NodeGene::activation_type() const {
  return node->activation_type();
}
