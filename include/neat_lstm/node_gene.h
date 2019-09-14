#ifndef NEAT_LSTM_NODE_GENE_H
#define NEAT_LSTM_NODE_GENE_H

#include <vector>

#include "proto/structures.pb.h"

// An actualized gene based on the Node blueprint.
// Mostly a convenience wrapper that also holds an activation value.
class NodeGene {
 public:
  const Node* node;
  double activation = 0;

  NodeGene(const Node* node) : node(node) {}

  int id() const;
  Node_Type type() const;
  ActivationType activation_type() const;
};

#endif
