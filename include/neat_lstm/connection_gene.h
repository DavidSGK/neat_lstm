#ifndef NEAT_LSTM_CONNECTION_GENE_H
#define NEAT_LSTM_CONNECTION_GENE_H

#include "neat_lstm/node_gene.h"
#include "proto/structures.pb.h"

// An actualized gene based on the Connection blueprint.
// Holds references to the connecting NodeGenes.
class ConnectionGene {
 public:
  const Connection* connection;
  const NodeGene* in_node_gene;
  const NodeGene* out_node_gene;

  ConnectionGene(const Connection* connection) : connection(connection) {}

  double weight();
};

#endif
