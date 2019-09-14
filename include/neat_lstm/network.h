#ifndef NEAT_LSTM_NETWORK_H
#define NEAT_LSTM_NETWORK_H

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "neat_lstm/lstm_unit_gene.h"
#include "neat_lstm/node_gene.h"
#include "proto/structures.pb.h"

// A network is the phenotype representation of a genome and acts as an organism
// that can be bred with others.
class Network {
 public:
  const Genome genome;

  Network(const Genome& genome);

  // Performs the propagation of the input through the network. The
  // states/activations of all nodes and LSTM units are updated.
  void activate(const std::vector<double>& inputs);

  // Return the current activations of the output nodes. Behavior is undefined
  // before the first call to activate().
  std::vector<double> activations() const;

  NodeGene* mutable_node_gene_by_id(int id);
  NodeGene* mutable_node_gene_by_index(int index);

 private:
  // Map of node ids to node genes
  std::unordered_map<int, NodeGene> node_genes_;
  // Map of node ids to the set of their incoming connections
  std::unordered_map<int, std::unordered_set<const Connection*>>
      in_connections_;
  std::vector<LSTMUnitGene> lstm_unit_genes_;
  std::vector<NodeGene*> input_node_genes_;
  std::vector<NodeGene*> output_node_genes_;
};

#endif
