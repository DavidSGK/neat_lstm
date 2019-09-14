#ifndef NEAT_LSTM_LSTM_UNIT_GENE_H
#define NEAT_LSTM_LSTM_UNIT_GENE_H

#include <google/protobuf/repeated_field.h>
#include <vector>

#include "node_gene.h"
#include "proto/structures.pb.h"

// An actualized gene based on an LSTMUnit blueprint
class LSTMUnitGene {
 public:
  const LSTMUnit* lstm_unit;

  LSTMUnitGene(const LSTMUnit* lstm_unit,
               const std::vector<const NodeGene*>& input_node_genes)
      : lstm_unit(lstm_unit),
        input_node_genes_(input_node_genes),
        state_(lstm_unit->capacity()),
        activations_(lstm_unit->capacity()) {}

  // Perform calculations at all gates using the previous state and current
  // values of the input nodes.
  void activate();

  // Returns the activation value at the specified index.
  // Behavior is undefined before a first activate() is called.
  double activation(int index);

 private:
  std::vector<const NodeGene*> input_node_genes_;
  std::vector<double> state_;
  std::vector<double> activations_;

  // Performs a matrix multiplication of weights (linear representation of
  // capacity * (capacity + input_size)) and variables ((capacity + input_size)
  // * 1)
  std::vector<double> gate_and_squash(
      const google::protobuf::RepeatedField<double>& weights,
      const std::vector<double>& variables, double bias,
      ActivationType activation_type);
};

#endif
