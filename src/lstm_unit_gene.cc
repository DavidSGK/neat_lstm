#include <google/protobuf/repeated_field.h>
#include <cassert>
#include <vector>

#include "neat_lstm/activation.h"
#include "neat_lstm/lstm_unit_gene.h"
#include "proto/structures.pb.h"

void LSTMUnitGene::activate() {
  // Previous activation concatenated with new input values (size capacity +
  // input_size)
  std::vector<double> input_values = activations_;

  for (const auto& node_gene : input_node_genes_) {
    input_values.push_back(node_gene->activation);
  }

  // Forget gate
  auto f_t = gate_and_squash(lstm_unit->forget_weights(), input_values,
                             lstm_unit->forget_bias(), SIGMOID);

  // Input gate and new candidate values for unit state
  auto i_t = gate_and_squash(lstm_unit->input_weights(), input_values,
                             lstm_unit->input_bias(), SIGMOID);

  auto c_t = gate_and_squash(lstm_unit->state_weights(), input_values,
                             lstm_unit->state_bias(), TANH);

  // Update unit state
  for (int i = 0; i < state_.size(); i++) {
    state_.at(i) = f_t.at(i) * state_.at(i) + i_t.at(i) * c_t.at(i);
  }

  // Output gate
  auto o_t = gate_and_squash(lstm_unit->output_weights(), input_values,
                             lstm_unit->output_bias(), SIGMOID);

  // Update activations
  for (int i = 0; i < activations_.size(); i++) {
    activations_.at(i) = o_t.at(i) * activation::tanh(state_.at(i));
  }
}

double LSTMUnitGene::activation(int index) { return activations_.at(index); }

std::vector<double> LSTMUnitGene::gate_and_squash(
    const google::protobuf::RepeatedField<double>& weights,
    const std::vector<double>& variables, double bias,
    ActivationType activation_type) {
  int capacity = lstm_unit->capacity();
  int input_size = input_node_genes_.size();
  auto activation_map = activation::get_activation_map();

  // Check correct dimensions
  assert(weights.size() == capacity * (capacity + input_size));
  assert(variables.size() == capacity + input_size);
  assert(activation_map.find(activation_type) != activation_map.end());

  std::vector<double> output;
  output.reserve(capacity + input_size);

  // Matrix vector multiplication, add bias, squash
  for (int i = 0; i < capacity; i++) {
    output.at(i) = 0;
    for (int j = 0; j < capacity + input_size; j++) {
      output.at(i) += weights.Get(i * capacity + j) * variables.at(j);
    }
    output.at(i) = (*activation_map[activation_type])(output.at(i) + bias);
  }

  return output;
}
