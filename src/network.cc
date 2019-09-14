#include <cassert>
#include <iostream>
#include <vector>

#include "macros/assert.h"
#include "neat_lstm/activation.h"
#include "neat_lstm/network.h"
#include "proto/structures.pb.h"

Network::Network(const Genome& genome) : genome(genome) {
  // Construct map of node genes from genome and reserve input and output nodes
  for (const auto& node : genome.nodes()) {
    node_genes_.insert({node.id(), NodeGene{&node}});
    switch (node.type()) {
      case Node::INPUT: {
        input_node_genes_.push_back(&node_genes_.at(node.id()));
        break;
      }
      case Node::OUTPUT: {
        output_node_genes_.push_back(&node_genes_.at(node.id()));
        break;
      }
      case Node::BIAS: {
        node_genes_.at(node.id()).activation = 1;
        break;
      }
      default:
        break;
    }
  }

  // Construct connections map
  for (const auto& connection : genome.connections()) {
    int node_id = connection.out_node();
    if (in_connections_.find(node_id) != in_connections_.end()) {
      in_connections_.at(node_id).insert(&connection);
    } else {
      in_connections_.insert({node_id, {&connection}});
    }
  }

  if (genome.lstm_units_size() != 0) {
    // Copy input node genes to const pointers to pass to construct LSTM units
    std::vector<const NodeGene*> input_node_genes = {};
    for (const auto& input_node_gene : input_node_genes_) {
      input_node_genes.push_back(input_node_gene);
    }
    // Construct list of LSTM units
    for (const auto& lstm_unit : genome.lstm_units()) {
      lstm_unit_genes_.emplace_back(&lstm_unit, input_node_genes);
    }
  }
}

void Network::activate(const std::vector<double>& inputs) {
  // Input size must match genome schema
  assert(genome.input_size() == inputs.size());

  auto activation_map = activation::get_activation_map();

  std::vector<double> outputs;

  // Load input nodes
  for (int i = 0; i < inputs.size(); i++) {
    input_node_genes_.at(i)->activation = inputs.at(i);
  }

  // TODO: FIX!! We're using a stacked architecture now
  // Activate LSTM units and propagate to connected hidden nodes
  for (auto& lstm_unit_gene : lstm_unit_genes_) {
    lstm_unit_gene.activate();
    for (int i = 0; i < lstm_unit_gene.lstm_unit->out_nodes_size(); i++) {
      node_genes_.at(lstm_unit_gene.lstm_unit->out_nodes(i)).activation =
          lstm_unit_gene.activation(i);
    }
  }

  // Traverse through hidden/output nodes, calculating activations
  for (int i = inputs.size() + 1; i < genome.nodes_size(); i++) {
    NodeGene* node_gene = this->mutable_node_gene_by_index(i);

    // Skip this node if there are no incoming connections
    if (in_connections_.find(node_gene->id()) == in_connections_.end()) {
      continue;
    }

    double weighted_sum = 0;
    double weighted_bias = 0;
    for (const auto& connection : in_connections_.at(genome.nodes(i).id())) {
      if (!connection->enabled()) {
        continue;
      }

      NodeGene in_node_gene = node_genes_.at(connection->in_node());

      if (in_node_gene.type() == Node::BIAS) {
        weighted_bias = connection->weight() * in_node_gene.activation;
        continue;
      }

      weighted_sum += in_node_gene.activation * connection->weight();
    }

    // Make sure activation type is defined
    if (activation_map.find(node_gene->activation_type()) ==
        activation_map.end()) {
    }
    ASSERT(activation_map.find(node_gene->activation_type()) !=
               activation_map.end(),
           "Index %d, Node type: %d, Activation type: %d\n", i,
           node_gene->type(), node_gene->activation_type());

    node_gene->activation = (*activation_map.at(node_gene->activation_type()))(
        weighted_sum + weighted_bias);
  }
}

std::vector<double> Network::activations() const {
  std::vector<double> activations = {};
  for (const auto& output_node_gene : output_node_genes_) {
    activations.push_back(output_node_gene->activation);
  }
  return activations;
}

NodeGene* Network::mutable_node_gene_by_id(int id) {
  return &node_genes_.at(id);
}

NodeGene* Network::mutable_node_gene_by_index(int index) {
  return &node_genes_.at(genome.nodes(index).id());
}
