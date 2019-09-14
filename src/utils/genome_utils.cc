#include "neat_lstm/utils/genome_utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "neat_lstm/config_store.h"
#include "neat_lstm/innovation.h"
#include "neat_lstm/utils/node_utils.h"
#include "neat_lstm/utils/random.h"
#include "proto/structures.pb.h"

namespace utils {

int genome_id = 0;

Genome create_genome(size_t input_size, size_t output_size) {
  Genome genome;
  genome.set_id(genome_id++);
  genome.set_input_size(input_size);
  genome.set_output_size(output_size);

  int node_id = 0;

  // Input nodes
  for (int i = 0; i < input_size; i++) {
    Node* input_node = genome.add_nodes();
    *input_node =
        create_node(node_id++, Node::INPUT, ActivationType::UNDEFINED);
  }

  // Bias node
  Node* bias_node = genome.add_nodes();
  *bias_node = create_node(node_id++, Node::BIAS, ActivationType::UNDEFINED);

  // Output nodes
  int output_start_id = node_id;
  for (int i = 0; i < output_size; i++) {
    Node* output_node = genome.add_nodes();
    *output_node =
        create_node(node_id++, Node::OUTPUT, ActivationType::SIGMOID);
  }

  // Connections
  double weight = (ConfigStore::bounds().min_weight() +
                   ConfigStore::bounds().max_weight()) /
                  2;
  for (int o = output_start_id; o < node_id; o++) {
    for (int i = 0; i < input_size; i++) {
      Connection* connection = genome.add_connections();
      connection->set_innovation(Innovation::get(i, o));
      connection->set_in_node(i);
      connection->set_out_node(o);
      connection->set_enabled(true);
      connection->set_weight(
          utils::random::uniform(ConfigStore::bounds().min_weight(),
                                 ConfigStore::bounds().max_weight()));
    }
    Connection* bias_connection = genome.add_connections();
    bias_connection->set_innovation(Innovation::get(bias_node->id(), o));
    bias_connection->set_in_node(bias_node->id());
    bias_connection->set_out_node(o);
    bias_connection->set_enabled(true);
    bias_connection->set_weight(weight);
  }

  genome.set_max_node_id(node_id - 1);

  return genome;
}

bool check_connections(const Genome& genome) {
  for (int i = 0; i < genome.connections_size() - 1; i++) {
    if (genome.connections(i).innovation() >=
        genome.connections(i + 1).innovation()) {
      return false;
    }
  }
  return true;
}

double compatibility(const Genome& a, const Genome& b) {
  // TODO: Include differences in LSTM units for compatibility measure
  int excess_count = 0;
  int disjoint_count = 0;
  int matching_count = 0;
  double weight_diff_sum = 0;

  auto a_it = a.connections().begin();
  auto b_it = b.connections().begin();

  while (a_it != a.connections().end() || b_it != b.connections().end()) {
    if (a_it == a.connections().end()) {
      excess_count++;
      b_it++;
      continue;
    } else if (b_it == b.connections().end()) {
      excess_count++;
      a_it++;
      continue;
    }
    if (a_it->innovation() == b_it->innovation()) {
      matching_count++;
      weight_diff_sum += std::abs(a_it->weight() - b_it->weight());
      a_it++;
      b_it++;
    } else if (a_it->innovation() < b_it->innovation()) {
      disjoint_count++;
      a_it++;
    } else {
      disjoint_count++;
      b_it++;
    }
  }

  int N;
  if (a.connections_size() < 20 && b.connections_size() < 20) {
    N = 1;
  } else {
    N = std::max(a.connections_size(), b.connections_size());
  }

  auto config = ConfigStore::speciation();

  return config.excess_coefficient() * excess_count / N +
         config.disjoint_coefficient() * disjoint_count / N +
         config.weights_coefficient() * weight_diff_sum / matching_count;
}

Node_Type node_type(const Genome& genome, int id) {
  if (id < genome.input_size()) {
    return Node::INPUT;
  } else if (id == genome.input_size()) {
    return Node::BIAS;
  } else if (id < genome.input_size() + genome.output_size() + 1) {
    return Node::OUTPUT;
  }
  return Node::HIDDEN;
}

int bias_id(const Genome& genome) { return genome.input_size(); }

}  // namespace utils
