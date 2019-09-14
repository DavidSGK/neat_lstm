#include "neat_lstm/mutation.h"

#include <google/protobuf/repeated_field.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>

#include "neat_lstm/config_store.h"
#include "neat_lstm/innovation.h"
#include "neat_lstm/utils/genome_utils.h"
#include "neat_lstm/utils/math.h"
#include "neat_lstm/utils/random.h"
#include "proto/config.pb.h"
#include "proto/structures.pb.h"

namespace mutation {
namespace {

using google::protobuf::RepeatedPtrField;

// Comparison fnctor for connections based on innovation numbers.
// TODO: Consider moving to utils.
class InnovationComparator {
 public:
  InnovationComparator(bool reversed = false) : reversed_(reversed) {}

  bool operator()(const std::unique_ptr<Connection>& a,
                  const std::unique_ptr<Connection>& b) {
    return reversed_ ? a->innovation() > b->innovation()
                     : a->innovation() < b->innovation();
  }

 private:
  bool reversed_ = false;
};

// Inserts a node to the genome's node list at the specified index, transfering
// ownership of the node.
void insert_node(Genome& genome, Node* node, int index) {
  assert(index >= 0 && index < genome.nodes_size());
  int cur_index = genome.nodes_size() - 1;
  genome.mutable_nodes()->AddAllocated(node);
  for (; cur_index >= index; cur_index--) {
    assert(cur_index >= 0 && cur_index < genome.nodes_size());
    genome.mutable_nodes()->SwapElements(cur_index, cur_index + 1);
  }
}

// Inserts a node to the genome's node list to precede the node with the target
// id, transfering ownership of the node.
void insert_node_target_id(Genome& genome, std::unique_ptr<Node> node, int id) {
  int cur_index = genome.nodes_size() - 1;
  genome.mutable_nodes()->AddAllocated(node.release());
  for (; genome.nodes(cur_index).id() != id; cur_index--) {
    assert(cur_index >= 0 && cur_index < genome.nodes_size());
    genome.mutable_nodes()->SwapElements(cur_index, cur_index + 1);
  }
  assert(cur_index >= 0 && cur_index < genome.nodes_size());
  genome.mutable_nodes()->SwapElements(cur_index, cur_index + 1);
}

// Insert new connections to the genome in a way that connections are sorted by
// ascending innovation numbers. The input vector is invalidated after the
// insertion.
void insert_connections(Genome& genome,
                        std::vector<std::unique_ptr<Connection>> connections) {
  // Min-heap based on innovation numbers
  auto comparator = InnovationComparator(true);
  std::make_heap(connections.begin(), connections.end(), comparator);
  RepeatedPtrField<Connection> new_connections;

  for (auto& existing_connection : *genome.mutable_connections()) {
    if (connections.size() > 0 &&
        connections.front()->innovation() < existing_connection.innovation()) {
      std::pop_heap(connections.begin(), connections.end(), comparator);
      new_connections.AddAllocated(connections.back().release());
      connections.pop_back();
    }
    auto* new_connection = new_connections.Add();
    std::swap(*new_connection, existing_connection);
  }
  // Add excess connections
  while (connections.size() > 0) {
    std::pop_heap(connections.begin(), connections.end(), comparator);
    new_connections.AddAllocated(connections.back().release());
    connections.pop_back();
  }

  genome.mutable_connections()->Swap(&new_connections);
}

}  // namespace

void mutate_all(Genome& source) {
  if (utils::random::uniform(0, 1) < ConfigStore::mutation().p_add_node()) {
    add_node(source);
  }
  if (utils::random::uniform(0, 1) <
      ConfigStore::mutation().p_add_connection()) {
    add_connection(source);
  }
  if (utils::random::uniform(0, 1) <
      ConfigStore::mutation().p_toggle_connection()) {
    toggle_connection(source);
  }
  if (utils::random::uniform(0, 1) <
      ConfigStore::mutation().p_perturb_weights()) {
    perturb_weights(source);
  }
}

void add_connection(Genome& source) {
  // Prepare set of innovation numbers in source
  std::unordered_set<int> innovations;
  for (const auto& connection : source.connections()) {
    innovations.insert(connection.innovation());
  }

  // Only create forward connections
  int start_node_index = utils::random::uniform_int(0, source.nodes_size() - 2);
  int end_node_index =
      utils::random::uniform_int(start_node_index + 1, source.nodes_size() - 1);

  while (source.nodes(end_node_index).type() == Node::INPUT ||
         source.nodes(end_node_index).type() == Node::BIAS) {
    end_node_index++;
  }
  while (source.nodes(start_node_index).type() == Node::OUTPUT ||
         source.nodes(start_node_index).type() == Node::BIAS) {
    start_node_index--;
  }

  int start_id = source.nodes(start_node_index).id();
  int end_id = source.nodes(end_node_index).id();
  int innovation = Innovation::get(start_id, end_id);

  // Found a new innovation
  if (innovations.find(innovation) == innovations.end()) {
    auto connection = std::make_unique<Connection>();
    connection->set_innovation(innovation);
    connection->set_enabled(true);
    connection->set_in_node(start_id);
    connection->set_out_node(end_id);
    connection->set_weight(
        utils::random::uniform(ConfigStore::bounds().min_weight(),
                               ConfigStore::bounds().max_weight()));

    std::vector<std::unique_ptr<Connection>> new_connections;
    new_connections.push_back(std::move(connection));
    insert_connections(source, std::move(new_connections));
  }
}

void add_node(Genome& source) {
  int old_index = utils::random::uniform_int(0, source.connections_size() - 1);
  // If source is the bias node, look for another connection
  while (utils::node_type(source, source.connections(old_index).in_node()) ==
         Node::BIAS) {
    old_index = utils::random::uniform_int(0, source.connections_size() - 1);
  }
  int source_id = source.connections(old_index).in_node();
  int target_id = source.connections(old_index).out_node();

  // Disable old connection
  source.mutable_connections(old_index)->set_enabled(false);

  // Allocate new node with SIGMOID activation
  auto new_node = std::make_unique<Node>();
  new_node->set_type(Node::HIDDEN);
  new_node->set_activation_type(ActivationType::SIGMOID);
  new_node->set_id(source.max_node_id() + 1);
  source.set_max_node_id(new_node->id());

  // Create 2 new connections
  auto c_1 = std::make_unique<Connection>();
  c_1->set_enabled(true);
  c_1->set_in_node(source_id);
  c_1->set_out_node(new_node->id());
  c_1->set_weight(1);
  c_1->set_innovation(Innovation::get(source_id, new_node->id()));
  auto c_2 = std::make_unique<Connection>();
  c_2->set_enabled(true);
  c_2->set_in_node(new_node->id());
  c_2->set_out_node(target_id);
  c_2->set_weight(source.connections(old_index).weight());
  c_2->set_innovation(Innovation::get(new_node->id(), target_id));
  // Create connection from bias to new node
  auto c_3 = std::make_unique<Connection>();
  int bias_id = utils::bias_id(source);
  c_3->set_enabled(true);
  c_3->set_in_node(bias_id);
  c_3->set_out_node(new_node->id());
  c_3->set_weight(1);
  c_3->set_innovation(Innovation::get(bias_id, new_node->id()));

  // Insert new node right before old target node (to maintain topological
  // order)
  insert_node_target_id(source, std::move(new_node), target_id);

  // Insert new connections
  std::vector<std::unique_ptr<Connection>> new_connections;
  new_connections.push_back(std::move(c_1));
  new_connections.push_back(std::move(c_2));
  new_connections.push_back(std::move(c_3));
  insert_connections(source, std::move(new_connections));
}

void toggle_connection(Genome& source) {
  int index = utils::random::uniform_int(0, source.connections_size());
  Connection* connection = source.mutable_connections(index);
  connection->set_enabled(!connection->enabled());
}

void perturb_weights(Genome& source) {
  for (auto& connection : *source.mutable_connections()) {
    if (ConfigStore::mutation().p_randomize_weight()) {
      connection.set_weight(
          utils::random::uniform(ConfigStore::bounds().min_weight(),
                                 ConfigStore::bounds().max_weight()));
    } else {
      connection.set_weight(utils::math::clamp(
          utils::random::uniform(
              connection.weight() -
                  ConfigStore::mutation().perturb_weight_power(),
              connection.weight() +
                  ConfigStore::mutation().perturb_weight_power()),
          ConfigStore::bounds().min_weight(),
          ConfigStore::bounds().max_weight()));
    }
  }
}

void add_lstm_unit(Genome& source) {
  // TODO: Implement
}

void expand_lstm_state(Genome& source) {
  // TODO: Implement
}

}  // namespace mutation
