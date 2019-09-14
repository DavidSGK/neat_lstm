#ifndef NEAT_LSTM_MUTATIONS_H
#define NEAT_LSTM_MUTATIONS_H

#include "proto/structures.pb.h"

// Each mutation operation has an in-place version and a copy version.
namespace mutation {

// Probabilistically performs all mutation operations based on the configuration
// parameters.
// When LSTM features are mutated, other types of mutations are not performed.
void mutate_all(Genome& source);

// Mutation to add a random conneciton.
void add_connection(Genome& source);

// Mutation to add a random node. Disables a connection and inserts 2 new
// connections and a new node between the source and target of the original
// connection.
void add_node(Genome& source);

// Mutation to toggle a random connection.
void toggle_connection(Genome& source);

// Mutation to perturb each connection weight.
// A connection may be assigned a random weight based on config parameters.
void perturb_weights(Genome& genome);

// Mutation to add a new LSTM unit to the stack of the genome.
// The output nodes of the predecessor are moved to the new unit.
void add_lstm_unit(Genome& source);

// Mutation to increase the size of a random LSTM unit's state by 1.
// The unit is expanded and a new hidden node is created that connects to all
// output nodes with weights of 1.
// TODO: Evaluate weight selection
void expand_lstm_state(Genome& source);

}  // namespace mutation

#endif
