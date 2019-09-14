#ifndef NEAT_LSTM_UTILS_GENOME_UTILS_H
#define NEAT_LSTM_UTILS_GENOME_UTILS_H

#include "proto/structures.pb.h"

namespace utils {

extern int genome_id;

// Create a basic genome with the specified numbers of input and output nodes.
// One bias node is automatically created, and all input/bias nodes are
// connected to all output nodes with connections of random weights. Output
// nodes will have SIGMOID activation types.
Genome create_genome(size_t input_size, size_t output_size);

// TODO: create_genome with LSTM

// Checks that connections are sorted by ascending innovation numbers.
bool check_connections(const Genome& genome);

// Measures how compatible 2 genomes are.
// Takes into account connection genes and LSTM units.
double compatibility(const Genome& a, const Genome& b);

// Returns the type of a node with the specified id.
// NOTE: This method relies on specific creation logic for genomes and results
// are likely to be invnalid for genomes with manually altered node ids.
Node_Type node_type(const Genome& genome, int id);

// Returns the id of the bias node.
// NOTE: This method relies on specific creation logic for genomes.
int bias_id(const Genome& genome);

}  // namespace utils

#endif
