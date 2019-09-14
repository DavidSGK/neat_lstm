#ifndef NEAT_LSTM_REPRODUCTION_H
#define NEAT_LSTM_REPRODUCTION_H

#include "proto/structures.pb.h"

namespace reproduction {

// Crosses over the 2 parent genomes.
// Basic crossover behavior follows the NEAT methdology.
// LSTM structures are selected from the more fit genome.
// TODO: Consider crossing over weights that match
Genome crossover(const Genome& more_fit, const Genome& less_fit);

}  // namespace reproduction

#endif
