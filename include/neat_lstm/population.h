#ifndef NEAT_LSTM_POPULATION_H
#define NEAT_LSTM_POPULATION_H

#include <memory>
#include <unordered_map>
#include <vector>

#include "neat_lstm/network.h"
#include "neat_lstm/species.h"
#include "proto/structures.pb.h"

class Population {
 public:
  std::unordered_map<std::shared_ptr<Genome>, double> g_fitnesses_;
  std::vector<std::shared_ptr<Genome>> genomes_;
  // Construct a 1st generation population using the seed genome.
  // Subsequent generations should be formed as the result of reproduction.
  Population(const Genome& seed, size_t size);

  // Bucket organisms in this population into species.
  void speciate();

  // Perform reproduction/mutations on a species-basis to produce the next
  // generation of the same size.
  Population reproduce();

  int generation() const;

  size_t species_size() const;

 private:
  int generation_ = 1;
  size_t size_;
  std::vector<std::shared_ptr<Species>> species_;


  Population() : size_(0) {}
};

#endif
