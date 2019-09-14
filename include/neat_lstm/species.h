#ifndef NEAT_LSTM_SPECIES_H
#define NEAT_LSTM_SPECIES_H

#include <memory>
#include <unordered_map>
#include <vector>

#include "proto/structures.pb.h"

// A species is a grouping of genomes that are compatible with each other.
// TODO: Consider adding generational context to prune stale species
class Species {
 public:
  // Species are created with a representative genome.
  Species(std::shared_ptr<Genome> representative)
      : representative_(representative) {
    genomes_.push_back(representative);
  }

  std::shared_ptr<Genome> representative() const;

  // Adds a genome to this species.
  void add_genome(std::shared_ptr<Genome> genome);

  // Returns the genomes in this species.
  const std::vector<std::shared_ptr<Genome>> genomes() const;

  // Returns the number of organisms in the species.
  size_t size() const;

  // Measures whether this species is compatible with a genome.
  bool compatible(const Genome& genome) const;

  // Creates a set of new genomes of the specified size by excluding
  // lowest-performing genomes and breeding the survivors.
  std::vector<std::shared_ptr<Genome>> reproduce(
      const std::unordered_map<std::shared_ptr<Genome>, double>& fitnesses,
      size_t size) const;

 private:
  std::shared_ptr<Genome> representative_;
  std::vector<std::shared_ptr<Genome>> genomes_;
};

#endif
