#include "neat_lstm/population.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>

#include "macros/assert.h"
#include "neat_lstm/mutation.h"
#include "neat_lstm/network.h"
#include "neat_lstm/utils/genome_utils.h"
#include "neat_lstm/utils/random.h"
#include "proto/structures.pb.h"

Population::Population(const Genome& seed, size_t size) : size_(size) {
  // Generate mutations of seed as organisms and initialize fitnesses
  for (int i = 0; i < size; i++) {
    Genome genome = seed;
    genome.set_id(utils::genome_id++);
    mutation::mutate_all(genome);
    auto organism = std::make_shared<Genome>(genome);
    genomes_.push_back(organism);
  }

  // Initial speciation
  speciate();
}

void Population::speciate() {
  for (const auto& genome : genomes_) {
    // Check all species, measuring compatibilities
    bool species_found = false;
    for (const auto& s : species_) {
      if (s->compatible(*genome)) {
        s->add_genome(genome);
        species_found = true;
        break;
      }
    }
    // If not compatible with any species, create a new one
    if (!species_found) {
      species_.push_back(std::make_shared<Species>(genome));
    }
  }
}

Population Population::reproduce() {
  Population population;
  population.generation_ = generation_ + 1;
  population.size_ = 0;

  // Calculate adjusted fitnesses to allocate offspring numbers of species
  double total_adjusted_fitness = 0;
  std::unordered_map<std::shared_ptr<Species>, double> species_fitnesses;
  for (const auto& s : species_) {
    species_fitnesses[s] = 0;
    for (const auto& genome : s->genomes()) {
      double adjusted_fitness = g_fitnesses_.at(genome) / s->size();
      species_fitnesses.at(s) += adjusted_fitness;
      total_adjusted_fitness += adjusted_fitness;
    }
  }

  // Generate offspring from species based on adjusted fitnesses
  for (const auto& s : species_) {
    auto offspring = s->reproduce(
        g_fitnesses_,
        std::round(species_fitnesses.at(s) / total_adjusted_fitness * size_));
    // TODO: Keep some previous species based on staleness, currently we
    // re-speciate at each generation

    // Make sure we don't exceed size of population
    size_t num_offspring = std::min(offspring.size(), size_ - population.size_);
    population.genomes_.insert(population.genomes_.end(), offspring.begin(),
                               offspring.begin() + num_offspring);
    population.size_ += num_offspring;
  }

  // Fill out remaining spaces with random genomes from current generation
  while (population.size_ < size_) {
    population.genomes_.push_back(
        genomes_.at(utils::random::uniform_int(0, size_ - 1)));
    population.size_++;
  }
  ASSERT(population.size_ == size_, "Specified: %zu, Actual: %zu\n", size_,
         population.size_);

  population.speciate();

  return population;
}

int Population::generation() const { return generation_; }

size_t Population::species_size() const { return species_.size(); }
