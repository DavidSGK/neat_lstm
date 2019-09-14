#include "neat_lstm/species.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <memory>
#include <unordered_map>
#include <vector>

#include "macros/assert.h"
#include "neat_lstm/config_store.h"
#include "neat_lstm/mutation.h"
#include "neat_lstm/reproduction.h"
#include "neat_lstm/utils/genome_utils.h"
#include "neat_lstm/utils/random.h"

namespace {

class FitnessComparator {
 public:
  FitnessComparator(
      const std::unordered_map<std::shared_ptr<Genome>, double>& fitnesses)
      : fitnesses_(fitnesses) {}

  bool operator()(std::shared_ptr<Genome> a, std::shared_ptr<Genome> b) {
    return fitnesses_.at(a) < fitnesses_.at(b);
  }

 private:
  std::unordered_map<std::shared_ptr<Genome>, double> fitnesses_;
};

}  // namespace

// In practice, returns first genome in list which is guaranteed to be from the
// previous generation.
std::shared_ptr<Genome> Species::representative() const {
  return genomes_.front();
}

void Species::add_genome(std::shared_ptr<Genome> genome) {
  genomes_.push_back(genome);
}

const std::vector<std::shared_ptr<Genome>> Species::genomes() const {
  return genomes_;
}

size_t Species::size() const { return genomes_.size(); }

bool Species::compatible(const Genome& genome) const {
  return utils::compatibility(*representative(), genome) <
         ConfigStore::speciation().compatibility_threshold();
}

std::vector<std::shared_ptr<Genome>> Species::reproduce(
    const std::unordered_map<std::shared_ptr<Genome>, double>& fitnesses,
    size_t size) const {
  std::vector<std::shared_ptr<Genome>> offspring;
  offspring.reserve(size);

  // If there is only 1 genome in the species, clone/mutate to reproduce
  if (genomes_.size() == 1) {
    while (offspring.size() < size) {
      auto clone = std::make_shared<Genome>(*genomes_.front());
      clone->set_id(utils::genome_id++);
      mutation::mutate_all(*clone);
      offspring.push_back(clone);
    }
    return offspring;
  } else if (size == 1) {
    offspring.push_back(
        genomes_.at(utils::random::uniform_int(0, genomes_.size() - 1)));
    return offspring;
  }

  // Otherwise make a pool of genomes to draw from the current generation
  // n = x + x * (x - 1) /  2
  // 2 * n = x + x * (x - 1)
  // 2 * n = x^2
  // x = sqrt(2 * n)
  int pool_size =
      std::min((size_t)std::ceil(std::sqrt(2 * size)), genomes_.size());
  std::vector<std::shared_ptr<Genome>> pool;
  pool.reserve(pool_size);
  std::copy(genomes_.begin(), genomes_.end(), std::back_inserter(pool));

  // Max-heap of organisms based on fitness
  auto comparator = FitnessComparator(fitnesses);
  std::make_heap(pool.begin(), pool.end(), comparator);

  // Get parents based on pool size and fitnesses
  std::vector<std::shared_ptr<Genome>> parents;
  parents.reserve(pool_size);
  for (int i = 0; i < pool_size; i++) {
    std::pop_heap(pool.begin(), pool.end(), comparator);
    parents.push_back(pool.back());
    offspring.push_back(pool.back());
    pool.pop_back();
  }
  if (offspring.size() == size) {
    return offspring;
  }

  // Cross over all combinations of parents
  for (int i = 0; i < pool_size - 1; i++) {
    for (int j = i + 1; j < pool_size; j++) {
      double fitness_a = fitnesses.at(parents.at(i));
      double fitness_b = fitnesses.at(parents.at(j));

      Genome child_genome =
          fitness_a > fitness_b
              ? reproduction::crossover(*parents.at(i), *parents.at(j))
              : reproduction::crossover(*parents.at(j), *parents.at(i));
      offspring.push_back(std::make_shared<Genome>(child_genome));
      mutation::mutate_all(*offspring.back());

      if (offspring.size() == size) {
        return offspring;
      }
    }
  }

  // In case of any remaining spaces, fill out with more of previous generation
  while (offspring.size() < size && !pool.empty()) {
    std::pop_heap(pool.begin(), pool.end(), comparator);
    offspring.push_back(pool.back());
    pool.pop_back();
  }

  // In case of still remaining spaces (e.g. disproportionately large size for
  // next generation), clone and mutate while rolling over
  for (int rolling_index = 0; offspring.size() < size; rolling_index++) {
    offspring.push_back(std::make_shared<Genome>(*offspring.at(rolling_index)));
    offspring.back()->set_id(utils::genome_id++);
    mutation::mutate_all(*offspring.back());
  }

  ASSERT(offspring.size() == size, "Offspring size: %zu, Size: %zu\n",
         offspring.size(), size);
  return offspring;
}
