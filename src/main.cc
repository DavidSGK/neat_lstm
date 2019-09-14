#include <google/protobuf/text_format.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

#include "neat_lstm/config_store.h"
#include "neat_lstm/mutation.h"
#include "neat_lstm/network.h"
#include "neat_lstm/population.h"
#include "neat_lstm/utils/genome_utils.h"
#include "proto/config.pb.h"
#include "proto/structures.pb.h"

using google::protobuf::TextFormat;

// Currently running XOR test
// ./neat_lstm res/default.config
int main(int argc, char* argv[]) {
  std::ifstream config_input(argv[1]);
  std::stringstream config_buffer;
  config_buffer << config_input.rdbuf();
  Config config;
  TextFormat::ParseFromString(config_buffer.str(), &config);
  ConfigStore::get().set(config);

  Genome xor_genome = utils::create_genome(2, 1);

  Population population{xor_genome, 150};

  std::map<std::pair<double, double>, double> tests = {
      {{0, 0}, 0}, {{0, 1}, 1}, {{1, 0}, 1}, {{1, 1}, 0}};

  Genome test = utils::create_genome(2, 1);

  test.mutable_connections(0)->set_weight(1);
  test.mutable_connections(1)->set_weight(1);
  test.mutable_connections(2)->set_weight(1);

  Network test_net{test};
  test_net.activate({1, 1});
  std::cout << test_net.activations().at(0) << std::endl;

  int generations = 1000;
  for (int i = 0; i < generations; i++) {
    std::shared_ptr<Genome> best_in_gen;
    double max_fitness = -10000;
    for (const auto& genome : population.genomes_) {
      double fitness = 0;
      Network network{*genome};
      for (const auto& test : tests) {
        network.activate({test.first.first, test.first.second});
        fitness += std::abs(test.second - network.activations().at(0));
      }
      fitness = 4 - fitness;
      fitness *= fitness;
      population.g_fitnesses_[genome] = fitness;
      if (fitness > max_fitness) {
        max_fitness = fitness;
        best_in_gen = genome;
      }
    }
    std::cout << "Gen " << population.generation() << ": " << max_fitness
              << "\t\tNum species: " << population.species_size()
              << "\t\tBest in gen: " << best_in_gen->id() << std::endl;
    if (true && i == generations - 1) {
      std::cout << best_in_gen->DebugString() << std::endl;
    }
    if (false && i == 99) {
      for (const auto& fitness : population.g_fitnesses_) {
        std::cout << fitness.first->id() << "\t\t" << fitness.second
                  << std::endl;
      }
    }
    population = population.reproduce();
  }
}
