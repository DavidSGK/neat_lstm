#include "neat_lstm/reproduction.h"

#include "neat_lstm/utils/genome_utils.h"
#include "neat_lstm/utils/random.h"
#include "proto/structures.pb.h"

namespace reproduction {

Genome crossover(const Genome& more_fit, const Genome& less_fit) {
  Genome genome;
  genome.set_id(utils::genome_id++);
  genome.set_input_size(more_fit.input_size());
  genome.set_output_size(more_fit.output_size());
  genome.set_max_node_id(more_fit.max_node_id());

  // According to NEAT, the genome will have all nodes of the more fit parent.
  genome.mutable_nodes()->CopyFrom(more_fit.nodes());

  // Based on assumption that connections are ordered by innovation number
  auto mf_it = more_fit.connections().begin();
  auto lf_it = less_fit.connections().begin();
  while (mf_it != more_fit.connections().end()) {
    if (lf_it == less_fit.connections().end() ||
        mf_it->innovation() < lf_it->innovation()) {
      // Disjoint or excess on more fit: inherit
      auto* connection = genome.add_connections();
      connection->CopyFrom(*mf_it);
      mf_it++;
    } else if (mf_it->innovation() == lf_it->innovation()) {
      // Matching gene: inherit randomly
      auto* connection = genome.add_connections();
      if (utils::random::uniform_int(0, 1) == 0) {
        connection->CopyFrom(*mf_it);
      } else {
        connection->CopyFrom(*lf_it);
      }
      mf_it++;
      lf_it++;
    } else {
      // Disjoint on less fit: ignore
      lf_it++;
    }
  }

  // TODO: Proper LSTM crossover
  genome.mutable_lstm_units()->CopyFrom(more_fit.lstm_units());

  return genome;
}

}  // namespace reproduction
