#ifndef NEAT_LSTM_INNOVATION_H
#define NEAT_LSTM_INNOVATION_H

#include <unordered_map>

// Maintains the global innovation numbers of every gene mutated.
// Innovation numbers are looked up based on the input and output nodes.
class Innovation {
 public:
  // Returns the current global max innovation number.
  static int get_max();

  // Returns a known innovation number if found, or returns a newly
  // incremented innovation number if not.
  static int get(int in_node_id, int out_node_id);

 private:
  static int max_innovation_num_;
  static std::unordered_map<long, int> innovations_;

  // Technically not a hashing function, just a lazy way to store pairs of ids
  // to innovation numbers.
  static long hash(int in_node_id, int out_node_id);
};

#endif
