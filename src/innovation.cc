#include "neat_lstm/innovation.h"

#include <unordered_map>

int Innovation::max_innovation_num_ = 0;
std::unordered_map<long, int> Innovation::innovations_ = {};

int Innovation::get_max() { return max_innovation_num_; }

int Innovation::get(int in_node_id, int out_node_id) {
  long key = hash(in_node_id, out_node_id);
  if (innovations_.find(key) != innovations_.end()) {
    return innovations_.at(key);
  } else {
    innovations_[key] = ++max_innovation_num_;
    return max_innovation_num_;
  }
}

long Innovation::hash(int in_node_id, int out_node_id) {
  return ((long)in_node_id) << 32 | out_node_id;
}
