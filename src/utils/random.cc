#include "neat_lstm/utils/random.h"

#include <cassert>
#include <random>

namespace utils {
namespace {

std::default_random_engine& generator() {
  static std::random_device rd;
  static std::default_random_engine generator{rd()};

  return generator;
}

}  // namespace

namespace random {

double uniform(double start, double end) {
  assert(start <= end);
  std::uniform_real_distribution<double> distribution{start, end};
  return distribution(generator());
}

int uniform_int(int start, int end) {
  assert(start <= end);
  std::uniform_int_distribution<int> distribution{start, end};
  return distribution(generator());
}

}  // namespace random
}  // namespace utils
