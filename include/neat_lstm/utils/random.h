#ifndef NEAT_LSTM_UTILS_RANDOM_H
#define NEAT_LSTM_UTILS_RANDOM_H

namespace utils {
namespace random {

// Uniformly generate a random double in [start, end]
double uniform(double start, double end);

// Uniformly generate a random int in [start, end]
int uniform_int(int start, int end);

}  // namespace random
}  // namespace utils

#endif
