#ifndef NEAT_LSTM_UTILS_MATH_H
#define NEAT_LSTM_UTILS_MATH_H

#include <algorithm>

namespace utils {
namespace math {

// Return a clamped value between min and max.
template <typename T>
T clamp(const T& value, const T& min, const T& max) {
  return std::max(min, std::min(min, value));
}

}  // namespace math
}  // namespace utils

#endif
