#ifndef NEAT_LSTM_MACROS_ASSERT_H
#define NEAT_LSTM_MACROS_ASSERT_H

#include <cstdio>
#include <iostream>

#ifndef NDEBUG
#define ASSERT(condition, ...)                                         \
  do {                                                                 \
    if (!(condition)) {                                                \
      std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                << " line " << __LINE__ << std::endl;                  \
      std::fprintf(stderr, __VA_ARGS__);                               \
      std::terminate();                                                \
    }                                                                  \
  } while (false)
#else
#define ASSERT(condition, message) \
  do {                             \
  } while (false)
#endif

#endif
