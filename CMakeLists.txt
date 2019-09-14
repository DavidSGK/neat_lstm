cmake_minimum_required(VERSION 3.1)
project(NEAT-LSTM)

set(CMAKE_CXX_STANDARD 14)

include_directories(
  ./include
  ./src
  ${CMAKE_CURRENT_BINARY_DIR}
)
add_subdirectory(proto)

set(
  PROJECT_SRCS
  src/activation.cc
  src/config_store.cc
  src/connection_gene.cc
  src/lstm_unit_gene.cc
  src/innovation.cc
  src/mutation.cc
  src/network.cc
  src/node_gene.cc
  src/population.cc
  src/reproduction.cc
  src/species.cc
  src/utils/genome_utils.cc
  src/utils/node_utils.cc
  src/utils/random.cc
)
set(
  PROJECT_HDRS
  include/neat_lstm/activation.h
  include/neat_lstm/config_store.h
  include/neat_lstm/connection_gene.h
  include/neat_lstm/lstm_unit_gene.h
  include/neat_lstm/innovation.h
  include/neat_lstm/mutation.h
  include/neat_lstm/network.h
  include/neat_lstm/node_gene.h
  include/neat_lstm/population.h
  include/neat_lstm/reproduction.h
  include/neat_lstm/species.h
  include/neat_lstm/utils/genome_utils.h
  include/neat_lstm/utils/math.h
  include/neat_lstm/utils/node_utils.h
  include/neat_lstm/utils/random.h
)
set(
  INTERNAL_HDRS
  src/macros/assert.h
)

add_executable(neat_lstm_bin ${PROJECT_HDRS} ${INTERNAL_HDRS} ${PROJECT_SRCS} src/main.cc)
target_link_libraries(neat_lstm_bin proto)
set_target_properties(neat_lstm_bin PROPERTIES OUTPUT_NAME neat_lstm)
