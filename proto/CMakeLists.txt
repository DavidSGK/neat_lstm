include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS structures.proto config.proto)
set_source_files_properties(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES GENERATED TRUE)

add_library(proto ${PROTO_HDRS} ${PROTO_SRCS})
target_link_libraries(proto ${PROTOBUF_LIBRARIES})
