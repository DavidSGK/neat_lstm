#include "neat_lstm/config_store.h"

#include "proto/config.pb.h"

const Config_Mutation& ConfigStore::mutation() {
  return get().config_.mutation();
}

const Config_Speciation& ConfigStore::speciation() {
  return get().config_.speciation();
}

const Config_Bounds& ConfigStore::bounds() {
  return get().config_.bounds();
}

void ConfigStore::set(const Config& config) { config_ = config; }
