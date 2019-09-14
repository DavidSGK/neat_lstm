#ifndef NEAT_LSTM_CONFIG_STORE_H
#define NEAT_LSTM_CONFIG_STORE_H

#include "proto/config.pb.h"

// A singleton class for storing and accessing configuration values.
// Check proto/config.proto for the different fields.
class ConfigStore {
 public:
  // Get singleton instance
  static ConfigStore& get() {
    static ConfigStore instance;
    return instance;
  }

  // Convenience methods to get config categories of the singleton instance.
  static const Config_Mutation& mutation();
  static const Config_Speciation& speciation();
  static const Config_Bounds& bounds();

  // Reads a config object and stores it.
  void set(const Config& config);

  ConfigStore(const ConfigStore&) = delete;
  ConfigStore& operator=(const ConfigStore&) = delete;

 private:
  Config config_;

  ConfigStore(){};
};

#endif
