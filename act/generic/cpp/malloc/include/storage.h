#ifndef STORAGE_H
#define STORAGE_H

#include <map>
#include <string>
#include <vector>

#include "solver.h"

class Storage {
private:
  std::string name;
  std::vector<operations_research::IntVar *> addressing_capacity;

  size_t addressing_dims;
  size_t non_addressing_dims;

public:
  Storage(const std::string &name,
          const std::vector<int64> &addressing_capacity_,
          const int non_addressing_dims)
      : name(name), addressing_dims(addressing_capacity_.size()),
        non_addressing_dims(non_addressing_dims) {
    this->addressing_capacity.reserve(addressing_capacity_.size());
    for (auto cap : addressing_capacity_) {
      this->addressing_capacity.push_back(MAKE_INT_KNOWN(cap));
    }
  }

  std::string get_name() const { return name; }
  operations_research::IntVar *get_capacity(int dim) const {
    return addressing_capacity[dim];
  }
  int64 get_size() const {
    int64 prod = 1;
    for (auto cap : addressing_capacity) {
      prod *= cap->Max();
    }
    return prod;
  }

  size_t get_non_addressing_dims() const { return non_addressing_dims; }
  size_t get_addressing_dims() const { return addressing_dims; }
  size_t get_total_dims() const {
    return addressing_dims + non_addressing_dims;
  }
};

extern std::map<std::string, Storage *> g_storage;

#define STORAGE_CAPACITY(buffer, dim) (g_storage[buffer]->get_capacity(dim))
#define STORAGE_SIZE(buffer) (g_storage[buffer]->get_size())
#define HBM_SIZE (1 << 30)

#endif // STORAGE_H
