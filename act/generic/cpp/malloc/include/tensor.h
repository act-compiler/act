#ifndef TENSOR_H
#define TENSOR_H

#include <string>
#include <vector>

#include "solver.h"
#include "storage.h"

class Instruction;

class Tensor {
private:
  std::string name;

  Storage *storage;
  std::vector<int64> sizes;
  std::vector<operations_research::IntVar *> offsets;
  Instruction *defined_by;
  std::vector<Instruction *> used_by;

public:
  enum node_type { INPUT, CONSTANT, INTERMEDIATE, OUTPUT };
  node_type type;

  Tensor(const std::string &name, Storage *storage,
         const std::vector<int64> &shape)
      : name(name), storage(storage) {

    assert(shape.size() == storage->get_total_dims());

    sizes.reserve(storage->get_addressing_dims());
    offsets.reserve(storage->get_addressing_dims());
    for (size_t i = 0; i < storage->get_addressing_dims(); i++) {
      sizes.push_back(shape[i]);
      // Offset upper bound should be capacity - size to ensure the tensor fits.
      int64 cap = storage->get_capacity(i)->Max();
      int64 max_off = cap - sizes[i];
      if (max_off < 0) max_off = 0;
      offsets.push_back(MAKE_INT_VAR(0, max_off));
    }

    defined_by = nullptr;
    used_by.resize(0);
  }

  std::string get_name() const { return name; }
  Storage *get_storage() const { return storage; }

  std::vector<int64> get_sizes() const { return sizes; }
  std::vector<operations_research::IntVar *> get_offsets() const {
    return offsets;
  }

  Instruction *get_defined_by() const { return defined_by; }

  const std::vector<Instruction *> &get_used_by() const { return used_by; }

  void set_defined_by(Instruction *instr) {
    if (defined_by != nullptr) {
      std::cerr << "Warning: Tensor " << name
                << " is already defined by another instruction." << std::endl;
    }
    defined_by = instr;
  }

  void add_used_by(Instruction *instr) { used_by.push_back(instr); }
};

#endif // TENSOR_H
