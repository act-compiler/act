#ifndef INSTRUCTIONS_H
#define INSTRUCTIONS_H

#include <string>
#include <utility>
#include <vector>

#include "solver.h"
#include "storage.h"
#include "tensor.h"

using IntExpr = operations_research::IntExpr;
using IntVar = operations_research::IntVar;
using Constraints = std::vector<operations_research::Constraint *>;

class Instruction {
protected:
  std::string op_name;
  Tensor *y;
  std::vector<Tensor *> X;

public:
  Instruction(const std::string &op_name, Tensor *y,
              const std::vector<Tensor *> &X)
      : op_name(op_name), y(y), X(X) {
    if (y != nullptr)
      y->set_defined_by(this);
    for (auto *x : X) {
      if (x != nullptr)
        x->add_used_by(this);
    }
  }

  virtual ~Instruction() = default;

  virtual std::string get_op_name() const { return op_name; }
  virtual std::string str() = 0;

  virtual Tensor *get_lhs() { return y; }

  virtual std::vector<Tensor *> get_rhs() { return X; }
  virtual std::vector<bool> get_rhs_inplace() {
    return std::vector<bool>(X.size(), false);
  }

  virtual std::vector<std::pair<IntExpr *, IntExpr *>> get_h(int num) = 0;
  virtual Constraints get_e() = 0;

  virtual std::vector<IntVar *> get_int_var() = 0;
};


{{INSTRUCTION_CLASSES}}

class Slice : public Instruction {
private:
  // [(start, end), ...] for each dimension of the slice
  std::vector<std::pair<int64, int64>> slices;
  std::vector<int64> rhs_sizes; // size of each dimension of rhs tensor

  std::vector<IntVar *> rhs_addrs;

  std::vector<std::pair<IntExpr *, IntExpr *>> get_h0() {
    std::vector<std::pair<IntExpr *, IntExpr *>> h;
    for (size_t i = 0; i < slices.size(); i++) {
      IntExpr *lhs_addr_start = MAKE_SUM(rhs_addrs[i], slices[i].first);
      IntExpr *lhs_addr_end = MAKE_SUM(rhs_addrs[i], slices[i].second);
      h.push_back({lhs_addr_start, lhs_addr_end});
    }
    return h;
  }

  std::vector<std::pair<IntExpr *, IntExpr *>> get_h1() {
    std::vector<std::pair<IntExpr *, IntExpr *>> h;
    for (size_t i = 0; i < rhs_sizes.size(); i++) {
      IntExpr *rhs_addr_start = rhs_addrs[i];
      IntExpr *rhs_addr_end = MAKE_SUM(rhs_addrs[i], rhs_sizes[i]);
      h.push_back({rhs_addr_start, rhs_addr_end});
    }
    return h;
  }

public:
  Slice(Tensor *y, std::vector<Tensor *> X,
        const std::vector<std::pair<int64, int64>> &slices)
      : Instruction("slice", y, X), slices(slices) {
    assert(y != nullptr);

    assert(X.size() == 1);

    assert(X[0] != nullptr);
    assert(X[0]->get_storage() == y->get_storage());

    rhs_sizes = X[0]->get_sizes();
    for (size_t i = 0; i < rhs_sizes.size(); i++) {
      rhs_addrs.push_back(MAKE_INT_VAR(INT64_MIN, INT64_MAX));
    }

    assert(slices.size() == y->get_sizes().size());
    for (size_t i = 0; i < rhs_sizes.size(); i++) {
      assert(slices[i].first >= 0 && slices[i].second <= rhs_sizes[i] &&
             slices[i].first < slices[i].second);
      assert(y->get_sizes()[i] == slices[i].second - slices[i].first);
    }
  }

  std::string str() override {
    for (auto *var : rhs_addrs) {
      assert(var->Bound());
    }
    return "";
  }

  std::vector<bool> get_rhs_inplace() override { return {true}; }

  std::vector<std::pair<IntExpr *, IntExpr *>> get_h(int num) override {
    switch (num) {
    case 0:
      return get_h0();
    case 1:
      return get_h1();
    default:
      assert(false && "Too many indices for get_h");
    }
  }

  Constraints get_e() override {
    Constraints constraints;
    return constraints;
  }

  std::vector<IntVar *> get_int_var() override { return rhs_addrs; }
};

class Concat : public Instruction {
private:
  int64 dimension;
  std::vector<int64> rhs1_sizes; // size of each dimension of rhs1 tensor
  std::vector<int64> rhs2_sizes; // size of each dimension of rhs2 tensor

  std::vector<IntVar *> rhs1_addrs;

  std::vector<std::pair<IntExpr *, IntExpr *>> get_h0() {
    std::vector<std::pair<IntExpr *, IntExpr *>> h;
    for (size_t i = 0; i < rhs1_sizes.size(); i++) {
      if (i == (size_t)dimension) {
        IntExpr *lhs_addr_start = rhs1_addrs[i];
        IntExpr *lhs_addr_end =
            MAKE_SUM(rhs1_addrs[i], rhs1_sizes[i] + rhs2_sizes[i]);
        h.push_back({lhs_addr_start, lhs_addr_end});
      } else {
        IntExpr *lhs_addr_start = rhs1_addrs[i];
        IntExpr *lhs_addr_end = MAKE_SUM(rhs1_addrs[i], rhs1_sizes[i]);
        h.push_back({lhs_addr_start, lhs_addr_end});
      }
    }
    return h;
  }

  std::vector<std::pair<IntExpr *, IntExpr *>> get_h1() {
    std::vector<std::pair<IntExpr *, IntExpr *>> h;
    for (size_t i = 0; i < rhs1_sizes.size(); i++) {
      IntExpr *rhs1_addr_start = rhs1_addrs[i];
      IntExpr *rhs1_addr_end = MAKE_SUM(rhs1_addrs[i], rhs1_sizes[i]);
      h.push_back({rhs1_addr_start, rhs1_addr_end});
    }
    return h;
  }

  std::vector<std::pair<IntExpr *, IntExpr *>> get_h2() {
    std::vector<std::pair<IntExpr *, IntExpr *>> h;
    for (size_t i = 0; i < rhs2_sizes.size(); i++) {
      if (i == (size_t)dimension) {
        IntExpr *rhs2_addr_start = MAKE_SUM(rhs1_addrs[i], rhs1_sizes[i]);
        IntExpr *rhs2_addr_end =
            MAKE_SUM(rhs1_addrs[i], rhs1_sizes[i] + rhs2_sizes[i]);
        h.push_back({rhs2_addr_start, rhs2_addr_end});
      } else {
        IntExpr *rhs2_addr_start = rhs1_addrs[i];
        IntExpr *rhs2_addr_end = MAKE_SUM(rhs1_addrs[i], rhs2_sizes[i]);
        h.push_back({rhs2_addr_start, rhs2_addr_end});
      }
    }
    return h;
  }

public:
  Concat(Tensor *y, std::vector<Tensor *> X, int64 dimension)
      : Instruction("concat", y, X), dimension(dimension) {
    assert(y != nullptr);

    assert(X.size() == 2);

    assert(X[0] != nullptr);
    assert(X[0]->get_storage() == y->get_storage());

    assert(X[1] != nullptr);
    assert(X[1]->get_storage() == y->get_storage());

    rhs1_sizes = X[0]->get_sizes();
    rhs2_sizes = X[1]->get_sizes();
    for (size_t i = 0; i < rhs1_sizes.size(); i++) {
      rhs1_addrs.push_back(MAKE_INT_VAR(INT64_MIN, INT64_MAX));
    }

    assert(rhs1_sizes.size() == y->get_sizes().size());
    assert(rhs2_sizes.size() == y->get_sizes().size());
    for (size_t i = 0; i < rhs1_sizes.size(); i++) {
      if (i == (size_t)dimension) {
        assert(y->get_sizes()[i] == rhs1_sizes[i] + rhs2_sizes[i]);
      } else {
        assert(y->get_sizes()[i] == rhs1_sizes[i]);
        assert(rhs1_sizes[i] == rhs2_sizes[i]);
      }
    }
  }

  std::string str() override {
    for (auto *var : rhs1_addrs) {
      assert(var->Bound());
    }
    return "";
  }

  std::vector<bool> get_rhs_inplace() override { return {true, true}; }

  std::vector<std::pair<IntExpr *, IntExpr *>> get_h(int num) override {
    switch (num) {
    case 0:
      return get_h0();
    case 1:
      return get_h1();
    case 2:
      return get_h2();
    default:
      assert(false && "Too many indices for get_h");
    }
  }

  Constraints get_e() override {
    Constraints constraints;
    return constraints;
  }

  std::vector<IntVar *> get_int_var() override { return rhs1_addrs; }
};

#endif // INSTRUCTIONS_H
