#ifndef CONSTRAINT_H
#define CONSTRAINT_H

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "instructions.h"
#include "solver.h"
#include "tensor.h"

using LiveRange = std::pair<size_t, size_t>;
using InterferenceGraph = std::vector<std::pair<Tensor *, Tensor *>>;

namespace constraint {

// CSP1: 𝑒(𝛼,𝛽) ∧ (𝐼𝑠(𝑦),𝐼𝑒(𝑦)) = ℎ(0)(𝛽)
void pii_node_constraints(const std::vector<Instruction *> &instructions);

// CSP2: (𝐼𝑠(𝑥𝑖), 𝐼𝑒(𝑥𝑖)) = ℎ(𝑖)(𝛽)
void def_use_constraints(const std::vector<Instruction *> &instructions);

// CSP3: 𝐼𝑠(𝑥𝑖) = 𝑐𝑜𝑛𝑠𝑡
void initial_constraints(
    const std::unordered_map<Tensor *, int64> &known_offsets);

// CSP4: NON_OVERLAP(ℎ(0), ℎ'(0))
void overlap_constraints(
    const std::vector<Instruction *> &instructions,
    const std::unordered_map<std::string, Tensor *> &tensor_map,
    std::vector<IntVar *> &new_vars);

} // namespace constraint

/* Helper functions for CSP4 */

LiveRange compute_live_range(Tensor *a,
                             const std::vector<Instruction *> &instructions);
InterferenceGraph build_interference_graph(
    const std::unordered_map<Tensor *, LiveRange> &live_ranges);

#endif // CONSTRAINT_H
