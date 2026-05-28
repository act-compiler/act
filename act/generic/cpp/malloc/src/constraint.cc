#include "constraint.h"

namespace constraint {

void pii_node_constraints(const std::vector<Instruction *> &instructions) {
  for (const auto &instr : instructions) {
    // Instruction constraints: 𝑒(𝛼,𝛽)

    Constraints constraints = instr->get_e();

    for (auto *constraint : constraints) {
      solver.AddConstraint(constraint);
    }

    // LHS Tensor addresses: (𝐼𝑠(𝑦),𝐼𝑒(𝑦)) = ℎ(0)(𝛽)

    auto lhs_tensor = instr->get_lhs();
    auto lhs_offsets = lhs_tensor->get_offsets();
    auto lhs_size = lhs_tensor->get_sizes();
    auto h0 = instr->get_h(0);

    for (size_t i = 0; i < lhs_offsets.size(); ++i) {
      solver.AddConstraint(MAKE_EQUALITY(h0[i].first, lhs_offsets[i]));
      solver.AddConstraint(
          MAKE_EQUALITY(h0[i].second, MAKE_SUM(lhs_offsets[i], lhs_size[i])));
    }
  }
}

void def_use_constraints(const std::vector<Instruction *> &instructions) {
  for (const auto &instr : instructions) {
    // RHS Tensor addresses: (𝐼𝑠(𝑥𝑖), 𝐼𝑒(𝑥𝑖)) = ℎ(𝑖)(𝛽)

    auto rhs_tensors = instr->get_rhs();
    for (size_t i = 0; i < rhs_tensors.size(); ++i) {
      auto offsets = rhs_tensors[i]->get_offsets();
      auto sizes = rhs_tensors[i]->get_sizes();
      auto h_i = instr->get_h(i + 1);

      for (size_t j = 0; j < offsets.size(); ++j) {
        solver.AddConstraint(MAKE_EQUALITY(h_i[j].first, offsets[j]));
        solver.AddConstraint(
            MAKE_EQUALITY(h_i[j].second, MAKE_SUM(offsets[j], sizes[j])));
      }
    }
  }
}

void initial_constraints(
    const std::unordered_map<Tensor *, int64> &known_offsets) {
  for (const auto &pair : known_offsets) {
    // Input & Output Tensor addresses: 𝐼𝑠(𝑥𝑖) = 𝑐𝑜𝑛𝑠𝑡

    Tensor *tensor = pair.first;
    int64 offset = pair.second;
    auto offsets = tensor->get_offsets();
    assert(offsets.size() == 1); // Expected to be in "HBM"

    solver.AddConstraint(MAKE_EQUALITY(offsets[0], offset));
  }
}

void overlap_constraints(
    const std::vector<Instruction *> &instructions,
    const std::unordered_map<std::string, Tensor *> &tensor_map,
    std::vector<IntVar *> &new_vars) {

  // Compute live ranges for all tensors
  std::unordered_map<Tensor *, LiveRange> live_ranges;
  live_ranges.reserve(tensor_map.size());
  for (auto &pair : tensor_map) {
    Tensor *t = pair.second;
    live_ranges[t] = compute_live_range(t, instructions);
  }

  // Build interference graph
  InterferenceGraph interference_graph = build_interference_graph(live_ranges);

  // Add non-overlapping constraints for each edge in the interference graph
  for (const auto &edge : interference_graph) {
    Tensor *t1 = edge.first;
    Tensor *t2 = edge.second;
    auto offsets1 = t1->get_offsets();
    auto sizes1 = t1->get_sizes();
    auto offsets2 = t2->get_offsets();
    auto sizes2 = t2->get_sizes();

    size_t dims = offsets1.size();
    assert(dims == offsets2.size());
    assert(dims == sizes1.size());
    assert(dims == sizes2.size());

    // Non-overlapping constraint: (𝐼𝑒(𝑥1) ≤ 𝐼𝑠(𝑥2)) ∨ (𝐼𝑒(𝑥2) ≤ 𝐼𝑠(𝑥1))

    // Non-overlapping in ANY dimension suffices (disjunction across dims).
    std::vector<IntVar *> dim_non_overlap;
    for (size_t i = 0; i < dims; ++i) {
      auto end1 = MAKE_SUM(offsets1[i], sizes1[i]);
      auto end2 = MAKE_SUM(offsets2[i], sizes2[i]);

      IntVar *b1 = solver.MakeIsLessOrEqualVar(end1, offsets2[i]);
      IntVar *b2 = solver.MakeIsLessOrEqualVar(end2, offsets1[i]);
      // non_overlap_i = (b1 OR b2)
      IntVar *non_overlap_i = solver.MakeIsGreaterOrEqualVar(
          solver.MakeSum(std::vector<IntVar *>{b1, b2}), MAKE_INT_KNOWN(1));
      dim_non_overlap.push_back(non_overlap_i);
      new_vars.push_back(b1);
      new_vars.push_back(b2);
      new_vars.push_back(non_overlap_i);
    }
    // Require non-overlap in at least one dimension
    solver.AddConstraint(solver.MakeGreaterOrEqual(
        solver.MakeSum(dim_non_overlap), int64_t(1)));
  }
}

} // namespace constraint

// Note that instructions are ordered from 1 to N
// where 1 is the first instruction to be executed
// and N is the last instruction to be executed
// 0 refers to the time before any instruction is executed, i.e., initial state
LiveRange compute_live_range(Tensor *a,
                             const std::vector<Instruction *> &instructions) {
  // Helper lambdas
  // (i) panic with error message
  auto panic = [&](const std::string &msg) -> LiveRange {
    std::cerr << "live_range error: " << msg << std::endl;
    assert(false && "unreachable");
    return {0, 0};
  };

  // If INPUT, then live range is [0, N]
  if (a->type == Tensor::node_type::INPUT) {
    return {0, instructions.size()};
  }

  // If CONSTANT, then live range is [0, b]
  // where b is the last instruction that uses it
  if (a->type == Tensor::node_type::CONSTANT) {
    auto uses = a->get_used_by();
    if (uses.empty())
      return panic("constant tensor has no uses");

    size_t last_use = 0;
    for (const auto &instr : uses) {
      auto it = std::find(instructions.begin(), instructions.end(), instr);
      if (it != instructions.end()) {
        size_t index = std::distance(instructions.begin(), it);
        if (index > last_use) {
          last_use = index;
        }
      } else {
        return panic("instruction using constant tensor not found in list.");
      }
    }

    return {0, last_use + 1};
  }

  // If INTERMEDIATE, then live range is [a, b]
  // where a is the instruction that defines it
  // and b is the last instruction that uses it
  if (a->type == Tensor::node_type::INTERMEDIATE) {
    auto def = a->get_defined_by();
    if (def == nullptr)
      return panic("tensor has no defining instruction.");

    auto uses = a->get_used_by();
    if (uses.empty())
      return panic("tensor has no uses");

    size_t first_def = 0;
    auto it = std::find(instructions.begin(), instructions.end(), def);
    if (it != instructions.end()) {
      first_def = std::distance(instructions.begin(), it);
    } else {
      return panic("instruction defining tensor not found in list.");
    }

    size_t last_use = first_def;
    for (const auto &instr : uses) {
      auto it = std::find(instructions.begin(), instructions.end(), instr);
      if (it != instructions.end()) {
        size_t index = std::distance(instructions.begin(), it);
        if (index > last_use) {
          last_use = index;
        }
      } else {
        return panic("instruction using tensor not found in list.");
      }
    }

    return {first_def + 1, last_use + 1};
  }

  // If OUTPUT, then live range is [N, N]
  if (a->type == Tensor::node_type::OUTPUT) {
    return {instructions.size(), instructions.size()};
  }

  return panic("unknown tensor type.");
}

InterferenceGraph build_interference_graph(
    const std::unordered_map<Tensor *, LiveRange> &live_ranges) {
  std::vector<std::pair<Tensor *, Tensor *>> interference_graph;

  for (auto it1 = live_ranges.begin(); it1 != live_ranges.end(); ++it1) {
    for (auto it2 = std::next(it1); it2 != live_ranges.end(); ++it2) {
      Tensor *t1 = it1->first;
      Tensor *t2 = it2->first;

      // Check if they are on the same storage
      if (t1->get_storage() != t2->get_storage()) {
        continue;
      }

      LiveRange lr1 = it1->second;
      LiveRange lr2 = it2->second;

      // Check if live ranges overlap. Treat touching intervals as
      // non-overlapping (i.e. [a,b] and [b,c] do NOT interfere).
      if (!(lr1.second <= lr2.first || lr2.second <= lr1.first)) {
        // Check if they are allowed to overlap
        Instruction *def1 = t1->get_defined_by();
        Instruction *def2 = t2->get_defined_by();
        if (def1 != nullptr && def2 != nullptr) {
          auto rhs1 = def1->get_rhs();
          auto rhs2 = def2->get_rhs();
          auto inplace1 = def1->get_rhs_inplace();
          auto inplace2 = def2->get_rhs_inplace();

          bool allowed_to_overlap = false;
          for (size_t i = 0; i < rhs1.size(); ++i) {
            if (inplace1[i] && rhs1[i] == t2) {
              allowed_to_overlap = true;
              break;
            }
          }
          for (size_t i = 0; i < rhs2.size(); ++i) {
            if (inplace2[i] && rhs2[i] == t1) {
              allowed_to_overlap = true;
              break;
            }
          }
          if (allowed_to_overlap) {
            continue;
          }
        }

        // Add edge to interference graph
        interference_graph.emplace_back(t1, t2);
      }
    }
  }

  return interference_graph;
}
