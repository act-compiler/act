#include "scheduler.h"

namespace scheduler {

std::vector<Instruction *>
topological(const std::vector<Instruction *> &instructions, Instruction *root,
            const std::vector<std::string> &buffers) {
  EsunCache esun_cache;

  std::unordered_map<Instruction *, double> cost_map;
  for (auto *instr : instructions) {
    for (const auto &d : buffers) {
      esun(instr, d, esun_cache);
    }
    cost_map[instr] = cost(instr, esun_cache, buffers);
  }

  std::vector<Instruction *> topological_order;
  std::unordered_map<Instruction *, bool> visited;
  for (auto *instr : instructions)
    visited[instr] = false;

  dfs(root, visited, topological_order, cost_map);

  return topological_order;
}

} // namespace scheduler

int64 mem(Tensor *t, const std::string &d) {
  if (d != t->get_storage()->get_name()) {
    return 0;
  } else {
    int64 prod = 1;
    for (auto s : t->get_sizes()) {
      prod *= s;
    }
    return prod;
  }
}

int64 esun(Instruction *instr, const std::string &d, EsunCache &esun_cache) {
  if (esun_cache.count(instr) && esun_cache[instr].count(d)) {
    return esun_cache[instr][d];
  }

  Tensor *y = instr->get_lhs();
  int64 mem_y = mem(y, d);

  std::vector<Tensor *> xs = instr->get_rhs();
  std::vector<int64> mem_xs;
  std::vector<Instruction *> rhs_instrs;
  for (auto *t : xs) {
    mem_xs.push_back(mem(t, d));
    rhs_instrs.push_back(t->get_defined_by());
  }

  int64 mem_xs_sum = 0;
  for (auto m : mem_xs) {
    mem_xs_sum += m;
  }

  int64 tau_x_min = INT64_MAX;
  for (int i = 0; i < rhs_instrs.size(); ++i) {
    int esun_x_i;
    if (rhs_instrs[i]) {
      esun_x_i = esun(rhs_instrs[i], d, esun_cache);
    } else {
      esun_x_i = 0;
      // default if rhs operand's instruction is nullptr
      // (i.e., input or constant leaf)
    }
    int64 tau_x_i = esun_x_i + mem_xs_sum - mem_xs[i];
    tau_x_min = std::min(tau_x_min, tau_x_i);
  }

  int64 esun_final = std::max(mem_y, tau_x_min);
  esun_cache[instr][d] = esun_final;
  return esun_final;
}

double cost(Instruction *instr, const EsunCache &esun_cache,
            const std::vector<std::string> &buffers) {
  double cost = 0.0;
  for (const auto &d : buffers) {
    double c_d = double(esun_cache.at(instr).at(d)) / double(STORAGE_SIZE(d));
    cost = std::max(cost, c_d);
  }
  return cost;
}

void dfs(Instruction *instr, std::unordered_map<Instruction *, bool> &visited,
         std::vector<Instruction *> &order,
         const std::unordered_map<Instruction *, double> &cost_map) {
  if (visited[instr])
    return;
  visited[instr] = true;

  std::vector<Instruction *> next_candidates;
  for (auto *t : instr->get_rhs()) {
    Instruction *rhs_instr = t->get_defined_by();
    if (rhs_instr) {
      if (!visited[rhs_instr]) {
        next_candidates.push_back(rhs_instr);
      }
    } else {
      // default if rhs operand's instruction is nullptr
      // (i.e., input or constant leaf)
    }
  }

  // descending order of cost
  std::sort(next_candidates.begin(), next_candidates.end(),
            [&](Instruction *a, Instruction *b) {
              return cost_map.at(a) > cost_map.at(b);
            });

  for (auto *next : next_candidates) {
    dfs(next, visited, order, cost_map);
  }

  order.push_back(instr);
}
