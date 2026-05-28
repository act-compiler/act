#ifndef SCHEDULING_H
#define SCHEDULING_H

#include <string>
#include <unordered_map>
#include <vector>

#include "instructions.h"
#include "parser.h"
#include "storage.h"
#include "tensor.h"

using EsunCache =
    std::unordered_map<Instruction *, std::unordered_map<std::string, int64>>;

namespace scheduler {

std::vector<Instruction *>
topological(const std::vector<Instruction *> &instructions, Instruction *root,
            const std::vector<std::string> &buffer_name);

} // namespace scheduler

/* Helper functions for scheduling */

int64 mem(Tensor *t, const std::string &d);
int64 esun(Instruction *instr, const std::string &d, EsunCache &esun_cache);

double cost(Instruction *instr, const EsunCache &esun_cache,
            const std::vector<std::string> &buffers);
void dfs(Instruction *instr, std::unordered_map<Instruction *, bool> &visited,
         std::vector<Instruction *> &order,
         const std::unordered_map<Instruction *, double> &cost_map);

#endif // SCHEDULING_H
