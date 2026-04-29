#include "solver.h"
#include "storage.h"

operations_research::Solver solver("sched_alloc");

std::map<std::string, Storage *> g_storage = {
    {"HBM", new Storage("HBM", {HBM_SIZE}, 0)},
{{GLOBALS_DATA_MODELS}}
};
