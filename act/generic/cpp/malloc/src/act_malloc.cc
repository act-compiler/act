#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "act_malloc.h"

extern "C" int act_malloc(char *pii_path, char *metadata_path, char *asm_path) {
  std::ifstream pii_file(pii_path);
  if (!pii_file) {
    std::cerr << "Failed to open file: " << pii_path << std::endl;
    return -1;
  }

  std::unordered_map<std::string, Tensor *> tensor_map;
  std::unordered_map<Tensor *, int64> known_offsets;
  std::unordered_map<Tensor *, std::string> known_constants;
  std::vector<Instruction *> instructions;

  std::string line;
  while (std::getline(pii_file, line)) {
    // trim
    line.erase(0, line.find_first_not_of(" \t\r\n"));
    line.erase(line.find_last_not_of(" \t\r\n") + 1);
    if (line.empty() || line[0] == '#')
      continue;

    Instruction *instr = parser::create_instruction(
        parser::parse_line(line), tensor_map, known_offsets, known_constants);
    if (instr)
      instructions.push_back(instr);
  }

  if (instructions.empty()) {
    std::cout << "No ISA instructions (all nodes are constants/inputs)." << std::endl;
    std::cout << "No solution found." << std::endl;
    std::cout << "No ASM candidate generated." << std::endl << std::endl;
    return 0;
  }

  Instruction *root = instructions.back();
  root->get_lhs()->type = Tensor::node_type::OUTPUT;

  // Validate that no tensor is used after being consumed by an in-place operation
  std::cout << "Starting Phase 2 Module 3.5: PII Graph Validation..." << std::endl;
  for (auto *instr : instructions) {
    auto inplace = instr->get_rhs_inplace();
    auto rhs = instr->get_rhs();
    for (size_t i = 0; i < inplace.size() && i < rhs.size(); ++i) {
      if (inplace[i]) {
        // This tensor is consumed in-place - check if it's used after this instruction
        Tensor *consumed_tensor = rhs[i];
        for (auto *user : consumed_tensor->get_used_by()) {
          if (user != instr) {
            std::cerr << "Error: Invalid PII graph - tensor " << consumed_tensor->get_name()
                     << " is used by " << user->get_op_name() << " after being consumed in-place by "
                     << instr->get_op_name() << std::endl;
            return 0;
          }
        }
      }
    }
  }

  std::cout << "Starting Phase 2 Module 4: Topological Ordering Generator..."
            << std::endl;

  auto m4_start = std::chrono::high_resolution_clock::now();
  auto scheduled =
      scheduler::topological(instructions, root, {{{BUFFER_NAMES}}});
  auto m4_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now() - m4_start).count();
  std::cout << "[STATS] module=M4_TopoSort time=" << m4_elapsed << "us" << std::endl;

  if (scheduled.size() != instructions.size()) {
    std::cerr << "Error: scheduled instructions size (" << scheduled.size()
              << ") does not match original instructions size ("
              << instructions.size() << ")." << std::endl;
    return -1;
  }

  std::cout << "Starting Phase 2 Module 5: Constraint Satisfaction Problem "
               "Generator..."
            << std::endl;

  auto m5_start = std::chrono::high_resolution_clock::now();
  constraint::pii_node_constraints(scheduled);
  constraint::def_use_constraints(scheduled);
  constraint::initial_constraints(known_offsets);
  std::vector<operations_research::IntVar *> new_vars;
  constraint::overlap_constraints(scheduled, tensor_map, new_vars);

  std::vector<operations_research::IntVar *> all_vars;
  for (auto &buffer : g_storage) {
    auto dims = buffer.second->get_addressing_dims();
    for (size_t i = 0; i < dims; i++) {
      all_vars.push_back(buffer.second->get_capacity(i));
    }
  }
  for (auto &pair : tensor_map) {
    auto *t = pair.second;
    auto offset_vars = t->get_offsets();
    all_vars.insert(all_vars.end(), offset_vars.begin(), offset_vars.end());
  }
  for (auto *instr : scheduled) {
    auto vars = instr->get_int_var();
    all_vars.insert(all_vars.end(), vars.begin(), vars.end());
  }
  all_vars.insert(all_vars.end(), new_vars.begin(), new_vars.end());

  // Heuristic: order variables by domain width (smallest first) to prioritize
  // tightly constrained variables first
  std::vector<operations_research::IntVar *> ordered_vars = all_vars;
  std::sort(ordered_vars.begin(), ordered_vars.end(),
            [](operations_research::IntVar *a, operations_research::IntVar *b) {
              // domain width may overflow if computed as unsigned; use 128-bit
              // safe path
              int64 w_a = a->Max() - a->Min();
              int64 w_b = b->Max() - b->Min();
              if (w_a != w_b)
                return w_a < w_b;
              return a->Min() < b->Min();
            });
  auto m5_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now() - m5_start).count();
  std::cout << "[STATS] module=M5_CSPGen time=" << m5_elapsed << "us" << std::endl;

  std::cout << "Starting Phase 2 Module X: Google OR-Tools CP-SAT solver..."
            << std::endl;

  auto mx_start = std::chrono::high_resolution_clock::now();
  auto decision_builder = solver.MakePhase(
      ordered_vars, operations_research::Solver::CHOOSE_FIRST_UNBOUND,
      operations_research::Solver::ASSIGN_MIN_VALUE);

  // Add a conservative time limit (0.5 seconds) to prevent long hangs
  auto time_limit = solver.MakeTimeLimit(absl::Milliseconds(500));
  solver.NewSearch(decision_builder, time_limit);
  if (solver.NextSolution()) {
    auto mx_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - mx_start).count();
    std::cout << "[STATS] module=MX_Solver time=" << mx_elapsed << "us" << std::endl;
    std::cout << "Found a solution!" << std::endl;

    std::cout << "Starting Phase 2 Module 6: Code Emitter..." << std::endl;

    auto m6_start = std::chrono::high_resolution_clock::now();
    std::ifstream metadata_file(metadata_path);
    if (!metadata_file) {
      std::cerr << "Failed to open file: " << metadata_path << std::endl;
      solver.EndSearch();
      return -1;
    }
    emitter::MetaData metadata(metadata_file, tensor_map, known_constants);

    if (emitter::assembly_dump(asm_path, scheduled, metadata)) {
      auto m6_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - m6_start).count();
      std::cout << "[STATS] module=M6_Emitter time=" << m6_elapsed << "us" << std::endl;
      std::cout << "New ASM candidate generated: " << asm_path << std::endl
                << std::endl;
      solver.EndSearch();
      return 1;
    }

    std::cout << "No ASM candidate generated." << std::endl << std::endl;
    solver.EndSearch();
    return 0;

  } else {
    auto mx_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - mx_start).count();
    std::cout << "[STATS] module=MX_Solver time=" << mx_elapsed << "us" << std::endl;
    std::cout << "No solution found." << std::endl;

    std::cout << "Starting Phase 2 Module 6: Code Emitter..." << std::endl;
    std::cout << "No ASM candidate generated." << std::endl << std::endl;

    solver.EndSearch();
    return 0;
  }
}
