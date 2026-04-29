#ifndef COMMON_H
#define COMMON_H

#include "ortools/constraint_solver/constraint_solver.h"

typedef int64_t int64;
typedef uint64_t uint64;

extern operations_research::Solver solver;

#define MAKE_INT_VAR(a, b) (solver.MakeIntVar((a), (b)))
#define MAKE_INT_KNOWN(a) (solver.MakeIntVar((a), (a)))

#define MAKE_SUM(a, b) (solver.MakeSum((a), (b)))
#define MAKE_PROD(a, b) (solver.MakeProd((a), (b)))

#define MAKE_EQUALITY(a, b) (solver.MakeEquality((a), (b)))
#define MAKE_GREATER_OR_EQUAL(a, b) (solver.MakeGreaterOrEqual((a), (b)))
#define MAKE_LESS_OR_EQUAL(a, b) (solver.MakeLessOrEqual((a), (b)))

#endif // COMMON_H
