#ifndef ACT_MALLOC_H
#define ACT_MALLOC_H

#include "constraint.h"
#include "emitter.h"
#include "instructions.h"
#include "parser.h"
#include "scheduler.h"
#include "solver.h"
#include "storage.h"
#include "tensor.h"

extern "C" int act_malloc(char *pii_path, char *metadata_path, char *asm_path);

#endif // ACT_MALLOC_H
