# Failure policy mapped onto the classes

Owner decision 40: nothing fails without a genuine cause.
This file maps each tool's allowed stops onto the classes of `scheme.md`; any other ending is a recorded outcome or a bug.

## Allowed stops per tool

| Tool | Stops with | Class | The error names |
| --- | --- | --- | --- |
| Front end | A problem in the ISA specification | `R` | The object, the rule broken, the found value; what the architect edits |
| Generator, generation time | An ISA specification with no instruction left after exclusion, or an unknown key in its own metadata namespace | `U` | The generator's unsupported row or the key; what the architect removes or changes |
| Generator, generation time | Not a stop: an instruction its target cannot model is excluded | `U` at severity `warning` | The instruction, the construct, and the author's line; printed at generation and stored in the generated tool |
| Generated oracle | Invalid assembly code or binary input | `K` (scope `O`) | The assembly line or word offset, and what the user corrects |
| Generated oracle | A program that reaches a rule violation (S1.3) | `V` | The instruction, the iteration, the violated rule |
| Generated backend | An invalid kernel, or a kernel outside the input specification | `K` (scope `B`) | The kernel line and the construct the user changes |
| Generated backend | No candidate implements a valid kernel: no covering, or every candidate dropped | `K` (scope `B`) | The kernel, the lowest uncovered operation and the stop cause, or the drop reasons by code; as a hint outside the match key, each excluded instruction whose tensor operators include that operation |
| Any tool | Files, flags, builds | `X` | The path or flag |

- A generator or the oracle never stops on `I`: after the front end ships its checks, an `R` condition found late is a recorded finding of the tool (D5), not a user stop.
- An oracle that runs a valid program stops on nothing else; tensor data never faults, so fixed-value cases run to the end.

## Candidate search in the backend

- A candidate failure is a drop: the candidate id, the `code` of the class that failed it, and the reason are recorded, and the search continues.
- An internal error inside candidate search (allocator, extractor, writer) is a drop with its `I` code, not a stop and not a panic exit.
- Drops go to stderr with a count; the first-drop summary of `pipeline.rs:208-218` and the per-drop line of `pipeline.rs:265-266` become one record per drop in a drop log under `--log`.
- A drop never changes an answer: the emitted assembly is one a surviving candidate produced, and the wall-clock bound of decision 27 only ends the search.
- When every candidate drops, or the bound ends the search with no candidate, the run ends once with `KB041` listing the first three drop codes and the count; the full list is in the drop log.
- An `I`-coded drop is a tool bug: users see it as a recorded reason, and the test suite fails on one outside an allowlist (T36).
- A drop reason is not a new class and not a diagnostic that stops; it reuses the code of the failure, so the registry has one row per cause.

## The input kernel class

- The kernel is the StableHLO, XLA HLO, torch-mlir, or torch.fx input; an invalid one or one outside the input specification is class `K`, scope `B`.
- A `K` message states the line (`kernel.hlo:12`), the construct, and what the user changes; it is never raised for a candidate failure.
- `scheme.md` Input class also covers assembly code and binary input given to an oracle (scope `O`).
- The causes keep apart: invalid kernel (`KB001` to `KB040`), no candidate (`KB041`, with the excluded-instruction hint), invalid assembly or binary (`KO021` to `KO040`).

## Layer codes

- The layers add front-end `R` rows (`registry-seed.md`): a type out of range or `u64`, a type key naming no attribute, an encoding coverage or clash defect, a width mismatch, and a refused derived link.
- A value outside its type is `VF001`, S1.3.10 generalized; a word matching no instruction is a `KO` decode error.
- A declared range the allocator cannot hold excludes its instruction with a `U` reason at generation, never a clamp.
- Generator metadata raises no front-end error; a generator rejects an unknown key in its own namespace with a `U` coded error of its own (`UB020` for the backend) and ignores other namespaces.
- The oracle runs no check after the front end's `check` and excludes nothing (owner decision 42).

## Open

- `KB041` as `K` rather than `U`: default is `K`, because the kernel is valid and the ISA specification is what the architect edits.
