# Open questions with draft answers

Engineering questions raised by the tracks and reviewers, each with a draft answer.
None of these is decided: each goes to a mixed-model debate panel, which settles it on evidence.
Only the owner decisions are binding.

## Language and semantics

| ID | Question | Ruling | Reason |
| -- | -------- | ------ | ------ |
| R1 | Name of the user's file | "ISA description" | The owner's papers use it; "spec" means the language document |
| R2 | Rename internal names such as `computation_attr` | Yes, in v3, per the reconciled glossary | Breaking renames are batched into v3 |
| R3 | Initial value of `reduce_<reducer>` | The reducer's identity; integer max and min start at the type's lowest and highest value | One simple norm, matches the oracle today |
| R4 | Out-of-bounds region | Defined error at issue, detected by the oracle; the backend may assume it never happens | Owner decision 16 |
| R5 | Which output an in-place input aliases | An input and an output alias exactly when their region texts are identical; `set_inplace` is removed in v3 | Under owner decision 17 the mark is derivable, so it is redundant |
| R6 | Reference operation version | The StableHLO version bundled with the oracle's pinned jax; the spec states each TAIDL difference | One pin for spec and oracle |
| R7 | State read order | Starts, sizes, and constraints all read the state after the instruction's updates | One state view per instruction; starts already read it so (instruction.py:216-218) |
| R8 | Float `dot` and `reduce` accumulation | Accumulate in the result element type; the result is within a per-word tolerance of the exact result rounded to that type; wider accumulation is written with explicit `convert` lines | A defined norm that XLA-backed oracles can meet without fixing a summation order |
| R9 | `ROOT` on a non-last line | The spec allows it; the backend coverage row is partial until fixed | The spec wins over a generator |
| R10 | A violated constraint | Defined error checked by the oracle; the backend may assume it holds | Owner decision 16 |
| R11 | Instruction without semantics | Legal only without regions (a state-only instruction such as a configuration); with regions it is rejected in v3; the backend checks state-only instructions instead of dropping them | Configuration instructions need it; silent drops are defects |
| R12 | Contents of storage that no output region writes | Unchanged; every buffer starts zero-filled | One simple norm, no undefined read |
| R13 | Multi-core and tile-array concurrency | Outside v3; a Q2 design study; implementation after v3 | Large, and v3 must ship in two quarters |
| R14 | Typed state kinds for descriptors and async groups (P11) | In v3 | Hopper TMA and Trainium DMA descriptors both need it, and TMA is in scope |
| R15 | Missing Ascend case study | Added at the start of C12 | Ten studies already cover the gap matrix |
| R16 | Run the case-study sketches through the front end | Yes, as part of C11 | A sketch that does not parse is weak evidence |
| R55 | Name of a defined run-time error | A "fault": it writes no region and no state, and the generator reports it with the clause id | One word for every case of owner decision 16 |
| R56 | Integer arithmetic in expressions | Exact; a final value outside signed 64 bits is a fault | Simpler than checking every intermediate value |
| R57 | Constant literals | The value must fit the element type; floats round once, to nearest-even; out-of-range literals are rejected, and out-of-range backtick expressions fault | The generators disagree today (spec_checks.py:94-107, utils.py:398-426) |
| R58 | `set_apart` and `set_no_partial_overlap` | Removed in v3; the same condition is written as a constraint | No example or tutorial calls them |
| R59 | Language extensions | An extension stays in v3 only if at least two case studies need it; others are deferred | The owner's rule: abstraction at the second use site |
| R60 | `cholesky`, `triangular_solve`, `batch_norm_inference` | Removed in v3; reintroduced at the second real ISA description that needs them | No case-study accelerator has them; they alone need three attribute forms |
| R61 | Spec pages that copy normative text | The grammar page links the .g4 files instead of copying rules; the reserved-names table is cut | One source of truth |
| R62 | Shape and type rules | One M-sized item: move the oracle's shape module into taidl and derive the S8 table from it (merges L3, L12, O6, O7) | One copy of the rules, used by every generator |
| R63 | Tutorials as verification | No fail point relies on a tutorial run until A10 migrates the tutorials | They use syntax the current grammar rejects (IDLV2.g4:6) |

## Backend

| ID | Question | Ruling | Reason |
| -- | -------- | ------ | ------ |
| R17 | Test gate for structural steps | First add a deterministic search mode bounded by iteration and candidate counts, and gate on identical chosen assembly and candidate logs | Wall-clock search limits make goldens unreliable |
| R18 | Order of the e-graph work | Type the e-graph payloads, then shape inference in egg `Analysis::make` (B7) and no per-extraction clone (B9), all in Q1 before B6 | Avoids regenerating code that is deleted next |
| R19 | Rewrite conditions as code or data | Data, with one generic applier, decided now; B6 and B12 are done once in that form | Avoids rewriting the same generator three times |
| R20 | Two extractors | Merge into one cost-guided extractor that keeps the best few candidates | The cost is an instruction count, so the extractor can rank directly |
| R21 | In-place aliasing in the allocator | An equality constraint in the solver | Owner decision 17; today overlap is only permitted |
| R22 | Overlapping rewrite rules | Rejected at generation; explicit priority only where declared | No resolution by e-node order |
| R23 | egg or egglog | Keep egg; borrow rule sets, explanations, and stop reasons | Lower risk, same benefits |
| R24 | `clang-format` at generation time | Not required; rustfmt formats the Rust output | Fewer tools in the generation path |
| R25 | Region element-count checks | Move to the front end (L5); the backend keeps its literal check until then | Checks of meaning belong to the front end |
| R26 | `TensorInfo` and `Buffer` identifiers | Rename `TensorInfo`; keep `Buffer` | "Tensor buffer" is the papers' term |
| R27 | State registers and `@s.*` in the backend | Implement in v3 (Q2), after the front end's typed instruction model; the documented rejection stays until then | The owner named it as the known deficiency |

## Oracle

| ID | Question | Ruling | Reason |
| -- | -------- | ------ | ------ |
| R28 | Unroll every loop at trace time | Size-S spike in Q1 before O1; adopt if compile time stays within an agreed bound, then delete the loop algebra and O11 | Removes about 400 lines and makes decision 16 checkable per call |
| R29 | Differential reference | StableHLO reference interpreter, NumPy only for words it lacks | Size M instead of L; new dependency, owner approval first |
| R30 | Runtime code inside `API_FILE` | Move to `runtime/` as plain Python in Q1; O2 and O3 in Q1-Q2; fold O4 into them | Largest simplification first |
| R31 | Scope of O7 | Widen: region checks, constant ranges, and compare-type fit move to the front end | Checks of meaning belong to the front end |
| R32 | O12 | Drop the unused matmul-precision flag and avoid `jax.device_put` | Neither affects a module compiled from text |
| R33 | `docs/operations.md` | Fold into the coverage table and delete (O14) | It duplicates the language spec |
| R35 | Generated-code line splitter | Drop (O18) | rustfmt-style layout is not needed in Python output |
| R36 | Public `kernel(...)` entry name | Keep | "Kernel" is the glossary term for an oracle function |
| R37 | `generic/` directory name | Rename to `runtime/` | Says what it is |
| R38 | Signed 64-bit overflow in addressing | Defined error at the call, replacing the warning | Owner decision 16 |

## Shared code

| ID | Question | Ruling | Reason |
| -- | -------- | ------ | ------ |
| R39 | Shape module in the generated oracle | Import the front end's module at run time, pinned to the same taidl tag | One copy of the rules |
| R40 | Docker and CI script consolidation | Into act, after v3 | Not on the v3 critical path |
| R41 | Rank-0 shape | `[]` everywhere in v3 | One spelling |
| R42 | Front end to generator contract | A documented, typed Python object model, not a serialized format | No non-Python generator exists yet |

## Tests

| ID | Question | Ruling | Reason |
| -- | -------- | ------ | ------ |
| R43 | Test kit location | `taidl.testing` for the loader and tolerances; `act/testkit` for goldens and the oracle import | Each part lives with what it depends on |
| R44 | Clause ids in error messages | Yes in v3, as a suffix such as `[S9.4]`; L numbers the S9 rows | Ties every rejection to the spec |
| R45 | `hypothesis` for property tests | Test-only dependency | New dependency, owner approval first |
| R46 | Smoke test per generator repo | One `mini_trn` smoke test each | Fast local signal |
| R47 | Extra real ISA description | One GPU tensor-core description (mma/wgmma and TMA) | In scope per owner decision 15 |
| R48 | GPU goldens | Reference tolerance only | Bitwise GPU goldens are not portable |
| R49 | Differential test cadence | Nightly and on release | Too slow for every pull request |

## act umbrella

| ID | Question | Ruling | Reason |
| -- | -------- | ------ | ------ |
| R50 | Tutorials | All three are outdated; A10 refreshes one against current ACT, A11 merges the copies | Owner decision 6 |
| R51 | README trims (A1, A2) | Plan items only, executed with owner sign-off | Owner decision 5 |
| R52 | `act-ecosystem` distribution name | Keep; fix the README sentence within A1 | Renaming a published package costs users |
| R53 | Glossary terms `oracle`, `backend`, `kernel` | Added | Used across all repos |
| R54 | ASCII rule for archived tutorials | Exempt until A11 replaces them | Archived content is not edited |
