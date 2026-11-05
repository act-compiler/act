# Later, core tracks

Plan items scheduled for later in the core tracks, by track.

## language

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| L9 | Worked examples per section | L3 | M | -- |
| L12 | Shape and element type checker: merged into U4 | U4 | - | R62 |
| L13 | Integer literal lexing: no target after U14 | U14 | - | -- |
| L18 | Write masks on output regions: an idiom in L24 | L24, L3 | M | -- |
| L19 | Strided regions, first in line after v3 | L5 | M | -- |
| L20 | `convolution` with window attributes: an idiom in L24 | L3 | M | -- |
| L21 | Rounding and saturation attributes on `convert`: an idiom in L24 | L3 | S | -- |
| L22 | Sub-computations for `reduce`, `scatter`, `sort`, `map` | L3 | L | -- |
| L23 | Indirect regions: starts read from a buffer, joined with the data-dependent addressing study | L5 | M | -- |
| L27 | Apply the glossary to B, O, and A docs | L25 | M | -- |
| L28 | Banned-synonym lint over the D10 word list | D10, L26 | S | -- |
| L42 | Python surface by tracing: merged into U12 | U12 | - | -- |
| L44 | Converter script on the 2.x line and per-file `Model` equivalence: merged into U13 | U13 | - | -- |
| L45 | Remove the string surface, `taidl/antlr4/`, grammar tests, and the ANTLR dependency: merged into U14 | U14 | - | -- |

## backend

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| B22 | Merged into B21 | -- | - | -- |
| B23 | Cut: line agreement (D3) is U4's | -- | - | -- |
| B47 | StableHLO importer: C++ tool reading portable artifacts, printing today's kernel text, coded errors | B42, B43, U15 | M | -- |
| B48 | Dialect spike: generic instruction op against per-ISA ODS, per-ISA build cost measured | B13, U15 | S | -- |
| B49 | E-graph library core: arena, typed key, union-find, deferred rebuild, class data, read-only view, property tests | B26, B29, B31, B57, B7, B9 | L | -- |
| B50 | ISA dialect generator per B48: typed attributes, checked verifier, semantics region from O3 | B48, O3, U10 | M | -- |
| B51 | Matcher and rules: backtracking per operator, early preconditions, matched ids, applied memo, C++ rule emitter | B49, B6, B8 | L | -- |
| B52 | Dual-engine harness and swap through the B2 C ABI, three-tier gate, egg deleted | B10, B2, B3, B32, B34, B35, B51 | M | -- |
| B53 | Torch adapter: `ExportedProgram` to portable artifact; pairing check picks torch-mlir or the walker; strict | B42, B47 | M | -- |
| B54 | Typed import: importer builds library terms, chosen program as dialect ops, text bridge deleted | B47, B50, B52 | M | -- |
| B55 | XLA HLO text reader in C++, holding every message of `tests/test_reader.py` | B54 | M | -- |
| B56 | One C++ driver: importers, engine, extraction, allocator, cost, exclusion report; Rust crate deleted | B46, B54, B55 | L | -- |
| B57 | Class data record and `merge` change report for B49 | B7 | S | -- |

## oracle

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| O4 | One word table: folded into O3 | O3 | - | R30 |
| O34 | fsim step 1: C++ emitter per instruction from O3's generic emitter | O29, O3, U15 | L | -- |
| O35 | fsim step 2: `Pipeline::submit`, `end_trace`, `sync` replace `compile_and_load` and `execute` | O34 | M | -- |
| O36 | fsim steps 3 and 4: debug print out of `custom_call`, PJRT pinned by header and plugin hash; every golden and norm row holds | O35, U15 | M | -- |

## tests

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| T39 | StableHLO corpus exported once from jax, differential against the HLO path | B47, T13 | S | -- |
| T40 | Torch tests: eager PyTorch against ACT on the oracle per operation and dtype, coverage table | A29, T29 | M | -- |
| T41 | Platform gate: B3 on the C++ driver, oracle on fsim | B56, O36 | M | -- |

## act

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| A6 | Comment pass on docker/ scripts | -- | S | -- |
| A13 | Give every exercise of the new tutorial a runnable solution | A24 | S | -- |
| A18 | Add a scheduled compatibility build | A16 | M | -- |
| A29 | `torch.compile` and `torch.export` entry points, artifact-hash cache, one released torch per release, CI row per minor | A23, B53 | M | -- |
| A30 | Partition opt-in: supported subgraphs to ACT, every undelegated node printed | A29, B46 | M | -- |
