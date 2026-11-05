# Waves 8 to 10

The long-term platform of the panel 12 decision (Q12.4): StableHLO importer, C++ e-graph library, and the torch entry.
These waves follow M5 and the wave 7 follow-ons; wave 10 ends at M6.
No date is set before the M1 rebaseline.

## Wave 8: pin set, importer, library core

8 items, weight 24.

| ID | Item | Depends on | Size | Note |
| -- | ---- | ---------- | ---- | ---- |
| U15 | Toolchain pin set: one released StableHLO, LLVM from its tag, MLIR and StableHLO static libraries in one build image, CI cache, prebuilt importer | U9 | M | risk K28 |
| B57 | Class data record and `merge` change report for B49 | B7 | S | split from B7: class data before B49 |
| B47 | StableHLO importer: C++ tool reading portable artifacts, printing today's kernel text, coded errors | B42, B43, U15 | M | -- |
| B48 | Dialect spike: generic instruction op against per-ISA ODS, per-ISA build cost measured | B13, U15 | S | -- |
| B49 | E-graph library core: arena, typed key, union-find, deferred rebuild, class data, read-only view, property tests | B26, B29, B31, B57, B7, B9 | L | -- |
| O34 | fsim step 1: C++ emitter per instruction from O3's generic emitter | O29, O3, U15 | L | -- |
| O35 | fsim step 2: `Pipeline::submit`, `end_trace`, `sync` replace `compile_and_load` and `execute` | O34 | M | -- |
| T39 | StableHLO corpus exported once from jax, differential against the HLO path | B47, T13 | S | -- |

## Wave 9: dialect, matcher, engine swap, torch adapter

7 items, weight 24.
B53 may move to wave 8, after B47, so the pairing check runs early (K29).

| ID | Item | Depends on | Size | Note |
| -- | ---- | ---------- | ---- | ---- |
| B50 | ISA dialect generator per B48: typed attributes, checked verifier, semantics region from O3 | B48, O3, U10 | M | -- |
| B51 | Matcher and rules: backtracking per operator, early preconditions, matched ids, applied memo, C++ rule emitter | B49, B6, B8 | L | -- |
| B52 | Dual-engine harness and swap through the B2 C ABI, three-tier gate, egg deleted | B10, B2, B3, B32, B34, B35, B51 | M | swap gate, risk K30 |
| B53 | Torch adapter: `ExportedProgram` to portable artifact; pairing check picks torch-mlir or the walker; strict | B42, B47 | M | pairing check first, risk K29 |
| O36 | fsim steps 3 and 4: debug print out of `custom_call`, PJRT pinned by header and plugin hash; every golden and norm row holds | O35, U15 | M | -- |
| A29 | `torch.compile` and `torch.export` entry points, artifact-hash cache, one released torch per release, CI row per minor | A23, B53 | M | -- |
| T40 | Torch tests: eager PyTorch against ACT on the oracle per operation and dtype, coverage table | A29, T29 | M | -- |

## Wave 10: typed import, C++ driver

5 items, weight 18.

| ID | Item | Depends on | Size | Note |
| -- | ---- | ---------- | ---- | ---- |
| B54 | Typed import: importer builds library terms, chosen program as dialect ops, text bridge deleted | B47, B50, B52 | M | -- |
| B55 | XLA HLO text reader in C++, holding every message of `tests/test_reader.py` | B54 | M | -- |
| B56 | One C++ driver: importers, engine, extraction, allocator, cost, exclusion report; Rust crate deleted | B46, B54, B55 | L | -- |
| A30 | Partition opt-in: supported subgraphs to ACT, every undelegated node printed | A29, B46 | M | -- |
| T41 | Platform gate: B3 on the C++ driver, oracle on fsim | B56, O36 | M | -- |
