# Later, supporting tracks

Plan items scheduled for later in the supporting tracks, by track.

## priorart

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| P6 | Cost-aware extraction | B10, P4 | L | -- |

## shared

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| U9 | One copy of docker and CI scripts | U8 | S | -- |
| U15 | Toolchain pin set: one released StableHLO, LLVM from its tag, MLIR and StableHLO static libraries in one build image, CI cache, prebuilt importer | U9 | M | -- |

## diagnostics

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| D13 | Machine-readable output and code explanations | D1, D7, D8 | S | -- |

## semantics

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| F7 | Conformance suite derived from the rules: cut, carried by T7, T8, T26, T29 | -- | - | -- |
| F9 | Executable reference interpreter of the exact classes: cut, carried by T8 | -- | - | -- |
