# Glossary

The single glossary for the TAIDL language specification and its four repos.

- [terms.md](terms.md): every canonical term with its definition, source, banned synonyms, and the v3 identifier renames; it covers the overlap relations, markers, implicit constraints, the attribute width and encoding layers (encoding block, encoded and derived attributes), generator metadata and its metadata entries, and the failure terms (rejection, fault, dropped candidate, invalid input kernel).
- [words-to-avoid.md](words-to-avoid.md): the banned column of `terms.md`, one row per canonical term.
- [rulings.md](rulings.md): the rulings in force that no term row carries, including the overlap, layer, and generator metadata rulings.

## Source tags

| Tag | Source |
| --- | --- |
| [ACT] | ACT paper, OOPSLA 2026: https://act-compiler.github.io/assets/pdf/2026-oopsla-act.pdf |
| [TAIDL] | TAIDL paper, MICRO 2025: https://act-compiler.github.io/assets/pdf/2025-micro-taidl.pdf |
| [SHLO] | StableHLO specification: https://openxla.org/stablehlo/spec |
| [XLA] | XLA operation semantics: https://openxla.org/xla/operation_semantics |
| [MLIR] | MLIR language reference and dialect docs: https://mlir.llvm.org/docs/ |

A source cell may also cite an owner decision ("Decision 19") or a panel question ("Q7.8").
