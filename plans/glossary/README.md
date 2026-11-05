# Reconciled glossary

The single glossary for the TAIDL spec and its four repos, with a ruling on every conflict between the two glossary drafts and the tracks' objections.

Owner decisions 19-21 override these files where they differ:

- A data model has buffers (not "tensor buffers") and state (not "control registers" or "initial state"); the API adds them as buffers and state.
- `hbm` stays `hbm` (not "global memory").
- "Region" stays (not "data slice").
- The `@s.` prefix stays.

- [terms.md](terms.md): every canonical term with its definition, source, banned synonyms, and the v3 identifier renames.
- [rulings.md](rulings.md): every conflict, its ruling, and the reason.
- [words-to-avoid.md](words-to-avoid.md): each word to replace and its replacement.

## Source tags

| Tag | Source |
| --- | --- |
| [ACT] | ACT paper, OOPSLA 2026: https://act-compiler.github.io/assets/pdf/2026-oopsla-act.pdf |
| [TAIDL] | TAIDL paper, MICRO 2025: https://act-compiler.github.io/assets/pdf/2025-micro-taidl.pdf |
| [SHLO] | StableHLO specification: https://openxla.org/stablehlo/spec |
| [XLA] | XLA operation semantics: https://openxla.org/xla/operation_semantics |
| [MLIR] | MLIR language reference and dialect docs: https://mlir.llvm.org/docs/ |
