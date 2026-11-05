# Prior art for TAIDL and its generators

One file per system, a cross-system ranking, and plan items P1 to P12.
Terminology follows the owner's papers first, then StableHLO and MLIR.

| file | contents |
| --- | --- |
| [sail.md](sail.md) | Sail: ISA specs as state, decode mappings and execute clauses; one source, many backends |
| [ila-3la.md](ila-3la.md) | ILA and 3LA: decode and state-update semantics, flexible matching with two rule kinds |
| [exo.md](exo.md) | Exo: instructions as procedures with asserts and bodies, replace by unification |
| [tvm-halide.md](tvm-halide.md) | TVM tensorize, TensorIR intrinsics and Halide: two-description intrinsics and ordered rule tables |
| [llvm-tablegen.md](llvm-tablegen.md) | LLVM TableGen, SelectionDAG and GlobalISel: typed patterns, PatFrag, skipped-pattern reports |
| [cranelift-isle.md](cranelift-isle.md) | Cranelift ISLE: typed rules, extractors, priorities, overlap |
| [egg-egglog.md](egg-egglog.md) | egg and egglog: runner limits, analyses, rulesets, extraction, explanations |
| [isaria-diospyros-hydride.md](isaria-diospyros-hydride.md) | Isaria, Diospyros, Hydride: rule synthesis, phased saturation, pseudocode-derived selectors |
| [mlir-dialects.md](mlir-dialects.md) | ArmSME, NVGPU, X86Vector, AMX and IREE microkernels: typed tiles, descriptors, two op layers |
| [stablehlo.md](stablehlo.md) | StableHLO spec style: numbered constraints, three-word taxonomy, examples as tests |
| [nki.md](nki.md) | AWS NKI ISA documentation: per-instruction pages with engine, rules and undefined cases |
| [borrow.md](borrow.md) | Ranked borrow list and ranked avoid list across systems |
| [plan/README.md](plan/README.md) | Plan item index table: ID, item, dependencies, size, quarter |
| [plan/P1-numbered-constraints.md](plan/P1-numbered-constraints.md) | P1 Numbered per-operation constraints in S8 |
| [plan/P2-taxonomy.md](plan/P2-taxonomy.md) | P2 Three-word taxonomy for every S1.3 item |
| [plan/P3-named-typed-rules.md](plan/P3-named-typed-rules.md) | P3 Named typed rewrite rules with priorities |
| [plan/P4-phased-saturation.md](plan/P4-phased-saturation.md) | P4 Phased saturation schedule |
| [plan/P5-reports.md](plan/P5-reports.md) | P5 Skipped-pattern and stop-reason reports |
| [plan/P6-cost-extraction.md](plan/P6-cost-extraction.md) | P6 Cost-aware extraction |
| [plan/P7-examples-as-tests.md](plan/P7-examples-as-tests.md) | P7 Executable examples per spec row |
| [plan/P8-normative-execute.md](plan/P8-normative-execute.md) | P8 Normative execute function and reference-interpreter status |
| [plan/P9-doc-pages.md](plan/P9-doc-pages.md) | P9 Generated instruction documentation pages |
| [plan/P10-check-extractors.md](plan/P10-check-extractors.md) | P10 Reusable check extractors |
| [plan/P11-state-kinds.md](plan/P11-state-kinds.md) | P11 Typed hardware state kinds and data-model layout properties |
| [plan/P12-expansion-rule-tests.md](plan/P12-expansion-rule-tests.md) | P12 Named and oracle-tested expansion rules |
