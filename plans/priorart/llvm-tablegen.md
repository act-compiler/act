# LLVM TableGen, SelectionDAG and GlobalISel

## What it is

- TableGen records describe instructions: `OutOperandList`, `InOperandList`, `AsmString`, `Pattern`, `Predicates`, with `class`, `multiclass`, `defm` and `let` for sharing.
- A SelectionDAG pattern is a DAG with typed named operands, as in `def STrr : F3_1<3, 0b000100, (outs), (ins MEMrr:$addr, IntRegs:$src), "st $src, [$addr]", [(store i32:$src, ADDRrr:$addr)]>`.
- `PatFrag` names reusable sub-patterns, `ComplexPattern` names a C++ matcher such as `SelectADDRrr`, and each pattern has a complexity and cost the generated matcher uses to rank candidates.
- GlobalISel imports the same patterns into a match table; `--warn-on-skipped-patterns` and `--stats` report which patterns could not be imported and why.
- Sources: https://llvm.org/docs/TableGen/ProgRef.html, https://llvm.org/docs/WritingAnLLVMBackend.html, https://llvm.org/docs/GlobalISel/InstructionSelect.html.

## TAIDL today, for comparison

- The backend's rule is a numbered string `0: lhs => rhs` whose checks are a Rust function picked by matching the number (act-backend/act_backend/generic/src/isel/rewrites/applier.rs:9-50).
- Operand constraints are emitted inline per rule by `generate_precond_function` (act-backend/act_backend/isel_ir2isa_rewrites_rs_generator.py:565-623): shape checks, attribute recovery, constant checks, constraints, allocator integer checks.
- Each instruction is one `TensorOp` variant generated into egraph.rs (act-backend/act_backend/ir_egraph_rs_generator.py:20-60).

## Borrow

- Named, typed rules: give each rewrite a name and an operand list with declared kinds (value, constant, attribute) rather than a number and a wildcard string.
- Reusable sub-matchers: `PatFrag` and `ComplexPattern` correspond to shape-check, attribute-recovery and constant-check helpers that today are re-emitted per rule; emit them once and compose.
- Skipped-pattern report: emit, at generation time, the list of instructions and rules that cannot be selected and the reason, as GlobalISel does.
- Pattern complexity as an explicit number: the extractor's preference between candidates (fast first, slow with limit, act-backend/act_backend/generic/src/isel/extractor/fast.rs:13 and slow.rs:14) should rest on a declared per-rule cost or priority.

## Avoid

- TableGen's untyped `dag` values and string-pasted assembly: both are sources of silent errors LLVM has learned to lint; TAIDL should keep assembly operands declared.
- Greedy bottom-up single-pattern selection: e-graphs already remove the phase-ordering problem; do not reintroduce a first-match-wins table.

## Relevance

- High for the backend's rule representation.
