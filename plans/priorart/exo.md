# Exo

## What it is

- Exo is a Python-embedded scheduling language where hardware is defined in user libraries, not the compiler.
- An instruction is a procedure under `@instr("<C template>")` whose body is ordinary Exo code giving its meaning, whose arguments carry memory annotations such as `@ Neon`, and whose `assert stride(dst, 0) == 1` lines state when it applies.
- `replace(proc, cursor, instr)` unifies the instruction body with a code fragment: statements and non-integer expressions must match exactly, integer control expressions become linear equations that are solved.
- Memories are classes with allocation, free and address-formatting hooks.
- Sources: https://exo-lang.dev, https://github.com/exo-lang/exo/blob/main/docs/instructions.md, https://github.com/exo-lang/exo/blob/main/docs/Design.md, PLDI 2022 https://dl.acm.org/doi/10.1145/3519939.3523446.

## TAIDL today, for comparison

- TAIDL's semantics block is Exo's instruction body in dataflow form (taidl/docs/semantics.md S7.1).
- TAIDL's constraints are Exo's asserts, but only over attributes and state, never over the matched program's layout (taidl/taidl/antlr4/IDLV2Expr.g4:22-25).
- Exo's `replace` is one-shot syntactic unification; ACT's e-graph search explores many rewritten forms before mapping (act-backend/act_backend/generic/src/pipeline.rs:491-495).

## Borrow

- Instruction as a single self-contained unit: name, arguments with their memories, applicability asserts and body in one place; TAIDL's `add_instruction`, `set_inputs`, `set_outputs`, `set_inplace` (pending panel 9), `add_semantics`, `add_constraints` sequence (taidl/taidl/instruction.py:96-329) could gain one declarative form that calls the same checks.
- Memory as a declared class with its own rules: TAIDL's data model (S4.3) is this; add per-data-model properties a generator reads, such as alignment or in-place rules, instead of per-instruction marks.
- Unification with integer equation solving: the backend's attribute recovery (`generate_comp_attr_extraction`, act-backend/act_backend/isel_ir2isa_rewrites_rs_generator.py:220) does a restricted version; stating it as "solve attribute from shape" in the spec makes the accepted forms explicit.

## Avoid

- Lowering by string template: `@instr("vst1q_f32(&{dst_data}, {src_data});")` is opaque to analysis; TAIDL's assembly printing should stay generated from declared operands.
- Scheduling as the user's job: Exo externalizes optimization to a human; ACT's goal is an automatic backend, so Exo's cursors and rewrites are not the model.

## Relevance

- Medium: strong on the instruction-as-procedure shape, weak on automatic selection.
