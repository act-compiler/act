# ILA and 3LA

## What it is

- ILA (Instruction-Level Abstraction) is an ISA-like formal model of an accelerator: architectural state variables, interface inputs, and per-instruction `SetDecode` (when the instruction fires) and state-update functions.
- ILAng is the C++ DSL: `m.NewBvState`, `m.NewMemState`, `m.NewInstr`, `instr.SetDecode`, `instr.SetUpdate`; child ILAs model multi-step instructions.
- 3LA uses ILA as the software/hardware interface for compilation: "flexible matching" with two rule kinds, accelerator-independent compiler-IR rewrites and accelerator-specific IR-to-accelerator mapping rules, run by equality saturation over Glenside, then code generation against ILA simulators.
- All 3LA rewrites are polymorphic over tensor size; mappings state input-to-output size relations.
- Sources: https://arxiv.org/abs/2203.00218 (3LA), https://github.com/PrincetonUniversity/ILAng, https://bo-yuan-huang.gitbook.io/ilang/.

## TAIDL today, for comparison

- TAIDL has the same two-kind split: `ir2ir_rewrites.txt` and `ir2isa_rewrites.txt` concatenated into one rule list (act-backend/act_backend/generic/src/isel/rewrites/lib.rs:6-19).
- ILA's decode condition is TAIDL's constraints (S5.3); ILA's state update is `update` (S5.4); both exist in the language.
- The backend rejects any read or update of state (act-backend/docs/coverage.md, row "State variable and its initial value"), so stateful instructions are oracle-only today.

## Borrow

- Decode-before-update discipline: ILA checks decode on the pre-state, which is exactly S5.4 step 1; keep it and make the oracle's constraint check cite it.
- Keep stateful effects only in mapping rules: 3LA states that IR rewrites stay purely functional and state lives only in the accelerator mapping; the backend should model state as sequencing in the PII graph rather than as e-graph nodes.
- Refinement obligation: ILA verifies an RTL implementation against the ILA; TAIDL's analogue is oracle-vs-backend agreement on every selected program, which the test track should state as the conformance property.
- Size-polymorphic mapping rules: TAIDL's `dot_?` and `slice_?` wildcards (act-backend/act_backend/generic/src/isel/rewrites/ir2ir_rewrites.txt:1-10) are the same idea; name the shape relation each rule assumes.

## Avoid

- Hand-written mapping rules per accelerator: 3LA's designers wrote mappings by hand; TAIDL's generated `ir2isa` rules from semantics are the better route, so do not regress to hand rules for special cases.
- Modeling the accelerator at the MMIO command level; TAIDL's region-level instructions are the right granularity for a compiler.

## Relevance

- High: closest prior art for both the state model and the two-kind rewrite split.
