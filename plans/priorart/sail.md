# Sail

## What it is

- Sail is a first-order imperative language for ISA specifications, used for RISC-V, Arm, CHERI and x86 models.
- A spec declares `register` state, an `instr` union with one constructor per instruction, a `decode` mapping and an `execute` function with one `function clause` per constructor.
- Bitvector types carry dependent lengths with constraints, as in `val replicate : forall 'n 'm, 'n >= 1 & 'm >= 1. (int('n), bits('m)) -> bits('n * 'm)`.
- One source generates emulators (C, OCaml), theorem-prover definitions (Lem, Coq, Lean) and SMT problems.
- Sources: https://github.com/rems-project/sail/blob/sail2/doc/asciidoc/language.adoc and https://github.com/riscv/sail-riscv.

## TAIDL today, for comparison

- TAIDL splits an instruction across Python calls and three text grammars: `add_instruction`, `set_inputs`, `add_semantics`, `add_constraints` (taidl/taidl/instruction.py:96-329).
- Execution order is prose, S5.4 in taidl/docs/instructions.md: constraints, then updates in order, then starts and shapes, then semantics.
- Undefined behavior is one bucket, S1.3, cited from S5.2, S5.3 and S7.2.

## Borrow

- One normative execution function: write S5.4 as a pseudo-code `execute(instr, state)` in the spec, and have both generators cite its steps by number.
- Dependent shape types: a region value's shape is a function of attributes and state, the way Sail bitvector lengths are functions of type variables; the spec should name that function once (S7.4 and S5.2) so shape checks in act-backend's `generate_precond_function` (act-backend/act_backend/isel_ir2isa_rewrites_rs_generator.py:565-623) and in the oracle's `check_shapes` (act-oracle/taidl_to/oracle_visitor.py:301) implement the same definition.
- Exhaustiveness: Sail warns on a non-exhaustive `match`; TAIDL's analogue is an instruction set where some attribute value satisfies no instruction's constraints, which a front end could report.
- Multiple backends from one source: Sail's emulator-plus-prover model is the shape TAIDL already has with oracle and backend; the formal track should state which generated artifact is the reference.
- Explicit `undefined` as a value distinct from undefined behavior: Sail lets a register start `undefined`; TAIDL should say whether an unwritten data model element is a value (any bit pattern) or undefined behavior to read.

## Avoid

- A full imperative language inside the ISA description; TAIDL's choice of a pure dataflow block per instruction is the simpler generator target.
- Scattered definitions: Sail's `scattered function` lets one instruction be spread over files, which TAIDL's `add_semantics` replaces-on-second-call rule (S7.1) already forbids.

## Relevance

- High for the spec track; medium for the generators.
