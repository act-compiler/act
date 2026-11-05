# Overview

This page is S1 of the spec: its scope, what conformance means, and the run-time cases the spec gives a fixed outcome.
Section ids `S1` to `S12` are stable; a generator cites them when it lists the features it does not support.

## Scope

- An ISA specification is a Python program that builds one `Accelerator`: its buffers and its state, then its instructions.
- The buffers and the state together are the accelerator's data model.
- TAIDL describes what each instruction computes, as a function from its input regions, attributes, and state to its output regions and state.
- TAIDL does not describe timing, pipelining, or how instructions overlap in time.
- Terms are used as S10 defines them.

## S1.1 Conforming front end

- A conforming front end accepts every ISA specification this document allows and rejects every ISA specification S9 lists.
- The `taidl` package is the reference front end.
- It rejects at the call that takes the faulty text, by raising `ValueError`, `TypeError`, or `SyntaxError` as S9 gives.
- It rejects only problems the architect can correct, such as a typo or an invalid ISA specification, and each error names the clause it breaks (S9).
- The width layer, the encoding layer, and generator metadata are optional (S11, S12); without them every attribute is a signed 64-bit integer.

## S1.2 Conforming generator and unsupported features

- A generator takes an accelerator that the front end accepted, through the object model of S4.6.
- A conforming generator gives each instruction the meaning S5 to S8 define, for every feature it supports.
- For a feature it does not support, it rejects the ISA specification with an error that names the feature; it never gives it another meaning.
- Each generator lists the features it does not fully support in its own docs, one row per feature, citing the section id here.
- Each row has one status: `partial`, which works with stated limits, or `rejected`, which raises an error naming the feature.

## S1.3 Faults and defined behavior

TAIDL has no undefined run-time case: every run-time violation is a fault or has the fixed result its section gives.
A case the front end cannot check is checked when the instruction is issued (S5.6); a failed check is a fault.
A fault depends only on attributes, state, and region text; tensor values never fault.
A fault writes no region and no state; the program stops at the first fault, and the generator reports it with the instruction's name and the section id below.
A compiler backend never emits an issue that faults (S5.6).

| id | case | outcome | section |
| --- | --- | --- | --- |
| S1.3.1 | An operator of a start, size, shape dimension, operation attribute, or state update expression whose value lies outside signed 64 bits | fault | S6.2 |
| S1.3.2 | A computed `/` or `%` divisor of 0 in an expression | fault | S6.2 |
| S1.3.3 | A region whose evaluated start is below 0, whose size is below 1, or whose start plus size passes its access dimension; an attribute value may be negative | fault | S5.2 |
| S1.3.4 | Two output regions of one instruction that overlap | fault | S5.2 |
| S1.3.5 | A constraint, or any other REQUIRE check, that does not hold | fault | S5.3 |
| S1.3.6 | An input region that overlaps an output region outside their declared relation | fault, as S1.3.5 | S5.5 |
| S1.3.7 | A backtick shape dimension below 1 | fault | S7.4 |
| S1.3.8 | A shape or element type that depends on a backtick value and breaks S7.2 or S8 | fault | S7.2, S8.4 |
| S1.3.9 | A backtick `constant` whose value the line's element type cannot hold | fault | S7.5 |
| S1.3.10 | An attribute value outside its type at issue, `s64` by default | fault | S5.6, S11.2 |
| S1.3.11 | A read of an element of a buffer other than `d0` that no earlier issue of the program wrote | fault | S5.6 |
| S1.3.12 | Integer division or remainder by 0, shifts past the width, out-of-range indices inside a semantics | fixed result; indices clamp as StableHLO states | S8.4 |
| S1.3.13 | `reduce_<reducer>` initial value | the reducer's identity at the type; a float sum starts from +0.0 | S8.4 |
| S1.3.14 | Byte layout of multi-byte, sub-byte, and `pred` values in `d0` | fixed layout | S8.4 |
| S1.3.15 | Float accumulation of `dot`, `reduce_add`, and `reduce_multiply` | within the bound of S8.4 | S8.4 |
| S1.3.16 | Float rounding, denormals, NaN, and signed zero | one rounding per operation, denormals kept, NaN and zero as S8.4 states | S8.4 |
| S1.3.17 | A `convert` whose value the result type cannot hold | fixed result per conversion class | S8.4 |
| S1.3.18 | A binary word that matches the fixed bits of no instruction | decode error, naming the word and its position | S11.4 |
