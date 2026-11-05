# Overview

This page is S1 of the TAIDL language specification: its scope, what conformance means, and the run-time cases it gives a fixed outcome.
Section ids `S1` to `S12` are stable; a generator cites them when it lists the features it does not support.

## Scope

- An ISA specification is a Python file that builds one `Accelerator`: its buffers and its state, then its instructions.
- An instruction is a decorated Python function that the front end runs once on symbolic attributes (S3).
- The buffers and the state together are the accelerator's data model.
- TAIDL describes what each instruction computes, as a function from its input regions, attributes, and state to its output regions and state.
- TAIDL does not describe timing, pipelining, or how instructions overlap in time.
- Terms are used as S10 defines them.

## S1.1 Conforming front end

- A conforming front end accepts every ISA specification this document allows and rejects every ISA specification S9 lists.
- The `taidl` package is the reference front end.
- Its check function `taidl.check` collects every problem S9 lists and raises one `Rejection`; a construct the tracer cannot record raises at the author's line (S3.2).
- It rejects only problems the architect can correct, such as a typo or an invalid ISA specification, and each error names the clause it breaks (S9).
- The width layer, the encoding layer, and generator metadata are optional (S11, S12); without them every attribute is a signed 64-bit integer.

## S1.2 Conforming generator and unsupported features

- A generator takes the `Model` that `check` returns (S4.6), and runs its own checks after `check` (S9).
- A conforming generator gives each instruction the meaning S5 to S8 define, for every feature it supports.
- For a feature it does not support, it excludes the instruction that uses it with a printed and recorded reason, or rejects the ISA specification when the architect can correct it (S9).
- Either outcome names the feature; neither gives it another meaning.
- Each generator lists the features it does not fully support in its own docs, one row per feature, citing the section id here.
- A row has one status: `supported`, `partial`, which works with stated limits, `excluded`, which drops the instruction with a reason, or `rejected`, which raises an error naming the feature.
- The three non-supported statuses carry a code and a section id.
- The feature table is generated from the generator's expectations table (S12.4).
- The corpus is the set of ISA specifications the `act` repository ships for generator tests; a corpus case is one of them.

## S1.3 Faults and defined behavior

TAIDL has no undefined run-time case: every run-time violation is a fault or has the fixed result its section gives.
A case the front end cannot check is checked when the instruction is issued (S5.6); a failed check is a fault.
A fault depends only on attributes, state, the recorded expressions, the size of `hbm`, and which elements earlier issues of the program wrote; tensor values never fault.
A fault writes no region and no state; the program stops at the first fault, and the test oracle reports it with the issue's index, the instruction's name, and the section id below.
A compiler backend never emits an issue that faults (S5.6).

| id | case | outcome | section |
| --- | --- | --- | --- |
| S1.3.1 | An operator of any recorded expression (S6.1), a constraint and a symbolic `constant` value included, whose value lies outside signed 64 bits | fault | S6.2 |
| S1.3.2 | A symbolic `//` or `%` divisor that evaluates to 0 | fault | S6.2 |
| S1.3.3 | A region whose evaluated start is below 0, whose size is below 1, or whose start plus size exceeds its access dimension; an attribute value may be negative | fault | S5.2 |
| S1.3.4 | Two output regions of one instruction that overlap | fault | S5.2 |
| S1.3.5 | A constraint of any source that does not hold | fault | S5.3 |
| S1.3.6 | An input region that overlaps an output region outside their declared relation | fault, as S1.3.5 | S5.5 |
| S1.3.7 | A symbolic shape dimension below 1 | fault | S7.4 |
| S1.3.8 | A shape dimension that depends on an attribute or state and breaks S7.2 or S8 | fault | S7.2, S8.4 |
| S1.3.9 | A symbolic `constant` value its element type cannot hold | fault | S7.5 |
| S1.3.10 | An attribute value outside its type at issue, `s64` by default; reported as S1.3.10 even when a constraint of another source also fails on that issue | fault | S5.6, S11.2 |
| S1.3.11 | A read of an element of a buffer other than `d0`, with no declared initial value (S4.3), that no earlier issue of the program wrote | fault | S5.6 |
| S1.3.12 | Tensor integer edge cases inside a semantics: division and remainder by 0, and of the least value by -1; overflow; shifts past the width; negative `power` exponents; out-of-range indices | fixed result; indices clamp as StableHLO states | S8.4 |
| S1.3.13 | `reduce_<reducer>` initial value | the reducer's identity at the type; a float sum starts from +0.0 | S8.4 |
| S1.3.14 | Byte layout of multi-byte, sub-byte, and `pred` values in `d0` | fixed layout | S8.4 |
| S1.3.15 | Float accumulation of `dot`, `reduce_add`, and `reduce_multiply` | within the bound of S8.4 | S8.4 |
| S1.3.16 | Float rounding, denormals, NaN, and signed zero | one rounding per operation, denormals kept, NaN and zero as S8.4 states | S8.4 |
| S1.3.17 | A `convert` whose value the result type cannot hold | fixed result per conversion class | S8.4 |
| S1.3.18 | A binary word that matches the fixed bits of no instruction | decode error, naming the word in hexadecimal and its index from 0 | S11.4 |
| S1.3.19 | A finite result of an approximate word, such as `exponential`, `log`, or `tanh` | within the ulp bound of S8.4 | S8.4 |
