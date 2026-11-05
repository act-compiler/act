# Overview

This page is S1 of the spec: its scope, what conformance means, and the run-time cases the spec gives a fixed outcome.
Section ids `S1` to `S10` are stable; a generator cites them when it lists the features it does not support.

## Scope

- An ISA description is a Python program that builds one `Accelerator`: its buffers and its state, then its instructions.
- The buffers and the state together are the accelerator's data model.
- TAIDL describes what each instruction computes, as a function from its input regions, attributes, and state to its output regions and state.
- TAIDL does not describe timing, pipelining, or how instructions overlap in time.
- Terms are used as S10 defines them.

## S1.1 Conforming front end

- A conforming front end accepts every ISA description this document allows and rejects every ISA description S9 lists.
- The `taidl` package is the reference front end.
- It rejects at the call that takes the faulty text, by raising `ValueError`, `TypeError`, or `SyntaxError` as S9 gives.

## S1.2 Conforming generator and unsupported features

- A generator takes an accelerator that the front end accepted, through the object model of S4.6.
- A conforming generator gives each instruction the meaning S5 to S8 define, for every feature it supports.
- For a feature it does not support, it rejects the ISA description with an error that names the feature; it never gives it another meaning.
- Each generator lists the features it does not fully support in its own docs, one row per feature, citing the section id here.
- Each row has one status: `partial`, which works with stated limits, or `rejected`, which raises an error naming the feature.

## S1.3 Faults and defined behavior

No run-time case of TAIDL is undefined.
A case the front end cannot check is checked when the instruction is issued (S5.6); a failed check is a fault.
A fault writes no region and no state, and the generator reports it with the instruction's name and the row id below.
Every other run-time case has the fixed result its section gives.

| id | case | outcome | section |
| --- | --- | --- | --- |
| S1.3.1 | A start, size, shape dimension, operation attribute, or state update whose final value lies outside signed 64 bits | fault | S6.2 |
| S1.3.2 | A computed `/` or `%` divisor of 0 in an expression | fault | S6.2 |
| S1.3.3 | A region whose start is below 0, whose size is below 1, or whose start plus size passes its access dimension | fault | S5.2 |
| S1.3.4 | Two output regions of one instruction that overlap | fault | S5.2 |
| S1.3.5 | A constraint that does not hold | fault | S5.3 |
| S1.3.6 | An in-place input whose starts differ from output region 0's at issue | fault | S5.5 |
| S1.3.7 | A backtick shape dimension below 1 | fault | S7.4 |
| S1.3.8 | A shape or element type that depends on a backtick value and breaks S7.2 or S8 | fault | S7.2, S8.4 |
| S1.3.9 | A backtick `constant` whose value the line's element type cannot hold | fault | S7.5 |
| S1.3.10 | An attribute value outside signed 64 bits at issue | fault | S5.6 |
| S1.3.11 | Contents of storage no output region writes | unchanged; every buffer starts zero-filled | S5.6 |
| S1.3.12 | Integer division by 0, shifts past the width, float-to-integer overflow inside a semantics | fixed result | S8.4 |
| S1.3.13 | `reduce_<reducer>` initial value | the reducer's identity | S8.4 |
| S1.3.14 | Byte layout of multi-byte, sub-byte, and `pred` values in `d0` | fixed layout | S8.4 |
| S1.3.15 | Float accumulation of `dot` and `reduce_*`, and transcendental accuracy | open; pending panel | S8.4 |
