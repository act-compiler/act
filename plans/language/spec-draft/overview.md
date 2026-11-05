# Overview

This page is S1 of the spec: its scope, what conformance means, and the cases the spec leaves open.
Section ids `S1` to `S10` are stable; a generator cites them when it lists the features it does not support.

## Scope

- An ISA description is a Python program that builds one `Accelerator`: its data models, its state variables, then its instructions.
- TAIDL describes what each instruction computes, as a function from its input regions, attributes, and state to its output regions and state.
- TAIDL does not describe timing, pipelining, or how instructions overlap in time.
- Terms are used as S10 defines them.

## S1.1 Conforming front end

- A conforming front end accepts every ISA description this document allows and rejects every ISA description S9 lists.
- The `taidl` package is the reference front end.
- It rejects at the call that takes the faulty text, by raising `ValueError`, `TypeError`, or `SyntaxError` as S9 gives.

## S1.2 Conforming generator and unsupported features

- A generator takes an accelerator that the front end accepted.
- A conforming generator gives each instruction the meaning S5 to S8 define, for every feature it supports.
- For a feature it does not support, it rejects the ISA description with an error that names the feature; it never gives it another meaning.
- Each generator lists the features it does not fully support in its own docs, one row per feature, citing the section id here.
- Each row has one status: `partial`, which works with stated limits, or `rejected`, which raises an error naming the feature.

## S1.3 Undefined and implementation-defined behavior

An ISA description whose instruction reaches an undefined case, for some attribute and state values, is in error for those values.
An implementation-defined case is one each generator decides and documents.

| case | kind | section |
| --- | --- | --- |
| An integer an expression computes falls outside signed 64 bits | implementation-defined | S6.2 |
| A region reaches outside its data model's addressed dimensions | undefined | S5.2 |
| A region's size evaluates to 0 or less | undefined | S5.2 |
| A computed `/` or `%` divisor is 0 | undefined | S6.2 |
| An operation's shapes or element types break its reference operation's rules | undefined; the front end does not check them | S8.4 |
| An operation's result where its reference operation leaves the result open, such as a float `convert` that overflows | implementation-defined, within what the reference operation allows | S8.4 |
| A region value line whose element count or element type differs from its region's | undefined; the front end does not check it | S7.2 |
| An addressing attribute outside a start or a constraint | undefined; the front end does not reject it | S6.1 |
| Two output regions of one instruction overlap | undefined | S5.2 |
| An instruction is issued where one of its constraints does not hold | undefined | S5.3 |
| The byte order `bitcast_convert` uses between types of different widths, which fixes how multi-byte values lie in `d0` | implementation-defined | S8.4 |
| The initial value of each `reduce_<reducer>` | implementation-defined | S8.4 |
| An initial state value outside signed 64 bits | undefined | S4.4 |
| The size of the off-chip memory `d0` | implementation-defined | S4.3 |
| What an in-place mark tells a generator beyond the computed result | implementation-defined | S5.5 |
