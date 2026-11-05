# Plan: Python surface, check function, converter, ANTLR removal

These items move ISA specifications to the Python surface of S2 and S3, and put one check function before every generator (owner decisions 41 and 42, panel 11).
All land in v3 in the order of the table's dependencies; the string surface and ANTLR stay on the taidl 2.x line only.
L42, L44, and L45 are merged into the shared items U12, U13, and U14, which carry their text, unioned edges, one owner, and their exits as sub-items.
Sizes assume the shared 3-4 engineers.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| L43 | Check function `taidl.check` returning a frozen `Model`, and generator checks | U2, U3 | M | Q1 |
| L42 | Python surface by tracing: merged into U12 | U12 | - | - |
| L44 | Converter script on the 2.x line and per-file `Model` equivalence: merged into U13 | U13 | - | - |
| L45 | Remove the string surface, `taidl/antlr4/`, grammar tests, and the ANTLR dependency: merged into U14 | U14 | - | - |

## L43 Check function and generator checks

- What: the layer over U3, which is the one check function `taidl.check(acc)`; L43 adds no second one (S4.6, S9).
- What: U3 owns the frozen `Model`, the one `Rejection`, and `python -m taidl check`; L43 makes `check` hold every S9 rule and adds the generator checks after it.
- Changes: `check` first builds the `Model` from the string front end's trees, so every generator port is tested on unchanged inputs.
- Changes: each S9 rule adds its problems to U3's one `Rejection`, each entry in the line-1 `[S<id>]` form.
- Changes: a generator entry takes a `Model`, or an `Accelerator` on which it calls `check` once; its body reads only the `Model`; B and O own their ports.
- Changes: generator checks are plain functions after `check`, with no registry and no checker subclass; the oracle adds none.
- Changes: B excludes an instruction that reads or updates state, has other than one output, has a constraint its allocator cannot hold, or uses a word outside its table.
- Changes: each exclusion carries a code, a section id, and the author's line; B prints it at generation and stores it in the generated crate; an empty usable set rejects.
- Changes: B's front-end duplicates in `spec_checks.py` go, as U3 already rules.
- Breaking: yes; a generator given an invalid specification now fails in `check`, with every problem at once.
- Benefits: "check always runs once before any generator" holds by type; an ISA with one stateful instruction still gets a backend for the rest.
- Caveats: S1.2 gains the `excluded` status; a kernel with no covering is rejected at kernel compile time, class `K`, listing excluded instructions as a hint (S9).
- Fail points: a generator reads the `Accelerator` behind the `Model`; notice by a test that hands each generator a `Model` only.
- Order and dependencies: after U2 and U3; before U12.
- Verification: a specification with five faults reports five entries in one run; B's printed exclusion list matches its crate's stored list on a corpus case with a semaphore instruction.

## L42 Python surface by tracing

- Merged into shared item U12, whose sub-item "L42 exit" is its exit; L14 depends on U12.

## L44 Converter and equivalence

- Merged into shared item U13, whose sub-item "L44 exit" is its exit; A28's dependency on L44 is carried by U13.
- The last 2.x release carries the converter script (A28).

## L45 Remove the string surface and ANTLR

- Merged into shared item U14, whose sub-item "L45 exit" is its exit; its dependency on T37 is carried by U14.

## S9 rows the reference front end does not yet check

Each row is part of S9; the plan item that makes `check` hold it is beside it.

| row | plan item |
| --- | --- |
| Every row, collected into one `Rejection` from a traced specification | U12, L43 |
| A buffer after any instruction | L11 |
| An addressing attribute outside a start or a constraint | L10 |
| A state value outside signed 64 bits, or of type `bool` | L16 |
| A plain `constant` value the element type cannot hold, or a plain shape dimension below 1 | L16 |
| An in-place input whose buffer or size expressions differ from the output's | L5 |
| `apart` on equal start expressions, or equal input and output expressions without `inplace` | L36 |
| Two outputs with equal start expressions, an exact-or-apart pair with equal starts and unequal plain sizes, or an input tied to two outputs | L36 |
| A plain region start, size, or bound past its limit, or an axis attribute that is not a plain integer | L16 |
| A `read` with no `write`, or a `require` that is `False` from plain values alone | U12, L43 |
| A removed construct, by the removed-names table | L33, U14 |
| Shapes and element types against S7.2 and S8 | U4 |
| Every row of Layers and generator metadata | L37, L38, L39, L40 |
| Generator checks, with exclusion | L43 |
