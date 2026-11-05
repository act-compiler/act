# Plan: Python surface, check function, converter, ANTLR removal

These items move ISA specifications to the Python surface of S2 and S3, and put one check function before every generator (owner decisions 41 and 42, panel 11).
All land in v3 in the order of the table's dependencies; the string surface and ANTLR stay on the taidl 2.x line only.
Sizes assume the shared 3-4 engineers.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| L43 | Check function `taidl.check` returning a frozen `Model`, and generator checks | U2, U3 | M | Q1 |
| L42 | Python surface by tracing | L43 | L | Q1 |
| L44 | Converter script on the 2.x line and per-file `Model` equivalence | L42 | M | Q2 |
| L45 | Remove the string surface, `taidl/antlr4/`, grammar tests, and the ANTLR dependency | L44 | S | Q2 |

## L43 Check function and generator checks

- What: `taidl.check(acc)` runs every S9 rule, freezes the accelerator, and returns a `Model`; only `check` constructs one (S4.6, S9).
- Changes: `check` subsumes U3's `validate()`; it first builds the `Model` from the string front end's trees, so every generator port is tested on unchanged inputs.
- Changes: every problem is collected into one `Rejection`; each entry has a code, a section id, the instruction, and the author's file and line, in the line-1 `[S<id>]` form.
- Changes: `python -m taidl check spec.py` runs it with no generator.
- Changes: a generator entry takes a `Model`, or an `Accelerator` on which it calls `check` once; its body reads only the `Model`; B and O own their ports.
- Changes: generator checks are plain functions after `check`, with no registry and no checker subclass; the oracle adds none.
- Changes: B excludes an instruction that reads or updates state, has other than one output, sizes over an addressing attribute or state, has a constraint its allocator cannot hold, or uses a word outside its table.
- Changes: each exclusion carries a code, a section id, and the author's line; B prints it at generation and stores it in the generated crate; an empty usable set rejects.
- Changes: B's front-end duplicates in `spec_checks.py` go, as U3 already rules.
- Breaking: yes; a generator given an invalid specification now fails in `check`, with every problem at once.
- Benefits: "check always runs once before any generator" holds by type; an ISA with one stateful instruction still gets a backend for the rest.
- Caveats: S1.2 gains the `excluded` status; a kernel that needs an excluded instruction is rejected at kernel compile time naming it.
- Fail points: a generator reads the `Accelerator` behind the `Model`; notice by a test that hands each generator a `Model` only.
- Order and dependencies: after U2 and U3; before L42.
- Verification: a specification with five faults reports five entries in one run; B's printed exclusion list matches its crate's stored list on a corpus case with a semaphore instruction.

## L42 Python surface by tracing

- What: the surface of S2 to S7: `@acc.instruction` functions traced once on symbolic attributes, `buffer.region`, `read`, `write` with markers, `require`, `any_of`, `all_of`, `update`, `ops.<word>`, and `Comp[t]` and `Addr[t]`.
- Changes: a tracer in `taidl` records each call into the same `Model` that L43 builds from strings; no source reading and no new dependency.
- Changes: `__bool__` on a symbolic value raises, which catches `if`, `and`, `or`, `not`, and a chained comparison; S3.2's other constructs raise naming the spelling to use.
- Changes: result types and shapes are inferred by U4's shape module; a target type or shape is written only where the operation takes one.
- Changes: every node records the first frame outside `taidl` as its call site; the `Model` printer emits XLA-HLO text as a log format.
- Changes: `acc.encode(instr, word, fixed, fields)` sits beside the instruction (S11.3); `meta=` sits on `Accelerator`, `acc.buffer`, and `@acc.instruction` (S12).
- Breaking: yes, every ISA specification is rewritten, by L44.
- Benefits: errors point at the author's line instead of a line inside a string; no parser, no `parameter(k)` type line, no `ROOT`.
- Caveats: the `ops` spelling of a word that is a Python keyword is `ops.and_`, `ops.or_`, `ops.not_`.
- Fail points: a construct the tracer cannot record yields a silent meaning; notice by a test per S3.2 row that expects the raise and the named spelling.
- Order and dependencies: after L43; U4 supplies inference.
- Verification: the `Model` traced from each example equals the one L43 builds from its string form, up to call sites.

## L44 Converter and equivalence

- What: a script on the taidl 2.x line that reads loaded v2 `Instruction` objects and prints the Python surface; not a `taidl` subcommand.
- Changes: one visitor over `instr_input_asts`, `instr_output_asts`, `constraint_asts`, `update_asts`, and the semantics tree; run once on the corpus in taidl, act-oracle, act-backend, and act; converted files are committed.
- Changes: one test per corpus file: the `Model` built from the v2 string form equals the `Model` traced from the converted file, up to call sites.
- Changes: a test that assembles semantics text at run time is rewritten by hand from the printer's output.
- Breaking: no on its own; it carries L42's break.
- Benefits: the corpus moves without hand edits, and the equivalence test checks the contract itself.
- Caveats: no public 2.x release carries the script; a release can wrap it if an outside owner of a v2 specification appears.
- Fail points: a construct the visitor misses; notice by the equivalence test on its file.
- Order and dependencies: after L42; before L45.
- Verification: every corpus file passes its equivalence test.

## L45 Remove the string surface and ANTLR

- What: delete `add_semantics`, the string start, size, constraint, and update forms, `taidl/antlr4/`, the generated parsers, the grammar tests, and the ANTLR runtime pin in v3.
- Changes: each removed call raises one coded rejection naming its replacement and the section id; `add_semantics` names the L44 script (S9 Removed constructs, L33's table).
- Breaking: yes, the v2 surface is gone.
- Benefits: `taidl` has no third-party runtime dependency; the trees and their unstable shapes leave the contract.
- Caveats: L13 has no target after it.
- Fail points: a tutorial or test still uses a string form; notice by the removed-names rejection in CI.
- Order and dependencies: after L44, last of these items.
- Verification: `taidl` installs with no ANTLR package; a grep for `antlr4` and `add_semantics` outside the removed-names table is empty in the four repos.
