# Wave 1

Foundations and every wave 1 gate; 55 items, weight 88.
Items with no dependency start here, except later-quarter items that nothing early needs.
The assisted weeks per item are recorded in this wave, and the M1 rebaseline sets dates from them (Q8.9).

| ID | Item | Depends on | Size | Note |
| -- | ---- | ---------- | ---- | ---- |
| L1 | Ship the spec skeleton and section ids | -- | S | -- |
| L2 | Pin the reference operator version | L1 | S | -- |
| L3 | Normative shape and element type rules per operation | L2 | L | Q8.4 step a; the L item measured in wave 1 (K1) |
| L6 | 64-bit integer arithmetic per operator with a fault | L1 | S | -- |
| L7 | ACT-Backend unsupported-feature table against section ids | L1 | S | -- |
| L10 | Reject `@a.` outside starts and constraints | L1 | S | -- |
| L11 | `check` reports a late buffer, a late state variable, and a partial instruction at the declaring call's site | L1 | S | -- |
| L24 | Multi-core, async DMA, mask, stride, saturation, and `scaled_dot` idioms, documented without syntax | L1 | S | -- |
| L25 | Reconcile the glossary with the terminology reference | L1 | S | -- |
| L29 | Object model contract test for the five fixed names | L1 | S | -- |
| L30 | v3 rename table: public API and `d0` as a one-dimension buffer | L1 | M | -- |
| L33 | Remove rare words and redundant spellings, with one coded rejection | L1 | S | -- |
| L36 | Expand each pair's relation with the size term and the exact-or-apart default | L1 | S | -- |
| L5 | Make the region, in-place, and overlap fault rows hold | L1, L36 | S | -- |
| L35 | Add `set_may_overlap` and the pair-relation record | L36 | S | -- |
| B25 | C++ errors returned, never `assert` (D5) | -- | S | -- |
| B29 | Deterministic search mode and `search` as a library function | -- | M | head of the e-graph chain (Q8.2) |
| B1 | Target module layout and stage names | B29 | S | -- |
| B3 | End-to-end gate: goldens, chosen assembly, candidate logs, oracle run | B29 | M | end-to-end gate on B29 (Q8.2) |
| B38 | Allocator attribute domains follow the width layer (D15); default `s64` | -- | S | allocator domain fix lands with the width layer (Q10.4) |
| O5 | Delete the dead branch and the dead `sh_copy` check | -- | S | -- |
| O16 | Generation log lines that state what happens | -- | S | -- |
| O18 | Drop the hand-written line splitter for generated code | -- | S | -- |
| O21 | XLA flags per compile: no excess precision, no flush to zero, IEEE min and max | -- | S | -- |
| O19 | Spike: unroll every loop on a synthetic stress kernel against the Q5.1 bound | O21 | S | gate: the R28 bound is the Q5.1 resolution; baselines measured before the spike (Q8.10) |
| T1 | Tag every surviving test with its language specification section | L1 | M | -- |
| T2 | Remove hidden fixture coupling in the two conftest.py files | -- | M | -- |
| T3 | One shared test kit for helpers duplicated across repos | T2 | M | -- |
| T5 | Corpus format, location, and naming | L1 | S | -- |
| T20 | Decision: impl tests survive with a header, PR-defense tests do not | -- | S | gate: test policy, binds T4 and T18 (Q8.4, Q8.10) |
| T22 | Answers in force for the test-suite questions | -- | S | gate: answers to the test-suite questions (Q8.10) |
| T28 | Mapped-deletion check | -- | S | -- |
| A2 | Fix README structure, punctuation, and CI description | -- | S | -- |
| A8 | One name per generator across the four repos | -- | M | -- |
| A5 | Fix act/ docstrings and the distribution name | A8 | S | -- |
| A9 | Retire the "current tutorial" label | -- | S | -- |
| A10 | List the tutorial syntax that fails on the v3 API | -- | M | gate: failing tutorial syntax listed (Q8.10) |
| A11 | Archive all three tutorials on their tags and remove them from master | A10 | M | absorbs A22: one tutorial lint scope (Q7.1) |
| A15 | Close gaps in check_versions.py | -- | S | -- |
| A16 | Write the release order and a checklist | -- | M | gate: release order and checklist before taidl 3.0 (Q8.5) |
| A32 | Repo topology, branching, private-to-public sync, private CI owner, and the 2.x maintenance policy | -- | S | gate: private development and 2.x patch releases (owner 47) |
| C10 | Modeling cookbook and scope notes | -- | S | -- |
| C3 | Instruction families and indexed state: idiom | C10 | S | -- |
| C9 | Float attributes and bit operators: idiom | C10 | S | -- |
| C11 | Case studies as an executable corpus | -- | M | -- |
| C4 | Mode-dependent semantics: idiom | C11 | S | -- |
| C13 | Defined-behavior audit of the case studies | C11 | S | -- |
| P1 | Numbered per-operation constraints in S8 | -- | M | -- |
| P2 | Three-word taxonomy (rejected, fault, defined value) for every S1.3 item | P1 | S | -- |
| F1 | Typing rules that U4 implements in the front end | -- | M | -- |
| F2 | Region aliasing rules and in-place mark meaning | -- | M | un-merged from L5 (Q8.6) |
| F3 | Total integer arithmetic: overflow and zero divisors fault | -- | S | held until R56 is ruled (Q8.6) |
| F6 | Byte order, sub-byte packing, size of d0 | -- | S | -- |
| C1 | Typed-view idiom and byte order | C10, F6 | S | -- |
| F8 | Front-end rejections the spec already implies | -- | S | -- |
