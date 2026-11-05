# Corpus extensions

T9 adds generated cases where hand-written ones run out.
T10 names the real ISA specifications the end-to-end suite runs.
T24 fuzzes the surface against the S9 contract, and T25 measures whether the suite would catch a mutated check.

## T9. Property-based tests for shapes, element types, and names

- What: Hypothesis strategies for ranks, dimension sizes, element types, and identifier names, driving the front end (shape agreement in S7.4), the oracle (S8 words at random shapes), and the backend (generated names compile, extending `act-backend/tests/test_names.py:95`).
- What: runs use `derandomize` with a fixed seed on every pull request; a shrunk counterexample is promoted to a corpus case with `origin = "property:<seed>"`.
- Benefits: finds the shape and type edge cases a hand-written corpus misses, such as the sub-byte packing rule in `act-oracle/docs/operations.md:14`.
- Caveats: adds the `hypothesis` dependency, allowed when this item lands; generation runs are slow, so backend properties run on generated source, not on a cargo build.
- Fail points: flaky examples from unseeded runs; fixed by the fixed seed and the Hypothesis database kept out of the repos.
- Order and dependencies: after T6 and T8; Q2.
- Verification: each property names the S section it tests; the per-pull-request run stays inside the T27 budget.
- Breaking: no.
- Files: `act-backend/tests/test_names.py` extended; no file removed.

## T10. Real ISA specifications as the end-to-end list

- What: a short list in `taidl/conformance/accelerators.md` naming the real ISA specifications the suite runs end to end: `mini_trn`, `mini_trn_double` (`taidl/examples/`, `act-backend/examples/`, `act-oracle/examples/`), plus the GPU tensor-core specification A writes.
- What: the scope is tensor accelerators plus GPU tensor-core instructions (mma, wgmma, TMA), so one row is the fixed Hopper sketch, whose kernels exercise S5.2 regions on shared memory and a dot at mma shapes.
- What: each row names the specification, its kernels, the reference for each kernel, whether a golden exists, and the S8 words and S5 features it needs.
- Benefits: one list replaces the `EXAMPLES` dict in `taidl/tests/test_examples.py` and the two `examples/` trees' implicit lists.
- Caveats: the examples live in three repos today; the list points to the taidl copy and the generators fetch it through the pin.
- Fail points: a listed specification that a generator does not support in full; T14 skips and reports a kernel whose expectations row says `unsupported`.
- Order and dependencies: after T5; input to T14.
- Verification: `test_examples.py` in taidl and act-oracle read the list, not their own constants.
- Breaking: no.
- Files: `taidl/tests/test_examples.py`, `act-oracle/tests/test_examples.py`; their checks preserved, their lists replaced by the shared one.

## T24. Nightly surface fuzzer

- What: a nightly job generates ISA specification files from the surface (random instruction functions built from the operation table, expressions, and `require` terms) and asserts one property: `check` returns a `Model`, or the tracer or `check` rejects with a `code`.
- What: the language specification says a conforming front end rejects exactly the listed descriptions (S9), which is this property on unwritten inputs.
- What: a counterexample is promoted to a corpus case with `origin = "property:<seed>"`.
- Benefits: tests the S9 contract beyond the hand-written cases.
- What: a generated file that is accepted must also print and reparse (the printer of Q3.4), so the fuzzer tests the printer.
- Caveats: starts after the T6 runner exists; uses the T9 strategies and the `hypothesis` dependency.
- Fail points: a crash that is neither acceptance nor a coded rejection; the job fails and keeps the seed.
- Order and dependencies: after T6 and T9; Q2.
- Verification: the nightly log names the seed and the case count.
- Breaking: no.
- Files: new nightly job config in taidl; no test file removed.

## T25. Mutation pilot over front-end and oracle fault checks

- What: `mutmut` mutates the front-end checks and the oracle's fault checks, running the corpus per mutant.
- What: each surviving mutant gets a new case or a written reason.
- What: the pilot runs once per release; it is never a pull request gate.
- Benefits: the only measure of whether the suite is sufficient, not just large.
- Caveats: adds the `mutmut` dependency, allowed when this item lands; a run executes the suite per mutant, so scope stays narrow.
- Fail points: a mutant that survives because the check is dead code; the reason is written and the check is removed.
- Order and dependencies: after T6 and T12; Q2.
- Verification: every survivor has a case id or a reason in the pilot report.
- Breaking: no.
- Files: new pilot config in taidl and act-oracle; no test file removed.
