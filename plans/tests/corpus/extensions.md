# Corpus extensions

T9 adds generated cases where hand-written ones run out.
T10 names the real ISA descriptions the end-to-end suite runs.

## T9. Property-based tests for shapes, element types, and names

- What: Hypothesis strategies for ranks, dimension sizes, element types, and identifier names, driving the front end (shape agreement in S7.4), the oracle (S8 words at random shapes), and the backend (generated names compile, extending `act-backend/tests/test_names.py:95`).
- Benefits: finds the shape and type edge cases a hand-written corpus misses, such as the sub-byte packing rule in `act-oracle/docs/operations.md:14`.
- Caveats: adds the `hypothesis` dependency, which the owner must approve (T22 question 3); generation runs are slow, so backend properties run on generated source, not on a cargo build.
- Fail points: flaky examples from unseeded runs; fixed by a fixed seed in CI and the Hypothesis database kept out of the repos.
- Order and dependencies: after T6 and T8; Q2.
- Verification: each property names the S section it tests; a shrunk counterexample becomes a corpus case under T5.
- Breaking: no.
- Files: `act-backend/tests/test_names.py` extended; no file removed.

## T10. Real ISA descriptions as the end-to-end list

- What: a short list in `taidl/conformance/accelerators.md` naming the real ISA descriptions the suite runs end to end: `mini_trn`, `mini_trn_double` (`taidl/examples/`, `act-backend/examples/`, `act-oracle/examples/`), plus the descriptions A names for the next quarters.
- What: the scope is tensor accelerators plus GPU tensor-core instructions (mma, wgmma, TMA), so the list reserves one row for a tensor-core description whose kernels exercise S5.2 regions on shared memory and a dot at mma shapes.
- What: each row names the description, its kernels, the reference for each kernel, and whether a golden exists.
- Benefits: one list replaces the `EXAMPLES` dict in `taidl/tests/test_examples.py` and the two `examples/` trees' implicit lists.
- Caveats: the examples live in three repos today; the list points to the taidl copy and the generators fetch it through the pin.
- Fail points: an ISA description listed that no generator supports in full; noticed by T12 expectations marking rows as rejected.
- Order and dependencies: after T5; input to T14.
- Verification: `test_examples.py` in taidl and act-oracle read the list, not their own constants.
- Breaking: no.
- Files: `taidl/tests/test_examples.py`, `act-oracle/tests/test_examples.py`; their checks preserved, their lists replaced by the shared one.
