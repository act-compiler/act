# Corpus format and runners

T5 fixes where a case lives and what it holds; T6 makes the `check` verdict executable; T29 runs instruction-level programs on an oracle.
Every case is written in the authoring surface of Q11.2: a Python file whose instructions are decorated functions.
All three land in the taidl repo and change no existing test.

## T5. Corpus format, location, and naming

- What: a directory `taidl/conformance/` in the taidl wheel, grouped by outline file, one case per directory named by a slug; the S ids live in `case.toml`.
- What: a case holds `spec.py` (the ISA specification: an `Accelerator` and `@acc.instruction` functions, traced by the tracer) and `case.toml`.
- What: optional files are `kernel.hlo` and `inputs.npz` (backend and T13) and `program.toml`; no case holds semantics text, since v3 has no string form.
- What: a case that needs the printed form of its `Model` stores it as `model.hlo` (the HLO-text printer of Q11.2), which a test compares and no generator reads.
- What: `program.toml` holds initial buffers and state, instruction calls, and the expected buffers and state, or a fault; it is executor-neutral, so a non-Python oracle can run it.
- What: `case.toml` fields are `section` (S ids), `description` (one line stating the case's intent), `origin`, `verdict` (`accepted` or `rejected`), and for a rejected case `phase` (`trace`, `check`, `generate`, `run`), `code`, `clause`, and `object`.
- What: `origin` is `hand`, `seeded:<old test path>`, `property:<seed>` for a promoted shrunk case (T9), or `accelerator:<name>`; T18 greps it for the successor of each deleted test.
- What: the coverage renderer prints `description` beside the case id.
- What: `code` and `object` are the match key (Q3.9); `clause` is the S9 or S1.3 id; the runner never matches message prose.
- What: a `trace` rejection is a Python exception raised at the author's line by the tracer (`__bool__`, `/` on an integer, slice on a region); `check` rejections arrive in one `Rejection` whose entries each carry `code`, `clause`, `object`, and the author's file and line (Q3.8).
- What: `error_type` is required for the `trace` phase only; a `check` case matches each entry by `code` and `object`; a `run` fault matches on `code`, since a C or C++ oracle has no Python class.
- What: a `check` case lists every entry it expects, since `check` collects all problems before it raises; a case with two typos expects two entries.
- What: a generation case may end `excluded` instead of `accepted` or `rejected`: the backend drops the instruction from its instruction set and the case names the instruction, `code`, and clause of the printed reason.
- What: `clause = "pending"` is allowed until L numbers S9, and CI fails on it once the numbered list ships.
- What: an `expect` scalar states the run outcome: `exact`, `almost` (with `ulps`, or the S1.3.15 bound for a float `dot` or `reduce`), or `fault`; `nan = "any"` relaxes NaN comparison.
- What: a `model` table keeps facts of the checked `Model` that a case asserts, such as an inferred result type or a literal value; the tracer and the converter produce the same table (T37).
- What: expected values are stored data; `reference` names NumPy (pinned version) or a closed form, never JAX.
- What: `xfail = "<id>"` runs the case and fails on XPASS.
- What: a removed v3 construct is a `rejected` case whose code names the replacement.
- What: `taidl/conformance/CHANGES.md` lists added and removed case ids, and `check_versions.py` prints the delta on a pin bump.
- Benefits: one place to read what the language accepts; the generators load cases by path with no copy.
- Caveats: the surface is the only ISA specification entry point in v3, so cases are Python; a text format waits for L.
- Fail points: a wrong S id in `case.toml`; noticed by the T26 report listing it under the wrong section.
- Order and dependencies: L1 publishes section ids and numbers the S9 rows (`taidl/docs/conformance.md:9-19` has rows without ids today); first corpus PR.
- Verification: the `case.toml` schema is documented in `taidl/conformance/README.md`; a loader test reads every case and rejects a missing field, including `description` and `origin`.
- Breaking: no.
- Files: new `taidl/conformance/`; no existing file changed.

## T6. Check verdict runner in taidl

- What: `taidl/tests/test_conformance.py` loads every case and asserts the verdict: an accepted case passes `taidl.check(acc)` and returns a `Model`, a rejected case raises the `trace` exception or the `Rejection` entries the case lists, with the case's `phase`, `code`, and `object`.
- What: `code` and `object` are read from the diagnostic's fields, so rewording a message fails no case; D owns the few text tests of the message format and the one golden test per class.
- What: the runner also runs `python -m taidl check spec.py` on every case once per run and expects the same verdict and a nonzero exit for a rejection (decision 42).
- What: a second pass runs `check` through a generator entry given the `Accelerator`, and expects it to call `check` once.
- Benefits: the S9 list becomes executable; a front-end change that widens or narrows the language fails here first.
- Caveats: a trace-time exception stops at once while `check` collects, so a case with a trace error lists exactly one expectation; the runner records the phase that rejects.
- Fail points: a code renumbered; codes are never reused or renumbered, so none.
- Order and dependencies: after T5 and D3 (front-end codes) and the `check` of U3; the gate for T7.
- Verification: the runner collects one test id per case directory, and the count equals the directory count.
- Breaking: yes: front-end error text gains the code and the clause id, released with TAIDL v3; the clause id sits on line 1 of the message until the owner decides otherwise.
- Files: new `taidl/tests/test_conformance.py`; no existing file changed.

## T29. Program runner over an oracle interface

- What: a runner executes each `program.toml` through an oracle interface (build state, issue calls, read buffers and state), never through JAX.
- What: a fault case asserts `code`, `clause`, and unchanged buffers and state; a result case, including a fixed-value case, compares buffers within the case tolerance.
- What: a fault depends only on attributes, state, and region text, so a `fault` case never sets a tensor value as its trigger.
- What: the interface has one adapter for the generated Python oracle; a C or C++ oracle adds an adapter later.
- Benefits: attribute values, addresses, and state reach S1.3.1 to S1.3.5, which an HLO kernel cannot set.
- Caveats: the seeds at `act-oracle/tests/test_kernel.py:590-598` and :1229-1240 compare the whole message and raise at `k('fsim-compile')`; the case names the outcome and the code, not when the oracle detects it.
- Fail points: an adapter that hides a write before the fault; the unchanged-state check catches it.
- Order and dependencies: after T5, T6, and D8 (oracle run-time codes).
- Verification: every fault case of the T26 catalogue runs here; the seeds are removed from `test_kernel.py` as their cases land.
- Breaking: no.
- Files: new `taidl/testing` runner and `act-oracle` adapter; seeds from `test_kernel.py:590-598`, :1229-1240.
