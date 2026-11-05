# Corpus format and runners

T5 fixes where a case lives and what it holds; T6 makes the front-end verdict executable; T29 runs instruction-level programs on an oracle.
All three land in the taidl repo and change no existing test.

## T5. Corpus format, location, and naming

- What: a directory `taidl/conformance/` in the taidl wheel, grouped by outline file, one case per directory named by a slug; the S ids live in `case.toml`.
- What: a case holds `spec.py` (the ISA specification as an `Accelerator` built with the front-end API) and `case.toml`.
- What: optional files are `semantics.txt` (S2, S3, S7, S8 text fed to the lexer and parser), `kernel.hlo` and `inputs.npz` (backend and T13), and `program.toml`.
- What: `program.toml` holds initial buffers and state, instruction calls, and the expected buffers and state, or a fault; it is executor-neutral, so a non-Python oracle can run it.
- What: `case.toml` fields are `section` (S ids), `verdict` (`accepted` or `rejected`), and for a rejected case `phase` (`lex`, `parse`, `build`, `generate`, `run`), `code`, `clause`, and `object`.
- What: `code` and `object` are the match key (Q3.9); `clause` is the S9 or S1.3 id; the runner never matches message prose.
- What: `error_type` is required for front-end phases only; a `run` fault matches on `code`, since a C or C++ oracle has no Python class.
- What: `clause = "pending"` is allowed until L numbers S9, and CI fails on it once the numbered list ships.
- What: an `expect` scalar states the run outcome: `exact`, `almost` (with `ulps`, or the S1.3.15 bound for a float `dot` or `reduce`), or `fault`; `nan = "any"` relaxes NaN comparison.
- What: a `parsed` table keeps parsed-value facts such as the literal text at `taidl/tests/test_grammar.py:352-357`; lex and parse collapse if they cannot be separated.
- What: expected values are stored data; `reference` names NumPy (pinned version) or a closed form, never JAX.
- What: `xfail = "<id>"` runs the case and fails on XPASS; `origin = "property:<seed>"` marks a promoted shrunk case.
- What: a removed v3 construct is a `rejected` case whose code names the replacement.
- What: `taidl/conformance/CHANGES.md` lists added and removed case ids, and `check_versions.py` prints the delta on a pin bump.
- Benefits: one place to read what the language accepts; the generators load cases by path with no copy.
- Caveats: `Accelerator` is the only ISA specification entry point today (`taidl/tests/test_instruction.py:10`), so cases are Python; a text format waits for L.
- Fail points: a wrong S id in `case.toml`; noticed by the T26 report listing it under the wrong section.
- Order and dependencies: L1 publishes section ids and numbers the S9 rows (`taidl/docs/conformance.md:9-19` has rows without ids today); first corpus PR.
- Verification: the `case.toml` schema is documented in `taidl/conformance/README.md`; a loader test reads every case and rejects a missing field.
- Breaking: no.
- Files: new `taidl/conformance/`; no existing file changed.

## T6. Front-end verdict runner in taidl

- What: `taidl/tests/test_conformance.py` loads every case and asserts the verdict: an accepted case builds without raising, a rejected case raises the diagnostic with the case's `phase`, `code`, and `object`.
- What: `code` and `object` are read from the diagnostic's fields, so rewording a message fails no case; D owns the few text tests of the message format and the one golden test per class.
- Benefits: the S9 list becomes executable; a front-end change that widens or narrows the language fails here first.
- Caveats: rejection happens at several layers today (`SyntaxError` at load per `act-oracle/README.md:56`, `ValueError` in the API per `taidl/tests/test_instruction.py:53-66`); the runner records the phase that rejects.
- Fail points: a code renumbered; codes are never reused or renumbered, so none.
- Order and dependencies: after T5 and D3 (front-end codes); the gate for T7.
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
