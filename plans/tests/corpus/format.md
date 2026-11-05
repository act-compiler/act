# Corpus format and front-end runner

T5 fixes where a case lives and what it holds; T6 makes the front-end verdict executable.
Both land in the taidl repo and change no existing test.

## T5. Corpus format, location, and naming

- What: a directory `taidl/conformance/` with one case per directory, `S<n>.<m>-<slug>/` holding `spec.py` (an `Accelerator` built with the front-end API), `case.toml` (section ids, front-end verdict, error text pattern when rejected), and optional `kernel.hlo` and `inputs.npz` for T13.
- What: the verdict field is one of `accepted`, `rejected`, or `implementation-defined`; `rejected` carries the S9 clause id and a regex of the error the front end must raise; `implementation-defined` names the S1.3 clause and takes no generator expectation.
- What: a case that exercises semantics text alone (S2, S3, S7) also carries `semantics.txt`, so the runner can feed the lexer and parser directly as `taidl/tests/test_grammar.py:9` does today.
- What: a case with a kernel also states its run outcome, `result` with a reference value, or `error` with the S id of the norm it violates and a regex of the defined error; TAIDL v3 has no undefined behavior, so every run-time violation (region outside its buffer, overlapping outputs, signed 64-bit overflow, zero divisor) is one or the other.
- Benefits: one place to read what the language accepts; the generators load cases by path with no copy.
- Caveats: `Accelerator` is the only ISA description entry point today (`taidl/tests/test_instruction.py:10`), so cases are Python, not a text format; a text format waits for L to decide whether one exists.
- Fail points: a case directory with a wrong S id; noticed by T11 tables listing it under the wrong section, and by L's review of the corpus listing.
- Order and dependencies: L must publish stable S ids and number the S9 rows (`taidl/docs/conformance.md:9-19` has rows without ids today); first corpus PR.
- Verification: `case.toml` schema documented in `taidl/conformance/README.md`; a loader test reads every case and rejects a missing field.
- Breaking: no.
- Files: new `taidl/conformance/`; no existing file changed.

## T6. Front-end verdict runner in taidl

- What: `taidl/tests/test_conformance.py` loads every case and asserts the verdict: an accepted case builds without raising, a rejected case raises the error the regex names.
- Benefits: the spec's S9 list becomes executable; a front-end change that widens or narrows the language fails here first.
- Caveats: today rejection happens at several layers (`SyntaxError` at load per `act-oracle/README.md:56`, `ValueError` in the API per `taidl/tests/test_instruction.py:53-66`); the runner records which layer rejects.
- Fail points: an error message rewording fails many cases; mitigated by matching on the S9 clause id once L and the front end carry it in messages (T22 question 2).
- Order and dependencies: after T5; the gate for T7.
- Verification: the runner collects one test id per case directory, named by the directory, and the count equals the directory count.
- Breaking: yes, when T22 question 2 is confirmed: front-end error text gains the S9 clause id, released with TAIDL v3.
- Files: new `taidl/tests/test_conformance.py`; no existing file changed.
