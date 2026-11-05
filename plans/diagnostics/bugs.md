# Diagnostic bugs

IDs use the prefix `E` because `D` is the plan prefix.
Severity: high means two generators disagree on one description or a failure is silent; medium means a message misleads or a check sits in the wrong layer; low means wording or channel.
Nothing was changed; each fix is a plan item (D1 to D13) or an item of another track.

| ID | Severity | Where | Bug | Fix sketch |
| --- | --- | --- | --- | --- |
| E1 | high | `act-backend/act_backend/generator.py:119-121` | An instruction with no semantics is dropped from the backend without a message; the oracle rejects one with outputs (`semantic_generator.py:16`) and runs one without as a no-op (`:19`) | Front end rejects (RF040, B24); then the skip is dead and deleted (D5) |
| E2 | high | `act-backend/act_backend/spec_checks.py:394-415` | A region value's element type is never compared with its data model's; the oracle rejects the same description (`semantic_generator.py:45`) | RF042 in the front end (B22, L12) |
| E3 | high | `generic/cpp/malloc/src/parser.cc:13-14, :187-188, :316`, `constraint.cc:112-187`, `CMakeLists.txt` | `assert(false)` is the error path for nine faults (two in the parser, seven in the live-range pass); `CMakeLists.txt` sets no build type, and a Release build removes the assert and returns `0`, `nullptr` or `{0,0}` | Return reasons with `I` codes (B25, D7); never assert for a check |
| E4 | high | `generic/src/isel/extractor/fast.rs:191` | `debug_assert!(false, ...)` is compiled out by `cargo build --release` (`generator.py:182`); the loop breaks with no error | Return `Err` with an `I` code (D7) |
| E5 | high | `taidl/taidl/instruction.py:90, :210-215` | `check_values` runs only after both `set_inputs` and `add_semantics`; a description without `set_inputs` passes the front end and fails later in a generator (`spec_checks.py:195`, `semantic_generator.py:40`) | One end-of-description check (RF041, D4) |
| E6 | high | `semantic_generator.py:53`, `spec_checks.py:415`, `generic/utils.py:662` | A region value larger than its region, a size below 1, and a value of another shape are rejected by one generator and accepted by the other (S1.3 lists them undefined) | Under v3 no undefined behavior: front end `R` for literals, `V` in the oracle, `U` or check in the backend (D4, D6) |
| E7 | medium | `taidl_to/oracle_visitor.py:255, :261`, `generator.py:41` | `NotImplementedError` marks both spec errors and unsupported features; the backend uses `ValueError` for unsupported (`spec_checks.py:48-85`) | Exception classes of `scheme.md`; `NotImplementedError` only for `U` (D2, O6) |
| E8 | medium | all of `taidl/taidl/*.py` | No message or exception carries the ISA description file and line; a rejection raised in a generator has no traceback into the description | Store the call site in each object; add it to line 2 of the format (D2) |
| E9 | medium | `generic/src/isel/initializer/lib.rs:154-187`, `nodes.rs:39-431`, `expression.rs:106-464`, `line.rs:96` | Kernel errors name the value but not the kernel line number | Keep a line number per kernel line in the reader; print `kernel.hlo:12` (B5, D7) |
| E10 | medium | `pipeline.rs:59-60, :208-218, :266` | The error shows only the first dropped candidate's reason; the others and their count go to stdout | Print the count and the first three codes on stderr (D7) |
| E11 | medium | `pipeline.rs:551-552, :558-563` | `stopped for an unexpected reason` and `timeout() called at ... not reached` name no cause and no way to report it | `I` code, and print the runner's stop reason (D7) |
| E12 | medium | `generic/cpp/malloc/src/emitter.cc:30, :76, :104, :130, :284, :317` | Constant messages are fragments (`a constant of element type {t}`) that read only after the wrapper adds `the allocator cannot write` | Whole sentences with `UB016` (D7) |
| E13 | medium | `act_backend/spec_checks.py:218, :252, :294`, `generic/utils.py:268-271` | A computed integer outside signed 64 bits is a hard rejection in the backend and a warning in the oracle; S1.3 calls it implementation-defined | One norm: a defined error (`VO005`, `VB001`) (D4, D6) |
| E14 | medium | `taidl/taidl/instruction.py:413-415`, `spec_checks.py:210, :255, :292`, `generic/utils.py:80` | Zero divisors are checked three ways: literal sibling, folded integers, per iteration | Front end folds integers (RF045, L14); oracle keeps `VO004` (D4) |
| E15 | medium | `generic/utils.py:555`, `decorator.py:21, :150`, `generic/utils.py:675-692` | Messages name no instruction or kernel | Add the name (D8) |
| E16 | medium | `tests/test_refusals.py` (backend and oracle), `tests/test_constraints.py:154-165` | Tests pin full message text, so any wording change breaks them | Assert code and object; one text test per format rule (D10) |
| E17 | medium | `spec_checks.py:1-418` | 35 lines use "refuse", "refusal", `honoured` (`:331`); the oracle has 22; the front end has 4 | Rename to "reject" (L26, L28, B27, O15) |
| E18 | medium | `generator.py:90-117` | A bad `cost=` path raises after the previous backend and crate were removed | Validate arguments before removing anything; keep removal before spec rejections (tested at `tests/test_refusals.py:439`) |
| E19 | low | `pipeline.rs:266, :344-349` | A drop and a warning print on stdout, errors on stderr | One channel rule in the scheme: diagnostics on stderr |
| E20 | low | `act/tests/check_versions.py:96-99` | A missing pattern raises `TypeError: 'NoneType' object is not subscriptable` with no file name | Test for `None` and name the file (A15) |
| E21 | low | `act/docker/build.sh:72, :82`, `run.sh:50`, `cleanup.sh:20`, `tutorials/asplos26/docker.sh:43` | Four spellings of an unknown argument; one error goes to stdout | One helper `fail()` in a shared shell file (D12) |
| E22 | low | `act/act/generators/__init__.py:6, :13` | A missing generator package gives a bare `ImportError` | Raise `XA` with the install command (D12) |
| E23 | low | `isel/initializer/parser.rs:268-306` | A line outside any computation is skipped in silence | Reject with `KB` code (B5) |
| E24 | low | `spec_checks.py:22-27`, `generator.py:119-121` | Only instructions with semantics are checked; the rest are never validated | Validate all, then skip (superseded by E1) |
