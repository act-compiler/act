# Inconsistent wording for the same condition

The canonical verb is "reject" (glossary S10, `work/terminology/words-to-avoid.md`).
Banned synonyms found in code are quoted so a grep finds them.

## The rejecting verb

| Word | Where | Count of lines |
| --- | --- | --- |
| "Refuse", "refused" | `taidl/taidl/names.py:8`, `taidl/taidl/instruction.py:116, :145, :453` | 4 |
| "refused", "Refuse", "Refuses", `start_refusal`, `division_refusal` | `act_backend/spec_checks.py:1, :23, :33, :180, :243, :258, :285` and others | 35 (all Python, Rust, C++ under `act-backend`) |
| "Refuse", "refusal", "refuses" | `taidl_to/api_generator.py:52`, `oracle_visitor.py:195, :285`, `instruction_codegen.py:199`, `generic/utils.py:77, :275` | 22 |
| "honoured" | function name `honoured` at `act_backend/spec_checks.py:331` | 1 |
| `refuse(...)`, `panic(...)`, `lacks(...)` as C++ helpers | `generic/cpp/malloc/src/parser.cc:22, :181, :187` | 3 |

## The unsupported-feature phrase

| Wording | Where |
| --- | --- |
| `which the selector does not support` | `act_backend/spec_checks.py:48, :58` |
| `which the backend does not support` | `spec_checks.py:364`, `isel/initializer/line.rs:61`, `expression.rs:464` |
| `which the backend does not model` | `spec_checks.py:250, :375, :382` |
| `the backend supports only a stride of one` | `isel/initializer/nodes.rs:346` |
| `is not one the backend supports` | `ir/dtype.rs:145` |
| `is reserved by the oracle` | `generator.py:41`, `api_generator.py:59`, `instruction_codegen.py:202`, `semantic_generator.py:29` |
| `is not supported by the oracle` | `oracle_visitor.py:239, :255, :261, :416`, `decorator.py:96` |

One condition, "this generator does not implement this feature", has seven phrasings and three exception types (`ValueError` in the backend, `NotImplementedError` in the oracle for the same kind of row, `TypeError` at `decorator.py:96`).
`NotImplementedError` is also used for spec errors (`oracle_visitor.py:255, :261`, O6).

## One condition, several wordings

| Condition | Wordings |
| --- | --- |
| Division by zero | `divides by zero` (`instruction.py:415`, `spec_checks.py:210, :292`); `takes a remainder by zero` (`spec_checks.py:255`); `an expression divides by zero with row=8, step='%i' at %i = 0` (`tests/test_constraints.py:154`) |
| Outside signed 64 bits | `is past 64 bits` (`instruction.py:406`); `which 64 bits cannot hold` (`spec_checks.py:218`); `past the signed 64-bit range` (`generic/utils.py:271`); `more bytes than 64 bits can count` (`lib.rs:119`) |
| State | `machine state` (`spec_checks.py:375`, `accelerator.py:47`), `state variable` (`accelerator.py:50`), glossary term "state variable" |
| Region | `[buffer, starts, sizes]` (`instruction.py:119`), `region`, `operand` (`spec_checks.py:308-312`), `storage` (C++), glossary term "region" and "data model" |
| Attribute kinds | `address attribute` (`spec_checks.py:203`) against glossary "addressing attribute"; `computation_attr` (`accelerator.py:76`) against "computational attribute" |
| Value in a semantics | `value` (front end), `tensor` (C++ `parser.cc:152`), `tile` (oracle comments), glossary "value" and "region value" |
| Semantics location | `semantics line L, column C` (front end), `its semantics` with no position (backend), `at i = v` (oracle) |
| Element type | `element type` (front end, backend), `dtype` (`decorator.py:79, :96`) |

## Prefixes and channels

| Layer | Prefix | Channel |
| --- | --- | --- |
| Front end | `{instruction}: ` | exception message |
| Backend Python | `{instruction}: its semantics ...` | exception message |
| Backend Rust | `Error: ` | stderr, exit 1; drops on stdout (`pipeline.rs:266`), warning on stdout (`:344`) |
| Backend C++ | none, `parse error: `, `create_instruction error: `, `live_range error: ` | `cerr` plus assert, or a reason string |
| Oracle | `{instruction}: `, `debug: `, `kernel {name!r}: ` | exception message; `UserWarning` for 64-bit |
| Shell | `Error: `, `Unknown flag`, `Unknown arg`, `Unknown argument`, `Aborting.` | stderr, one script on stdout (`act/tutorials/asplos26/docker.sh:43`) |

## Rule that fixes it

- One verb in prose: "reject".
- One phrase in a message for an unsupported feature: `{feature} is unsupported by {generator}` with a spec section.
- One phrase for a run-time violation: `{condition} violates {rule} (S<id>)`, raised as class `V`; v3 has no undefined behavior (owner ruling 16).
