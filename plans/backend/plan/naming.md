# Naming and terminology cleanup

Each rename is one row of the track's naming table (old | new | where), aligned to the spec glossary.
Error messages are part of the user interface and are pinned by tests, so the message rename is breaking and rides the major release.
Identifier renames inside the crate are not breaking.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| B27 | Glossary and owner vocabulary in messages, docs and Python names | L25, L27 | S | Q1 |
| B28 | Rust and C++ identifier and log renames | B1 | S | Q1 |

## B27 Glossary terms in messages, docs and Python names

- What: reject for refuse, addressing attribute for address attribute, state for machine state, region for operand, buffer for a declared memory, `is_supported_constraint` for `honoured`, element type for dtype.
- What: the size and constraint errors for `@s.` name the state, as the start and semantics errors do (`spec_checks.py:287-290,364-367`).
- Preserves: message structure (instruction name first); only words change.
- Benefits: messages use the spec's words; grep finds one term per concept.
- Caveats: `tests/test_refusals.py` and `tests/test_names.py` assert message text; T must update together.
- Fail points: a message changed in code but not in its test; noticed by the test.
- Order and dependencies: after L25; L27 applies the glossary to B docs in the same release; with T.
- Verification: `grep -rn -i 'refus\|honou\|address attribute\|machine state\|initial state\|tensor buffer' act_backend tests` empty.
- Breaking: yes (message text).

## B28 Rust and C++ identifier and log renames

- What: the Rust and C++ rows of the naming table (`TensorInfo` renamed, `Buffer` and `hbm` kept), plus log labels such as "Starting Phase 1 Module 3: Graph Extractor", which become instruction selection, extraction and memory allocation (`pipeline.rs:229`) and "Info:" (`pipeline.rs:521,537,548`).
- Preserves: behavior; `--verbose` content except labels.
- Benefits: names say what a thing is; no stage numbering a reader must decode.
- Caveats: `--verbose` output is user-visible; treat as non-breaking diagnostics unless the owner says otherwise.
- Fail points: a rename that collides with a generated name; noticed by the build.
- Order and dependencies: with B1, so each file moves and renames once.
- Verification: `cargo clippy` clean; B3 gate identical.
