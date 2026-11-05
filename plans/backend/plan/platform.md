# Long-term platform: importer, dialect, torch adapter, C++ driver

Owner decisions 43 and 44 move the backend to C++ on MLIR infrastructure with a PyTorch backend on it; panel 12 sets the order (Q12.1, Q12.2).
The front end, `check`, and every generator stay Python; generators emit C++ as they emit Rust today; no `Model` is serialized.
MLIR code lives only in the importer, the dialect, and the bridge; the e-graph library (`egraph.md`) and the allocator include no MLIR header.
The supported MLIR input is a StableHLO portable artifact at U15's version; text input serves tests within that one version.
Every step passes the B3 gate in B29's deterministic mode.

| ID | Item | Depends on | Size (S/M/L) | Quarter |
| --- | --- | --- | --- | --- |
| B47 | StableHLO importer: C++ tool reading portable artifacts, printing today's kernel text, coded errors | B42, B43, U15 | M | later |
| B48 | Dialect spike: generic instruction op against per-ISA ODS, per-ISA build cost measured | B13, U15 | S | later |
| B50 | ISA dialect generator per B48: typed attributes, checked verifier, semantics region from O3 | B48, U10, O3 | M | later |
| B53 | Torch adapter: `ExportedProgram` to portable artifact; pairing check picks torch-mlir or the walker; strict | B47, B42 | M | later |
| B54 | Typed import: importer builds library terms, chosen program as dialect ops, text bridge deleted | B47, B50, B52 | M | later |
| B55 | XLA HLO text reader in C++, holding every message of `tests/test_reader.py` | B54 | M | later |
| B56 | One C++ driver: importers, engine, extraction, allocator, cost, exclusion report; Rust crate deleted | B54, B55, B46 | L | later |

## Order

- Wave 8: B47 and B48; B49 (`egraph.md`) runs in parallel.
- Wave 9: B50, B53, and the engine swap B52 (`egraph.md`).
- Wave 10: B54, then B55, then B56.
- XLA HLO text stays an input (owner decision 40) and gets its C++ reader (B55) before the Rust crate is deleted.

## B47 StableHLO importer

- What: a C++ tool reads a StableHLO portable artifact and prints the kernel text today's reader (B5) accepts; no Rust changes.
- What: an artifact outside U15's version window, or an operation outside the input specification (B42), stops with a coded error naming it.
- What: StableHLO text is accepted for tests within U15's one pinned version.
- Preserves: the Rust reader, its messages, and every B3 golden.
- Benefits: StableHLO input works before any engine change; only portable artifacts carry StableHLO's backward and forward compatibility guarantees.
- Caveats: MLIR has no stable C++ API, so the importer builds only against U15's pin; the text bridge is retired by B54.
- Fail points: an operation mapped to the wrong HLO form; noticed by T39's differential run against the HLO path.
- Order and dependencies: after B42 and B43 (detection) and U15 (prebuilt MLIR and StableHLO libraries).
- Verification: each T39 artifact and its HLO form compile to equal assembly; an artifact at another version names its version.

## B48 Dialect spike

- What: build one ISA's dialect twice: one generic instruction op reading a table the backend generator writes, and per-ISA ODS through `mlir-tblgen`.
- What: measure each form's per-ISA build time and install footprint on the architect's machine.
- What: the generic op is the default; per-ISA ODS replaces it only if it builds from U15's prebuilt package with no further install.
- Benefits: the op form is chosen on measurement before B50 commits to one.
- Caveats: selection and allocation run outside MLIR, so no consumer needs per-ISA op classes.
- Fail points: a measurement on a machine with MLIR already installed; avoided by running in a fresh U15 image.
- Order and dependencies: after B13 (instruction model) and U15.
- Verification: a short record of both build times and the form chosen, read by B50.

## B50 ISA dialect generator

- What: the backend generator emits the dialect in B48's form: typed attributes per instruction and each instruction's StableHLO semantics region from O3's generic emitter.
- What: the verifier reuses the e-graph library's guard evaluator (B51), so checked guards have one implementation.
- What: the table is generator output read from the checked `Model` (U10), never the `Model` itself.
- Preserves: the generated Rust crate, unchanged until B56.
- Benefits: the chosen program has a typed MLIR form; the oracle and the dialect read one StableHLO source (O3).
- Caveats: under per-ISA ODS, `mlir-tblgen` and MLIR headers join every ISA build.
- Fail points: a verifier that accepts what a guard rejects; noticed by a test that runs both on each B8 near miss.
- Order and dependencies: after B48, U10, and O3.
- Verification: each example ISA specification generates a dialect that parses, verifies, and round-trips each instruction.

## B53 Torch adapter

- What: one adapter package runs `torch.export` with core ATen decompositions, sets `dynamic=False`, and hands B47 a portable artifact at U15's version.
- What: its first step is the pairing check: the latest released torch-mlir imports an `ExportedProgram` from the latest released torch and covers B42's input specification.
- What: on pass, torch-mlir is the producer, and torch-mlir input files enter through the same package.
- What: on fail, an ACT-owned walker maps each core ATen node in the input specification to StableHLO and stops with a coded error naming any other node; it is also the torch.fx reader.
- What: strict by default under owner decision 40; a node outside the input specification stops with a coded error, never a silent fallback.
- Benefits: torch inputs reach the backend through one producer behind one interface; its MLIR never links into ACT.
- Caveats: torch-mlir is a new dependency, asked when B53 lands (owner decision 28); vendoring it is out under owner decision 33; one large FX import has taken over 20 minutes.
- Fail points: a torch-mlir release that lowers a node differently from the walker's mapping; noticed by T40 against eager PyTorch.
- Order and dependencies: after B47 and B42; before A29.
- Verification: the pairing check result is recorded; each B42 operation exported from torch compiles; an unsupported node names itself.

## B54 Typed import

- What: the importer builds e-graph library terms directly, replacing the kernel text bridge; the chosen program is emitted as dialect ops (B50).
- Preserves: the reader's error kinds and the B3 goldens.
- Benefits: no text round trip between import and search.
- Caveats: the HLO text path still uses the Rust reader until B55.
- Fail points: a payload field lost between artifact and term; noticed by the B3 gate on T39's corpus.
- Order and dependencies: after B47, B50, and B52 (library engine in place).
- Verification: B3 gate identical; the text bridge is deleted.

## B55 XLA HLO text reader in C++

- What: a C++ reader of XLA HLO text builds library terms, as B54 does for StableHLO.
- Preserves: every message `tests/test_reader.py` checks.
- Benefits: HLO text stays an input (owner decision 40) once the Rust reader goes.
- Caveats: MLIR parses StableHLO but not HLO text, so this is a hand-written reader.
- Fail points: a message reworded; noticed by `tests/test_reader.py`.
- Order and dependencies: after B54.
- Verification: `tests/test_reader.py` passes on the C++ reader; B3 gate identical.

## B56 One C++ driver

- What: one C++ driver runs the importers, the engine, extraction, the allocator, cost, and the B46 exclusion report in one process; the Rust crate is deleted.
- Preserves: the CLI, the stop reasons, `--log` files, and the exclusion report.
- Benefits: one language in the generated backend; no FFI boundary.
- Caveats: extraction moves with the driver; it stays outside the e-graph library.
- Fail points: a CLI flag lost in the port; noticed by T41 running B3 on the driver.
- Order and dependencies: after B54, B55, and B46.
- Verification: T41 passes; no Rust source in the generated crate.
