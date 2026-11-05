# Decisions: selection tests, diagnostics, case format, kit, examples

Each decision names the item it changes and the evidence it rests on.

## Selection tests in `test_matching.py`

- Split by what the assertion reads.
- A test whose fact is visible in the result (`:898`, `:1247`, `:1177-1189`) becomes a T13 case.
- A test whose fact is only visible in the search log (`:661`, `:709`) stays as an impl test with a header.
- A selection assertion such as `:648-661` fails on a correct cheaper selection, which the suite must not do.
- T12 has a fast tier running every corpus case through `spec_checks` on each pull request without a build, and a slow tier per pull request (T27).
- The differential tier runs in act nightly and on release.
- Binds: T12, T13, T18, T20, T27.

## Diagnostics in cases

- Cases match on `code` and `object`, never on message prose (Q3.9).
- A rejected case also records `phase` and the clause id; a `run` fault matches on `code`, since a C or C++ oracle has no Python class (Q6.5).
- The clause id sits on line 1 of a message until the owner decides otherwise (Q3.8).
- A few text tests cover the message format; D owns them.
- No shim for the old message text.
- Binds: T5, T6, T12.

## Case format

- Python scripts for the ISA specification, with `semantics.txt` for the text-level sections (S2, S3, S7, S8).
- `program.toml` for instruction-level cases, executor-neutral so a non-Python oracle can run it (Q6.1).
- A text format for a whole ISA specification waits for L; the loader is the single place to extend.
- Binds: T5, T29.

## Where the shared kit lives

- `taidl.testing` ships the corpus loader and the tolerance implementation with the language.
- `act/testkit/` holds the golden, oracle import, and reference helpers that need both generators (Q6.11).
- Binds: T3, T16.

## Examples trees in generator repos

- The ISA specification scripts live in taidl and are listed by T10.
- Each generator keeps the kernels and one smoke test that generates the pinned specification.
- Data and goldens move to act with T14 and T15.
- Binds: T10, T14, T15.
