# Debate round two

The five points round one left contested, argued once more and ruled.
Each ruling names the item it changes and the evidence it rests on.

## Selection tests in `test_matching.py`

- Pragmatist: these 34 tests found real selection bugs; a differential test needs a cargo build per case and both generators in one image, so it is slow and lives in a repo the backend team does not run on every PR.
- Purist: the spec promises a result, not a selection; `test_matching.py:648-661` fails on a correct cheaper selection, which is a false alarm the suite trains people to ignore.
- Ruling: split by what the assertion reads; a test whose fact is visible in the result (`:898`, `:1247`, `:1177-1189`) becomes a T13 case, a test whose fact is only visible in the search log (`:661`, `:709`) stays as an impl test with a header.
- Ruling: T12 gets a fast tier so the backend repo still runs every corpus case through `spec_checks` on each PR without a build; the slow differential tier runs in act nightly and on release.
- Binds: T12, T13, T18, T20.

## Error messages carry clause ids

- Pragmatist: matching prose regexes works today in both repos (`act-oracle/tests/test_refusals.py:62`, `act-backend/tests/test_refusals.py:156`); changing every message is a breaking change for anyone parsing errors.
- Purist: a rewording fails many cases at once, and a user reading `[S9.4]` can open the spec; the owner allows one coordinated major release, so the cost is paid once.
- Ruling: ids in the v3 release, as a suffix the runner matches on, with the prose still asserted in a handful of cases so the message stays readable; no shim for the old text.
- Binds: T6, T12, T22 question 2; both items are marked breaking under that condition.

## Case format

- Pragmatist: `Accelerator` scripts are the only entry point (`taidl/tests/test_instruction.py:10`) and need no new parser.
- Purist: a text format would let L write cases in the spec itself and would make the corpus readable without Python.
- Ruling: Python scripts now, with `semantics.txt` beside them for the text-level sections (S2, S3, S7, S8), so the lexer and parser cases are already text; a text format for a whole ISA description waits for L to define one, and the loader is the single place to extend.
- Binds: T5.

## Where the shared kit lives

- Pragmatist: act installs both generators (`act/tests/test_end_to_end.py:11`), so a kit that imports an oracle and a backend can only be tested there.
- Purist: corpus loading and tolerance tables are language-level and belong to the taidl wheel, or each generator depends on act, which inverts the dependency graph.
- Ruling: two small pieces; `taidl.testing` ships the corpus loader and per-element-type tolerance helpers with the language, `act/testkit/` holds the golden, oracle import, and reference helpers that need both generators.
- Binds: T3, T16, T22 question 1 (the owner can still collapse the two).

## Examples trees in generator repos

- Pragmatist: a generator PR must build its own example; moving `examples/` away makes the generator repo depend on act for its smoke test.
- Purist: three copies of `mini_trn` (`taidl/examples/`, `act-backend/examples/`, `act-oracle/examples/`) drift, and the goldens are already duplicated.
- Ruling: the ISA description scripts live in taidl and are listed by T10; each generator keeps the kernels and one smoke test that generates the pinned description; data and goldens move to act with T14 and T15.
- Binds: T10, T14, T15, T22 question 4.
