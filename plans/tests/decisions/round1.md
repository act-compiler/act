# Debate round one

Each topic has the pragmatist position, the purist position, and the ruling with the item it binds.
Topics marked contested went to round two.

| topic | pragmatist | purist | ruling |
| --- | --- | --- | --- |
| corpus location | each generator keeps its own cases | one corpus in taidl, generators read it | purist (T19); the three copies of the word list show the cost of copies |
| existing rejection tests | keep `test_refusals.py` files as they are | replace with expectation rows | purist for word and type loops, pragmatist for one-off rejections until a corpus case exists (T12) |
| `test_matching.py` | keep; it caught real bugs | delete; selection is an implementation detail | contested; round two |
| `test_extraction.py`, `test_allocator.py` | keep | delete as implementation | pragmatist (T20); they guard invariants no result check sees |
| CLI and packaging tests | keep | out of scope | pragmatist; they are not spec tests and are not pretended to be |
| layout tests in `test_generators.py` | keep | delete | purist (T4); the formatter check in `conftest.py:468-472` already covers them |
| goldens | keep per repo | one in act | purist (T15); the record format is already the same code twice |
| property-based tests | not now | from the start | pragmatist on timing (Q2, T9); the corpus first |
| error messages carry clause ids | no, match prose | yes, match ids | contested; round two |
| case format | Python `Accelerator` scripts | a text format the spec defines | contested; round two |
| where the shared kit lives | act | taidl | contested; round two |
| examples trees in generator repos | keep | move to taidl | contested; round two |
