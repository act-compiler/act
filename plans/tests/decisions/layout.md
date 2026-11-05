# Decisions: location, existing tests, goldens, timing

| topic | decision | binds |
| --- | --- | --- |
| corpus location | one corpus in the taidl wheel, generators read it through the pin; a fifth repo is not needed until a second implementation outside the four repos exists | T19 |
| existing rejection tests | word and type loops become expectation rows; one-off rejections stay until a corpus case exists | T12 |
| `test_matching.py` | split by what the assertion reads (see cases.md) | T13, T18 |
| `test_extraction.py`, `test_allocator.py` | stay as impl tests; they guard invariants no result check sees | T20 |
| CLI and packaging tests | stay; they are not language tests and are not labelled as such | T20 |
| layout tests in `test_generators.py` | deleted; rustfmt runs at generation and `conftest.py:468-472` checks every generated tree | T4 |
| goldens | one in act, one record format, one tolerance beside the golden | T15 |
| property-based tests | per pull request with a fixed seed, after the corpus | T9, T24 |
| mutation pilot | once per release, never a pull request gate | T25 |
| error messages | cases match on the diagnostic `code` and `object` (see cases.md) | T6, T12 |
| case format | Python files in the authoring surface plus `case.toml` and `program.toml` (see cases.md) | T5 |
| shared kit | split between `taidl.testing` and `act/testkit/` | T3 |
| examples trees in generator repos | scripts live in taidl; each generator keeps kernels and one smoke test | T10, T14 |
