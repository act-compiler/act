# Diagnostics track

Catalog of every user-facing diagnostic in `taidl`, `act`, `act-backend` and `act-oracle`, one scheme for all of them, and the plan to adopt it in TAIDL v3.

| File | What it holds |
| --- | --- |
| [catalog/taidl-act.md](catalog/taidl-act.md) | Every diagnostic of the front end and of the act umbrella, with class |
| [catalog/act-backend.md](catalog/act-backend.md) | Every diagnostic of the backend generator, the Rust reader and search, and the C++ allocator |
| [catalog/act-oracle.md](catalog/act-oracle.md) | Every diagnostic of the oracle generator and of the generated oracle at a call |
| [analysis/duplicated.md](analysis/duplicated.md) | Conditions checked in both the front end and a generator |
| [analysis/late.md](analysis/late.md) | Checks done late in a generator that belong in the front end |
| [analysis/locations.md](analysis/locations.md) | Messages that name no instruction or source location |
| [analysis/wording.md](analysis/wording.md) | Several wordings for one condition, banned words, prefixes and channels |
| [analysis/asserts.md](analysis/asserts.md) | Asserts and panics used where an error belongs |
| [analysis/fallbacks.md](analysis/fallbacks.md) | Silent fallbacks where an error belongs |
| [review.md](review.md) | What works and should stay |
| [bugs.md](bugs.md) | Diagnostic bugs E1 to E24 with severity and fix sketch |
| [scheme.md](scheme.md) | The diagnostic scheme: classes (Reject, Unsupported, Fault, Input, Internal, Environment), codes, two-line format, wording, layers, exceptions, removed names |
| [failure-policy.md](failure-policy.md) | Decision 40 mapped onto the classes: allowed stops per tool, candidate drops, the input kernel class |
| [registry-seed.md](registry-seed.md) | First codes mapped from the catalog to TAIDL language specification sections |
| [plan/README.md](plan/README.md) | Plan index table, staffing and critical path |
| [plan/D01-registry-and-format.md](plan/D01-registry-and-format.md) | D1 registry, classes and message format |
| [plan/D02-call-site-and-exceptions.md](plan/D02-call-site-and-exceptions.md) | D2 call sites and exception classes |
| [plan/D03-recode-front-end.md](plan/D03-recode-front-end.md) | D3 recode and reword the front-end messages |
| [plan/D04-move-checks-to-front-end.md](plan/D04-move-checks-to-front-end.md) | D4 move checks into the front end |
| [plan/D05-delete-generator-copies.md](plan/D05-delete-generator-copies.md) | D5 delete duplicated generator checks |
| [plan/D06-unsupported-registries.md](plan/D06-unsupported-registries.md) | D6 unsupported-feature registries |
| [plan/D07-backend-runtime.md](plan/D07-backend-runtime.md) | D7 backend Rust and C++ diagnostics |
| [plan/D08-oracle-runtime.md](plan/D08-oracle-runtime.md) | D8 oracle run-time and call diagnostics |
| [plan/D09-silent-fallbacks.md](plan/D09-silent-fallbacks.md) | D9 silent fallbacks become errors |
| [plan/D10-tests-and-lint.md](plan/D10-tests-and-lint.md) | D10 registry coverage test and wording lint |
| [plan/D11-generated-docs.md](plan/D11-generated-docs.md) | D11 generate S9 and unsupported tables |
| [plan/D12-act-scripts.md](plan/D12-act-scripts.md) | D12 act umbrella scripts and entry points |
| [plan/D13-json-and-explain.md](plan/D13-json-and-explain.md) | D13 machine-readable output and code explanations |
| [plan/D14-run-time-norm.md](plan/D14-run-time-norm.md) | D14 fault table and run-time norm |
| [plan/D15-removed-names.md](plan/D15-removed-names.md) | D15 removed-names rejection |
| [plan/D16-failure-policy.md](plan/D16-failure-policy.md) | D16 failure policy |
