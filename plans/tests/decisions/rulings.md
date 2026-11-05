# Rulings as plan items

The three rulings that bind other items, written as items so they carry a size, a quarter, and a verification.

## T19. Decision: corpus lives in taidl, consumed through the pin

- What: `taidl/conformance/` ships in the taidl wheel; generators run the corpus of the taidl they pin.
- Benefits: the language and its tests version together; a generator cannot claim a version it does not test against.
- Caveats: a corpus fix needs a taidl release before a generator sees it; the override path in T16 covers local work.
- Fail points: a generator pins an old taidl to avoid new cases; `check_versions.py` and the release checklist name the pin.
- Order and dependencies: binds T5 and T16.
- Verification: the owner confirms, or names the alternative (a fifth repo for the corpus).
- Breaking: no.
- Files: binds T5 and T16 only.

## T20. Decision: impl tests survive with a header, PR-defense tests do not

- What: a test that guards an invariant no result check can see (extraction order, allocator drops, log directories) stays and its file header says `impl`; a test whose fact is a corpus case is deleted when the case runs in CI.
- Benefits: the suite says what each file is for; nobody mistakes an impl test for a language guarantee.
- Caveats: the line between the two is a judgement per test, made in the T18 PRs.
- Fail points: an impl test pins text, as `test_mini_trn.py:72` does; the rule is that an impl test reads behaviour (logs, exit codes, dropped candidates), never generated source text.
- Order and dependencies: binds T4 and T18.
- Verification: a header per test file; a grep for `read_text()` on generated sources in tests finds only impl files.
- Breaking: no.
- Files: binds T4 and T18 only.

## T21. Decision: expectations table is the source, coverage.md is rendered

- What: a generator's unsupported-feature documentation is rendered from `tests/expectations.toml`, never hand-edited.
- Benefits: the owner's request that tests follow from the docs holds in the other direction too; the two cannot disagree.
- Caveats: the clause prose per row still has to be written by B and O; the renderer only places it.
- Fail points: someone edits coverage.md by hand; CI compares the rendered file and fails.
- Order and dependencies: binds T11.
- Verification: the owner confirms the direction (table renders doc) over the alternative (doc is parsed into a table).
- Breaking: no.
- Files: binds T11 only.
