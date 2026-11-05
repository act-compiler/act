# Plan: terminology and glossary rollout

The glossary (S10) fixes one term per concept and bans its synonyms.
This plan rolls it out across the four repos, so docs, error messages, and generator tables use the same words.
Code renames come last, since B and O read the front end's attribute names.
Sizes assume two engineers; every item marked breaking ships together in the TAIDL v3 release with a migration note.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| L25 | Reconcile the glossary with the terminology reference | L1 | S | Q1 |
| L26 | Apply the glossary to the taidl docs and error messages | L25 | M | Q1 |
| L27 | Apply the glossary to B, O, and A docs | L25, B, O, A | M | Q1 |
| L28 | Banned-synonym lint | L26 | S | Q2 |

## L25 Reconcile the glossary with the terminology reference

- What: align each glossary row with the terminology track's reference (MLIR, StableHLO, PyTorch, NKI columns).
- Changes: `docs/glossary.md`; terms that agree are kept.
- Breaking: no.
- Benefits: terms match what the community already reads.
- Caveats: API names (`add_data_model`, `computation_attr`) fix some terms until L15.
- Fail points: two terms for one concept survive; notice by L28.
- Order and dependencies: first; disagreements are recorded as D entries.
- Verification: every concept in the reference has one glossary row or a recorded disagreement.

## L26 Apply the glossary to the taidl docs and error messages

- What: replace banned synonyms in `taidl/docs/` and in `ValueError` and `SyntaxError` texts, such as "buffer" and "refuse".
- Changes: `docs/*.md` and error texts in `taidl/*.py`; message structure kept.
- Breaking: yes for tools that match error text.
- Benefits: an error names the concept the spec defines.
- Caveats: error text changes break tests that match messages.
- Fail points: a test pins old text; notice by the T suite.
- Order and dependencies: after L25; T updates expected messages.
- Verification: L28 lint is clean on taidl.

## L27 Apply the glossary to B, O, and A docs

- What: B, O, and A tutorials use the canonical terms and cite section ids.
- Changes: B, O, A docs; owned by those tracks.
- Breaking: no.
- Benefits: one vocabulary across the ecosystem.
- Caveats: owned by other tracks.
- Fail points: a tutorial keeps "buffer"; notice by L28.
- Order and dependencies: after L25; B, O, A own it.
- Verification: L28 lint is clean on all four repos.

## L28 Banned-synonym lint

- What: a script that greps docs for each banned synonym in the glossary table.
- Changes: `tests/` gains a lint; no doc changes.
- Breaking: no.
- Benefits: terminology cannot drift back.
- Caveats: some banned words are fine in code or in other senses; needs an allow list per file.
- Fail points: false positives make people ignore it; notice by the allow list growing.
- Order and dependencies: after L26; T runs it in CI.
- Verification: the script fails on a deliberate "buffer" in a doc.
