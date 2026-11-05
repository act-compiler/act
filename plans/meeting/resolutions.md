# Resolutions of owner items

Every item a panel decision marked "owner", "confirm", or escalated, resolved by an owner decision (OD) or on the evidence.
Items on overlap, in-place, and placement marks go to panel 9; items on the data-fault and accumulation-bound split go to the joint moderator of panels 1 and 5.
Code paths are relative to /home/user.

| Item (panel, question) | Resolution | Basis (owner decision or evidence) | Critical for owner? |
| ---------------------- | ---------- | ---------------------------------- | ------------------- |
| Q2.12: name of the user's file | "ISA specification"; the panel default "ISA description" is replaced everywhere, including Q8.8's trial wording. | OD32. | No |
| Q2.1 (Sonnet): may a tied pair spell its input and output starts differently | Yes; a start is any expression over attributes, and the mark expands to the constraint that the evaluated starts are equal, violated like any constraint. | OD30, OD34; the other Q2.1 escalations (removal, unmarked overlap) go to panel 9, and OD22 and OD30 keep `set_inplace`. | No |
| Q3.6: does decision 16 cover the backend's unsupported-table assumption | No; the `merge` panic becomes the shared coded shape diagnostic at selection, reported as a rejection at kernel compile time. | act-backend/act_backend/generic/src/ir/egraph.rs:403-409 is a bare `panic!`, which is neither a defined error nor a defined behavior; Q7.8 names compile-time outcomes "reject"; Q1.12 and Q6.3 say the backend never faults. | No |
| Q3.8: clause id on line 1 or line 3 | Line 1 ends with `[S<id>]` (option A); line 3 lists further sections only. | R44 rules a suffix; act/plans/diagnostics/scheme.md:50-51 already puts it on line 1 for CI logs, `grep`, and `str(error)`; scheme.md:59 leaves room within 160 characters; review SD7 asks for this form. | No |
| Q3.8: namespace | Spec section id, as the panel ruled (option D). | Panel 3 decision; no owner input needed. | No |
| Q4.2: wall-clock search mode as user default | The wall-clock bound stays the user default; tests use the deterministic mode. | OD27. | No |
| Q4.9: integer norm the backend follows | Every intermediate is a checked signed 64-bit integer in both executors. | Q1.2 option B and Q5.3 option B agree; no conflict for a joint moderator. | No |
| Q4.11: cbindgen for the C header | Approved; the C header is derived from the Rust struct by cbindgen, as Q4.11 option A ruled. | OD28. | No |
| Q4.12: format generated crates | rustfmt runs on the generated crate at generation time; option B replaces option A. | OD29. | No |
| Q5.1 and Q8.6 (R28): fail point of the oracle unroll spike | Fixed before the spike, as listed in the next four rows; a failed gate leads to fallback C with an equivalence test. | Panel 5 decision on option D; panel 8 Q8.6 asks for the bound before O19. | No |
| Q5.1: stress-kernel shape | Two nested loops of 64 by 64 trips (4,096 trips) around 4 instructions (16,384 issues). | A 1024-square matmul tiled by 16 needs 64 by 64 tiles, the tiling of act-oracle/tests/test_kernel.py:400-405; the largest test trip count today is 8. | No |
| Q5.1: compile-time limit | Trace plus XLA compile of the stress kernel within 60 s on the CI CPU image. | Round-2 Sonnet, Opus, and Fable accept a trip count plus a CI time; 60 s keeps the spike inside one CI step. | No |
| Q5.1: relative limit | Every existing loop test compiles within 2x its time on the current `while` path, measured before the spike. | Round-1 Opus bound and round-2 Sonnet's two-part bound. | No |
| Q5.1: per-instruction op-count cap | 2x the largest per-instruction op count among the current tests and examples, measured before the spike; above it the oracle rejects with a coded coverage limit. | Q5.1 step 5 (a coverage limit, not a spec violation); OD16. | No |
| Q5.8: shape-module delivery | A copy at generation stamped with the taidl version (option C). | OD25; Q3.7 already rules the same. | No |
| Q6.6: StableHLO interpreter role | No role, neither cross-check nor regenerator; stored data comes from NumPy or closed forms. | OD28 ("the StableHLO interpreter is not added"); act-oracle/tests/test_reduce.py:87-129 shows NumPy references exist. | No |
| Q6.10: fuzzing, mutation, property tests | Approve all three: Hypothesis per PR with `derandomize`, the nightly grammar fuzzer once the Q6.5 runner exists, and a per-release mutmut pilot. | OD28 allows hypothesis and mutmut; the fuzzer needs no new tool, only the Q6.5 runner (panel-6 round1-sonnet:153). | No |
| Q7.3: act-backend version | 2.0. | OD26; act-backend/pyproject.toml:7 reads 1.16, so 2.0 is its next major. | No |
| Q7.6: singular of "state" | "state variable" is unbanned, as the panel ruled. | Shipped docstrings, messages, and docs use it (panel 7 Q7.6); OD19 bans only "initial state". | No |
| Q8.5: pre-release tags under decision 10 | No public pre-release tag and no public commit pin; generators pin taidl pre-release tags inside the private development repos, and the public v3 release pins released versions. | OD33 settles it with OD10; A16 stays in wave 1. | No |
| Q8.8: cold-start trial parameters | Two people outside the team, each alone, fresh environment, the pinned public release, README and tutorial only, no maintainer help. | Panel 8 Q8.8 default; round-2 Fable and Opus adopt it. | No |
| Q8.8: trial task and time limit | Each person installs, runs the tutorial, and changes one small ISA specification (the Q7.5 example, about 37 lines) within 4 hours. | One working half-day fits the Q7.5 example size; every blocker becomes an item. | No |
| Q8.8: fallback | One outside person, or a non-author teammate with a recorded transcript, stated as the weaker form. | panel-8 round1-sonnet:135. | No |
| Q8.9: capacity against "nothing is cut" | Dates move at the M1 rebaseline and scope does not; measure the assist factor in wave 1. | OD11 with the owner's standing direction in panel 8. | No |

## Not resolved here

- Q1.3 (Opus), Q2.1 (Opus, Fable), Q5.6, Q6.2 moderator note, Q6 Sonnet overlap norm, Q7.9: in-place and overlap items, for panel 9; OD24 already rejects every cross-buffer pair.
- Q1.5, Q1.9, and Q5.2 split: for the joint moderator of panels 1 and 5 (OD36, OD37).

## Critical for the owner

- None; every item is settled by an owner decision or by evidence within scope, capacity, and the release commitments OD10, OD26, and OD33.
