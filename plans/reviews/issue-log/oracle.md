# Issue log: oracle

Sources are `reviews/oracle/verification.md`, `architect.md`, and `critic.md`.
The dispositions come from `oracle/review-response.md` and were checked against the oracle plan items and panel 5.
Rows of that file that answer tech-lead rulings (R28 to R62) are not review findings; they appear in rulings.md.
Findings are keyed VER (verification), CRI (critic), and ARC (architect).
Panel questions are cited as Qn.k and owner decisions as ODn; D and O ids are oracle plan items.

## Findings

| Finding | Severity | Disposition | Where it landed |
| --- | --- | --- | --- |
| VER-1: C6 incomplete, a data model after an instruction is accepted | minor | fixed | oracle/coverage-table.md (partial row, L11) |
| VER-2: C14 overclaim, the overflow warning is narrower than stated | major | fixed | oracle/coverage-table.md; D23; O24 |
| VER-3: C16 shallow, `ROOT` is a front-end rule | minor | fixed | oracle/coverage-table.md |
| VER-4: C28 unverifiable, no XLA comparison run | minor | fixed | oracle/coverage-table.md (partial); O13 |
| VER-5: D4 GPU part unverifiable | major | fixed | O9 (measure first) |
| VER-6: D10 premise stale after OD17 | major | fixed | D10; O10 |
| VER-7: D11 divergence shown by reading only | minor | fixed | D11 |
| VER-8: D13 path unreachable | minor | fixed | D13; O5 |
| VER-9: D16 "latent" unverifiable | minor | fixed | D16 |
| VER-10: D17 wrong, only variables a value reads are enumerated | major | fixed | D17 |
| VER-11: D18 fix not free, ruff check runs in tests | minor | fixed | D18; O18 |
| VER-12: an integer literal above 2^53 rounds twice | major | fixed | D21; O8 |
| VER-13: S1.2 status vocabulary has no `supported` | minor | fixed in part: the planning table keeps `supported` for review, the shipped table keeps `partial` and `rejected` | oracle/coverage-table.md; O17 |
| VER-14: `bitcast_convert` byte order missing | major | fixed | oracle/coverage-table.md; O22; Q5.4 |
| VER-15: float `convert` overflow missing | major | fixed | oracle/coverage-table.md; O22; Q1.8 |
| VER-16: initial state outside 64 bits missing | major | fixed | oracle/coverage-table.md; D23; O24 |
| VER-17: overlapping output regions missing | major | fixed | oracle/coverage-table.md; D22; O10; Q9.1 |
| VER-18: `@a.` outside a start or constraint missing | minor | fixed | oracle/coverage-table.md |
| VER-19: constraint arithmetic overflow contradicts S6.2 | major | fixed | oracle/coverage-table.md; O24 |
| VER-20: computed divisor of 0 missing | minor | fixed | oracle/coverage-table.md |
| VER-21: region value type or count mismatch missing | minor | fixed | oracle/coverage-table.md; O7 |
| VER-22: `constant` operand forms missing | minor | fixed | oracle/coverage-table.md |
| VER-23: compare `type` rule missing | minor | fixed | oracle/coverage-table.md; O7 |
| VER-24: `dot`, `broadcast`, `pad` forms missing | minor | fixed | oracle/coverage-table.md |
| VER-25: `add_constraints`, `update=`, `constraints=` missing | minor | fixed | oracle/coverage-table.md |
| CRI-1: fidelity rests on XLA where StableHLO is silent | major | fixed | O22; oracle/coverage-table.md |
| CRI-2: small fidelity fixes should land before structure | major | fixed | oracle/plan/overhaul.md; O8, O21, O19 |
| CRI-3: wrong premise, tests compare with NumPy and not jax | major | fixed | oracle/plan/decisions.md; O13 |
| CRI-4: O3 conditional on O2, one function per instruction | major | fixed | O3 |
| CRI-5: O8 emit the decimal verbatim if MLIR reads it exactly | major | fixed in part: hex bits from `Fraction` work whatever the pinned jaxlib does | O8; Q5.5 |
| CRI-6: O9 decide the norm first, the CPU is the reference | major | fixed | O9, O14 |
| CRI-7: O10 settled by OD17 | major | fixed in part: Q9.2 and Q9.3 add the constraint list, the default, and the pre-state value | O10 |
| CRI-8: O11 fold D19 and D20 | minor | fixed | O11 |
| CRI-9: O12 test whether x64 is needed | minor | fixed | O12 |
| CRI-10: O13 smaller, NumPy from tests, `Fraction` literals | major | fixed | O13; Q5.7 |
| CRI-11: step order, norms and fixes ahead of O1 | major | fixed | oracle/plan/overhaul.md |
| CRI-12: keep the enumeration ruling | major | superseded by Q5.1: every loop is unrolled; the lowering half is kept | oracle/plan/decisions.md; O19 |
| CRI-13: `xla_allow_excess_precision` not handled | major | fixed | D24; O21 |
| CRI-14: numerics norm table owned by L | major | fixed | O22; Q5.4; language item L31 |
| CRI-15: `dot` relies on XLA upcast, test asserts single rounding | major | fixed | O27, O22; Q5.2 |
| CRI-16: in-module detection of data-dependent errors | major | superseded by the joint decision of panels 1 and 5, point A: tensor data never faults | O23 |
| CRI-17: no normative-platform statement | minor | fixed | O14 |
| CRI-18: version pin as the fidelity contract | minor | fixed | O13 |
| CRI-19: placement marks checked over every iteration | minor | rejected: already done, marks append constraints checked per iteration (`instruction_codegen.py:36`) | none |
| CRI-20: literal rounding is a spec gap | major | fixed | O8, depends on L16; Q1.10 |
| ARC-1: unroll every loop, spike first | major | fixed | O19, O20; Q5.1 |
| ARC-2: runtime code out of `API_FILE` first | major | fixed | O1 |
| ARC-3: one thin function per instruction, one record | major | fixed | O2, O3; Q5.9 |
| ARC-4: a generic form for all words | major | fixed | O3 |
| ARC-5: language columns of the word table to the front end | major | fixed | O3, O6, O7 |
| ARC-6: a StableHLO interpreter over NumPy | major | superseded by Q5.7 and OD28 (NumPy references, no interpreter) | O13 |
| ARC-7: split `utils.py` after deleting the loop algebra | minor | fixed | O1 after O20 |
| ARC-8: O3 earlier, size M | minor | fixed | O3 |
| ARC-9: O4 folds into O3 | minor | fixed | O4 |
| ARC-10: O5 merges the double visit into O2 | minor | fixed | O5, O2 |
| ARC-11: O6 paired with the front-end S8.3 export | minor | fixed | O6 |
| ARC-12: O7 widened to region, ranges, compare, literal shapes | major | fixed | O7 |
| ARC-13: O10 alias check at the call | major | fixed in part: a cross-buffer pair is rejected at declaration (OD24); same-buffer pairs are constraints checked at the call | O10; Q9.2 |
| ARC-14: O11 replaced by unrolling plus checked i64 | major | fixed | O11, O20, O24 |
| ARC-15: S1.3 case audit | major | fixed | O24; oracle/coverage-table.md; OD16 |
| ARC-16: O12 over-engineered | minor | fixed | O12 |
| ARC-17: O14 to O17 later, O16 early | minor | fixed | oracle/plan/overhaul.md |
| ARC-18: element norms guarded in the emitter | major | fixed | O22 |
| ARC-19: addressing errors at the call, elements always defined | major | fixed | O24, O23; Q5.3 |
| ARC-20: per-generation C++ build | minor | fixed | O28; Q5.9 |

## Confirmations

- The track recorded these rows as accepted with no change, because the review agrees with the plan; they are not findings.
- Verification: C1 to C5, C7 to C13, C15, C17 to C25, C27, C29 and D1 to D3, D5 to D9, D12, D14, D15, D19, D20 confirmed.
- Critic: O1, O2, O4, O5, O6, O7, O17, O18 keep; O14 keep the test-bounds table.
- Architect: O8 and O9 keep; O18 keep early.
