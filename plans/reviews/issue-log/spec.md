# Issue log: spec

Sources are `reviews/spec/ambiguity.md`, `implementer.md`, and `minimalist.md`.
The dispositions come from `language/review-response/` (ambiguity, implementer, minimalist) and were checked against the language plan and panels 1, 2, 3, and 9.
Findings are keyed AMB (ambiguity), IMP (implementer), and MIN (minimalist).
S ids are sections of the language specification draft; L, D, and F ids are language plan items and bugs.
Panel questions are cited as Qn.k and owner decisions as ODn.
The ambiguity review lists its findings most severe first; the proposed norm N0 and the first four findings are blockers, the next twenty majors, the rest minors.

## Ambiguity review

| Finding | Severity | Disposition | Where it landed |
| --- | --- | --- | --- |
| AMB-N0: one norm for OD16 (a failed check is a fault) | blocker | fixed | S1.3, S5.6, S10 `fault` |
| AMB-T1: S7.5 `constant` literal fit, generators disagree | blocker | fixed | S7.5, S1.3.9, S9; L16; Q1.10 |
| AMB-T2: S6.2 overflow rules contradict | blocker | fixed | S6.2, S1.3.1; L6, D7; Q1.2 |
| AMB-T3: S5.3 constraints read a different state than starts | blocker | fixed | S5.3, S5.4, S5.6; D30; Q1.1 |
| AMB-T4: S8.4 float accumulation order | blocker | fixed | S1.3.15, S8.4; L31, D32; Q1.5 |
| AMB-T5: S8.4 reduce initial value | major | fixed | S8.4; L4, D12; Q1.11 |
| AMB-T6: contents of storage before the first write | major | fixed | S5.6; D31; Q1.4 and the joint decision on unwritten reads |
| AMB-T7: S5.5 in-place aliasing | major | fixed | S5.5, S1.3.6, S9; L5, LB21, D29; OD17, Q9.2 |
| AMB-T8: byte order and sub-byte packing | major | fixed | S8.4, S1.3.14; LB6 |
| AMB-T9: negative start and the `d0` bound | major | fixed | S4.3, S5.2, S1.3.3; LB24, L5 |
| AMB-T10: StableHLO and XLA version pin | major | fixed in part: the pin is the oracle's jax image, not a literal release string | S8.4; L2 |
| AMB-T11: integer edge cases in a semantics | major | fixed | S8.4, S1.3.12; Q1.8 |
| AMB-T12: S7.2 partial instruction never checked | major | fixed | S5.2, S9; L11, LB27; Q2.9 |
| AMB-T13: S4.5 late `add_data_model` | major | fixed | S4.5, S9; L11 |
| AMB-T14: S7.4 dimension 0 | major | fixed | S7.4, S1.3.7, S9; L16, LB28 |
| AMB-T15: S7.2 element count or type mismatch | major | fixed | S7.2, S8.3, S1.3.8; L12 |
| AMB-T16: overlapping outputs | major | fixed | S5.2, S1.3.4; L5 |
| AMB-T17: failed constraint | major | fixed | S5.3, S1.3.5 |
| AMB-T18: computed divisor 0 | major | fixed | S6.2, S1.3.2 |
| AMB-T19: `@a.` in another place | major | fixed | S6.1, S9; L10 |
| AMB-T20: initial value past 64 bits | major | fixed | S4.4, S9; L16 |
| AMB-T21: S6.3 short circuit | major | fixed | S6.3, S5.6 |
| AMB-T22: `compare` default `type` | major | fixed | S8.4 |
| AMB-T23: `dot` with mixed element types | major | fixed in part: one operand type is stated; the result type sets the accumulation bound | S8.4; L31; Q1.5 |
| AMB-T24: S8.3 checks the wrong operand for `select` and `clamp` | major | fixed | S8.3 |
| AMB-T25: `INT` base | minor | fixed | S2.1, S6.2 |
| AMB-T26: `-inf`, `-nan`, and the NaN pattern | minor | fixed | S2.1, S7.5; Q1.7 |
| AMB-T27: least 64-bit literal | minor | fixed | S6.2 |
| AMB-T28: `(0)` and `-0` divisors | minor | fixed | S6.2, S9; L14, LB10 |
| AMB-T29: semantics with no `%Out` | minor | fixed | S7.1, S9 |
| AMB-T30: a front end may reject more than S9 | minor | fixed | S9 |
| AMB-T31: S7.1 bare-word sentence | minor | fixed | S7.1 |
| AMB-T32: starts compared by text | minor | fixed | S5.5 |
| AMB-T33: S8.5 run-time shapes | minor | fixed | S8.5 |
| AMB-T34: Python keyword list | minor | fixed in part: soft keywords allowed; the list is the running Python's | S4.1 |
| AMB-T35: non-`str` list entries | minor | fixed | S5.1, S9; L16, LB25 |
| AMB-T36: `bool` and tuple in S9 | minor | fixed | S4.3, S4.4, S9; L16 |
| AMB-G1: S3.2 omits `compare` | minor | fixed | S3; D9, L8 |
| AMB-G2: S2.2 longest match and `WS` | minor | fixed | S2.2 |
| AMB-G3: S2.1 `COMMENT` and `WS` rows | minor | fixed | S2.1 |
| AMB-G4: `-inf` and `-nan` lex as `NUMBER` | minor | fixed | S2.1 |
| AMB-M1: late buffer declaration (duplicate of T13) | major | fixed | S4.5, S9 |
| AMB-M2: regions without semantics (duplicate of T12) | major | fixed | S5.2, S9 |
| AMB-M3: dimension 0 (duplicate of T14) | major | fixed | S7.4 |
| AMB-M4: `constant` literal range (duplicate of T1) | blocker | fixed | S7.5 |
| AMB-M5: negative start, `d0` past end (duplicate of T9) | major | fixed | S5.2 |
| AMB-M6: contents before first write (duplicate of T6) | major | fixed | S5.6 |
| AMB-M7: float summation order (duplicate of T4) | blocker | fixed | S1.3.15; L31 |
| AMB-M8: integer edge cases (duplicate of T11) | major | fixed | S8.4 |
| AMB-M9: `inf` at `f8e4m3fn`, NaN pattern | minor | fixed | S7.3, S7.5 |
| AMB-M10: `and` and `or` evaluation (duplicate of T21) | major | fixed | S6.3 |
| AMB-M11: non-`bool` mark flags | minor | fixed | S5.5, S9; L16, LB26 |
| AMB-S1: S9 lacks "no `%Out<k>` line" (duplicate of T29) | minor | fixed | S9 |
| AMB-S2: start or size neither `int` nor `str` | minor | fixed | S9 |
| AMB-S3: Python `int` start past 64 bits | minor | fixed | S9 |
| AMB-S4: `%In01` | minor | fixed | S7.2, S9 |
| AMB-S5: name not a `str` | minor | fixed | S4.1, S9 |
| AMB-S6: non-`str` list entry (duplicate of T35) | minor | fixed | S9 |

## Implementer review

| Finding | Severity | Disposition | Where it landed |
| --- | --- | --- | --- |
| IMP-Q1: attributes of `Accelerator` | major | fixed | S4.6; L29 |
| IMP-Q2: attributes of a buffer | major | fixed | S4.6 |
| IMP-Q3: `d0` in the buffer list | minor | fixed | S4.6, S4.3; L30 |
| IMP-Q4: instruction name and attribute lists | major | fixed | S4.6; L15 |
| IMP-Q5: raw and parsed regions | major | fixed | S4.6 |
| IMP-Q6: tree node types | major | fixed | S4.6 (Trees); Q3.4 |
| IMP-Q7: backtick lookup | minor | fixed | S4.6 (Trees) |
| IMP-Q8: attribute names by text | minor | fixed | S3.1, S4.6 |
| IMP-Q9: `parameter(k)` and literal access | minor | fixed in part: accessor path stated; the worked tree waits for the examples item | S4.6; L9 |
| IMP-Q10: program form | major | fixed | S5.6 |
| IMP-Q11: initial contents | major | fixed | S5.6; D31 |
| IMP-Q12: size of `d0` | major | fixed | S4.3, S5.2, S1.3.3 |
| IMP-Q13: sub-byte layout | major | fixed | S8.4 |
| IMP-Q14: float types and libraries | minor | fixed in part: bit layout named as `ml_dtypes`; no library is mandated, since that adds a dependency | S7.3 |
| IMP-Q15: `d0` byte order | major | fixed | S8.4 |
| IMP-Q16: state width and wrap | major | fixed | S6.2, S4.4 |
| IMP-Q17: failed constraint | major | fixed | S1.3.5, S5.6 |
| IMP-Q18: placement constraints location | minor | fixed | S4.6 |
| IMP-Q19: `ROOT` at run time | minor | fixed | S7.1, S5.6 |
| IMP-Q20: line order | minor | fixed | S5.6 |
| IMP-Q21: regions without semantics | major | fixed | S5.2, S9; L11 |
| IMP-Q22: scalar region shape | minor | fixed in part: rule stated; the worked example waits for the examples item | S5.6; L9 |
| IMP-Q23: line dimension 0 or less | major | fixed | S7.4, S1.3.7 |
| IMP-Q24: attribute expressions read post-update state | major | fixed | S5.4, S5.6; Q1.1 |
| IMP-Q25: negative axes | minor | fixed | S8.4 |
| IMP-Q26: reference revision | major | fixed in part: pinned to the oracle's jax image, as AMB-T10 | S8.4; L2 |
| IMP-Q27: StableHLO versus XLA-HLO edge cases | major | fixed | S8.4 |
| IMP-Q28: `compare` default | minor | fixed | S8.4 |
| IMP-Q29: `dot` and bf16 accumulation | major | fixed | S1.3.15; L31; Q1.5 |
| IMP-Q30: reduce initial values | minor | fixed | S8.4; Q1.11 |
| IMP-Q31: `constant` fit | major | fixed | S7.5; Q1.10 |
| IMP-Q32: operand order | minor | fixed | S8.4 |
| IMP-Q33: `pred` and broadcast shapes | minor | fixed in part: shapes follow the reference until the per-word rules land | L3; U4 |
| IMP-Q34: trust the line's type and shape | major | fixed in part: a generator rejects a mismatch until the front end checks it | S8.3, S8.4; L12; Q3.1 |
| IMP-Q35: one exception type for unsupported features | minor | rejected: generators are separate programs, S1.2 fixes the message content, and a Python class name binds no Rust output | none |
| IMP-Q36: an example per S5 feature | minor | fixed in part: new examples ride on the examples item | L9 |
| IMP-Q37: `copy` is the identity | minor | fixed | S8.4 |
| IMP-Q38: float tolerance | major | fixed | L31; Q1.5 |

## Minimalist review

| Finding | Severity | Disposition | Where it landed |
| --- | --- | --- | --- |
| MIN-V2: merge the `d0` special case | minor | fixed | S4.3; L30, LB20; Q2.9 |
| MIN-V6: merge `add_constraints` into `constraints=` | minor | fixed | S5.3, S9; L30, D34; Q2.9 |
| MIN-V7: derive `set_inplace` from region text | major | superseded by OD30 and OD22 (the mark stays as shorthand); Q2.1 is overridden the same way | D33; Q9.2 |
| MIN-V8: cut placement marks | major | superseded by OD30 (the marks stay as shorthand, each expands to its relation's constraint) | D10, D33; Q9.2 |
| MIN-V10: defer `ROOT` | minor | fixed | S7.1; Q2.9 |
| MIN-V11: cut presumed reshape | major | fixed: Q2.9 removes it; the review-response row still reads rejected and is stale | S7.2; language/plan/norms.md |
| MIN-V13: cut three rare words | minor | fixed | L33, D33; Q2.8 |
| MIN-V16: shrink S1.3 to two outcomes | major | fixed | S1.3; OD16 |
| MIN-V17: merge the S2 and S3 grammar copies | minor | fixed | S2, S3; D9, L8 |
| MIN-V18: cut the reserved-names table | minor | fixed | S4.1; D17; Q2.10 |
| MIN-V19: move parsed trees to the README | minor | fixed in part: moved into the object model S4.6 because generators need it as a contract | S4.6 |
| MIN-V21: S10 bans for four words only | minor | superseded by OD19 and OD20 (retired terms need a recorded ban) | S10 |
| MIN-V23: merge L3, L12, O6, O7 | major | fixed | L3, L12, D26; Q3.1 |
| MIN-V24: L4 to size S | minor | fixed | L4 |
| MIN-V25: L5 to size S | minor | fixed | L5 |
| MIN-V27: cut L7 from this track | minor | fixed in part: L7 narrows to the backend | L7 |
| MIN-V28: simplify L8 | minor | fixed | L8 |
| MIN-V29: defer L9 | minor | fixed | L9 |
| MIN-V31: cut the code of L14 | minor | fixed | L14 |
| MIN-V32: shrink L15 | minor | fixed | L15, L30; Q7.10 |
| MIN-V33: defer L16 | major | rejected: the backend and the oracle disagree on literals today (`spec_checks.py:94-107`, `utils.py:398-426`) | L16 |
| MIN-V34: L17 keep and shrink | minor | fixed | L17; Q2.4 |
| MIN-V35: cut L18 | minor | fixed in part: deferred after v3 with ids stable; the SME idiom is D28 | L18, D28; Q2.4 |
| MIN-V36: defer L19 | minor | fixed | L19, L24 |
| MIN-V37: defer L20 | minor | fixed | L20; D27 |
| MIN-V38: cut L21 | minor | fixed in part: deferred after v3, `clamp` then `convert` describes saturation | L21, L24; Q2.4 |
| MIN-V39: cut L22 | minor | fixed | L22; Q2.4 |
| MIN-V40: L23 to late Q2 | minor | fixed | L23 |
| MIN-V41: L24 keep and add idioms | minor | fixed | L24 |
| MIN-V43: cut L27 and L28 | minor | fixed in part: L27 owned by B, O, A; L28 after v3 | L27, L28 |
| MIN-G1: fewer than two accelerators behind an extension | minor | fixed | L18, L21, L22 |
| MIN-G2: constructs describable today | minor | fixed | L24 |
| MIN-G3: only L17 and L23 add capability | minor | fixed | language/plan/extensions.md |
| MIN-G4: apply the admission rule to all of Q2 | minor | fixed in part: Q2.3 sets the admission rule for every extension | language/plan/extensions.md; Q2.3 |
| MIN-O1: sizes assume two engineers | major | fixed | language plan headers; OD11 |
| MIN-O2: 28 items, three of size L | minor | fixed in part: L22 cut; L3 and L12 stay L until the merge | L22 |
| MIN-O3: tutorial fail points void | major | fixed | L10 |
| MIN-O4: one breaking batch | minor | fixed in part: v3 batch per OD10; the marks stay per OD30 | language/plan/overhaul.md |
| MIN-O5: L5 and S1.3 are one S item | minor | fixed | L5 |
| MIN-O6: remove `set_inplace` | major | superseded by OD30 (same as MIN-V7) | D33 |
| MIN-O7: 78 words, not 70 | minor | fixed | L3, D1 |
| MIN-O8: order after the batch | minor | fixed | language/plan/overhaul.md |
| MIN-C1: the cuts list | minor | fixed in part: each cut is answered in its V row | see MIN-V rows |

## Confirmations

- The track recorded these rows as accepted with no change, because the review agrees with the plan; they are not findings.
- Minimalist: V1, V3, V4, V5, V9, V12, V14, V15, V20, V22, V26, V30, V42 keep.
