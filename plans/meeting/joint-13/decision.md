# Joint decision, panels 13a, 13b, 1, and 2

The joint moderator's ruling on the three records panel 13b left open (process.md step 4).
No owner decision is touched, and nothing goes to the owner.

| Record | Decision | Confidence |
| ------ | -------- | ---------- |
| Q13b.15 vs Q2.9 R11 | R11 stands as written; a `read` with no `write` is rejected (S9) | medium-high |
| Q13b.9 vs Q1.10 | "Exact decimal" means the exact value TAIDL receives; Q1.10 is kept, not amended | medium-high |
| Q13b.8 vs Q13a.7 (SF23) | One rule: fold a plain `bool` by truth only, never dropping a comparison; a constant comparison where one must remain | medium-high |

## Rules

- A `read` with no `write` is rejected; an architectural prefetch is written with no regions.
- A `float` is its binary64 value, an `int` or `Fraction` its exact value, each rounded once; the author spells a decimal as `Fraction`.
- A plain `bool` in `require` or `all_of` drops when `True`; in `any_of` it drops when `False`.
- A plain `bool` that cannot drop is recorded as the constant comparison `0 == 0` or `0 != 0`, so no sibling comparison loses its fault.
- A call written with no argument raises at the author's line; a call emptied by dropping is `True` for `all_of` and `False` for `any_of`.
- A `require` whose condition is `False` from plain values alone is rejected at `check`, naming Python `if`.

## Record 1: Q13b.15 against panel 2 Q2.9 R11

- Ruling: keep R11 unchanged; Fable's amendment, a legal `read` with no `write` in the traced surface, is declined.
- Evidence:
  - 1. R11's ruling text is "Legal only without regions ... with regions it is rejected in v3" (minutes/rulings.md:21); it states a rule on regions, and the string surface is not in it.
  - 2. Panel 11 reasoning 10 removes "the duplicated `parameter(k)` type line and `check_values` reconciliation" (panel-11/decision.md:92); it names no region rule, so it repeals no R11 clause.
  - 3. Fable's reading, that R11 only reconciled semantics text with regions, is therefore not what the ruling says; Opus conceded R11 in panel 2 (panel-2/decision.md:127).
  - 4. Panel 2's open sentence, "whether an input-only instruction ... may omit semantics" (panel-2/round2-sonnet.md:147), was closed by that concession, and the decision's R11 line carries the binding text.
  - 5. Q2.3 admits a change only on an in-scope instruction whose sketch ran (panel-2/decision.md:10); panel 13b found none, and the prefetches Fable cites are fault-free hints (panel-13b/round2-fable.md:60).
  - 6. Rejection has a genuine cause under OD40: the architect corrects it by deleting the region, and the model loses nothing.
  - 7. Legalising it later is a relaxation, not a break (OD10), so keeping R11 forecloses nothing.
- Binding text kept: panel 2 Q2.9 R11 (panel-2/decision.md:125) and panel 13b Q13b.15 as written.
- Authors: the S9 row stands; no sentence for S5.2 or S7.1.

## Record 2: Q13b.9 against panel 1 Q1.10

- Ruling: Q1.10 holds as written; panel 13b's reading is confirmed, so no amendment is made.
- Evidence:
  - 1. The front end "never reads its source text" (language/spec-draft/grammar.md:4), so the only value a rule can reach for a `float` is the one received.
  - 2. Every binary64 has an exact decimal expansion, so "exact decimal" is well defined for the value received, and `0.1` at `f32` is that expansion rounded once to nearest even.
  - 3. Q1.10 names `Fraction` or hex bits as the single-rounding spellings (panel-1/decision.md:146); both are exact values received, which the source-text reading cannot be.
  - 4. Q1.10 step 2 (rounding via f64 rounds twice) is met for an `int` or `Fraction`; a literal written as a decimal reaches the front end already rounded once, by Python, so the rule cannot reach it.
  - 5. The source-text reading would need a `str` constant, which all three panel 13b debaters dropped.
  - 6. OD16 asks for one simple norm; "a plain number is its exact value rounded once" is it.
- Binding text kept: Q1.10 options A and D, unchanged.
- Edit: semantics.md:78 changes "its exact decimal value" to "its exact value, a `float` being its binary64 value"; L17 tests `1 + 2**-8 + 2**-40` at `bf16` and `2**53 + 2**29 + 1` at `f32`.
- Cost: `0.1` at `bf16` is the binary64 `0.1` rounded once; an author who needs the decimal rounded once writes `Fraction('0.1')`.

## Record 3: Q13b.8 against panel 13a Q13a.7 (SF23)

- Ruling: one rule for both, set in Rules above; the two rulings disagree on how a plain `True` is held, and each is amended.
- Evidence:
  - 1. Both panels reject the `assert` advice, which `python -O` removes, and both accept a family loop's plain `n <= 4` (OD40; panel-13a/decision.md:73, panel-13b/decision.md:106).
  - 2. Expressions are evaluated "left to right, with no short circuit; a fault in any of them is a fault of the issue" (language/spec-draft/expressions.md:38).
  - 3. SF23 says `any_of` "holds with any True"; dropping the other arms then loses their faults, so `any_of(True, a // b == 1)` stops faulting at `b == 0`, which breaks expressions.md:38.
  - 4. Q13b.8 meets that with a leaf node, but panel 13a is right that conditions have no boolean leaf (language/spec-draft/object_model.md:21) and that `Model` equality must not split members.
  - 5. A constant comparison `0 == 0` or `0 != 0` is an existing node, so only the cases that must keep a comparison add one, and generators read nothing new.
  - 6. Dropping a `True` conjunct or a `False` disjunct removes no comparison and no fault, so those are SF23's drops with no loss.
  - 7. SF23's "`all_of()` left with no condition rejects" would reject `all_of(True)` from a passing family member; it is `True` instead, by OD40 (conformance.md:8).
  - 8. A call written with no argument is a signature error at the line (grammar.md:26); a result that is `False` from plain values alone is the R3 class and is rejected at `check`.
- Binding text amended:
  - Q13a.7 loses "`any_of` ... holds with any `True`" and "`any_of()` or `all_of()` left with no condition rejects"; it takes the Rules above.
  - Q13b.8 loses "a plain `bool` is a condition leaf that never faults"; it takes the constant comparison.
  - Kept from both: always-false `require` rejected, empty call raises, G19's `assert` note replaced by R8, instructions.md:63 edited to match.
- Authors: panel 13a and 13b authors apply this text to expressions.md, instructions.md, and conformance.md.
