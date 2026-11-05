# Review response

The language track's answer to the three spec reviews, one row per finding, split by review.
Binding inputs are the owner decisions, the panel decisions, and findings accepted on evidence.

- [ambiguity.md](ambiguity.md): the ambiguity review, 58 findings
- [implementer.md](implementer.md): the implementer review, 38 questions
- [minimalist.md](minimalist.md): the minimalist review, 56 findings

## Counts

| review | accepted | partly | rejected |
| --- | --- | --- | --- |
| ambiguity | 55 | 3 | 0 |
| implementer | 30 | 7 | 1 |
| minimalist | 41 | 9 | 6 |
| total | 126 | 19 | 7 |

## Settled by the panels

| topic | decision | where |
| --- | --- | --- |
| identifier renames | Q2.12, Q7.10 | L15, L30, D13, D21 |
| which state expressions read | Q1.1: the pre-state | S5.4, S5.6, D30 |
| float accumulation | Q1.5: two Higham bounds | S1.3.15, L31, D32 |
| extension admission | Q2.3 | extensions plan |
| three rare words | Q2.8: removed | L33, D33 |
| reserved-names table | Q2.10: one pointer sentence | S4.1, D17 |
| shape rules owner | Q3.1: shared item U4 | L3, L12, D26 |
| marks | owner decision 30: kept as shorthand; Q9.2: expansions, new `set_may_overlap` | S5.5, L35, L36, D10, D33 |
| overlap between input and output regions | Q9.1, Q9.3: four relations, exact-or-apart default, one fault, pre-issue state | S1.3.6, S5.2, S5.5, S5.6, D18, D29 |
