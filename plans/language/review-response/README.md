# Review response

The language track's answer to the three spec reviews, one row per finding, split by review.
Binding inputs are the owner decisions and findings accepted on evidence; a change that rests only on a tech-lead ruling is marked "pending panel" with the ruling ID.

- [ambiguity.md](ambiguity.md): the ambiguity review, 58 findings
- [implementer.md](implementer.md): the implementer review, 38 questions
- [minimalist.md](minimalist.md): the minimalist review, 56 findings

## Counts

| review | accepted | partly | rejected | pending panel |
| --- | --- | --- | --- | --- |
| ambiguity | 53 | 3 | 0 | 2 |
| implementer | 28 | 7 | 1 | 2 |
| minimalist | 38 | 9 | 3 | 6 |
| total | 119 | 19 | 4 | 10 |

## Rulings the draft waits on

| ruling | what waits | where |
| --- | --- | --- |
| R2 | identifier renames beyond owner decision 19 | L15, D13, D21 |
| R5 | deriving the in-place mark from region text | D33 |
| R7 | constraints reading the post-update state | D30 |
| R8 | float accumulation and transcendental accuracy | S1.3.15, L31, D32 |
| R58 | removing `set_apart` and `set_no_partial_overlap` | D10, D33 |
| R59 | the two-case-study rule for extensions | extensions plan |
| R60 | removing `cholesky`, `triangular_solve`, `batch_norm_inference` | D33 |
| R61 | cutting the reserved-names table | D17 |
| R62 | merging L3, L12, O6, O7 | L3, L12, D26 |
