# Shared-code plan index

Sizes assume three to four engineers: S under a week, M one to three weeks, L over three weeks.
Shared-code item IDs are U1 to U11; the track prefix in other plans' dependency columns is U.
Items of other tracks are named by their own prefix (L language, B backend, O oracle, T tests, A act umbrella).

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| U1 | Close the two emitter divergences | L6, O3 | S | Q1 |
| U2 | Typed front-end contract with a version | L15 | M | Q1 |
| U3 | One final validation that publishes derived facts | U2, L5 | M | Q1 |
| U4 | Shape and per-word element-type rules in the front end | U3, L3 | L | Q2 |
| U5 | Element-type table and literal check | U2, L16, O8 | M | Q1 |
| U6 | Integer folding in the front end | U2, L6, L14 | S | Q2 |
| U7 | Conformance corpus with expected verdicts | U3, L8, T | M | Q1 |
| U8 | Coordinated v3 release train | U2, U3, U4, U5, U6, U7, U10, A | M | Q2 |
| U9 | One copy of docker and CI scripts | A | S | Q2 |
| U10 | Generators call `validate()` first | U3 | S | Q1 |
| U11 | Adding a generator: template generator and expectations | U2, U3, U10, T11, T12, L40, L41 | M | Q2 |

## Notes

- U4 absorbs L12, O6, O7, and B23: it is one item in two ordered PRs (move, then new checks).
- U10 replaces B21 and B22; the backend's region-value check is deleted there.
- U11 is the extension path of decision 39; L41 owns the guide text.
- U2 precedes B13 and carries no B dependency.
