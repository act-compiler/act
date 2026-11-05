# P1 Numbered per-operation constraints in S8

## What

- Give every S8 operation row an Inputs table, an Outputs table and numbered constraints `(C1)..(Cn)` over `shape`, `element_type`, `rank` and attributes, in StableHLO's form.
- Define the helper functions once in S7 and reference them everywhere.
- Generate the oracle's `disagreement` and `check_shapes` and the backend's shape checks from these rows, or at minimum cite `(C<n>)` in each error.

## Benefits

- Closes the S9 gap "shapes and element types are not checked against S8".
- One source for two generators' checks; error messages become spec citations.

## Caveats

- The S8 table has 78 words; writing constraints for all is a quarter of one engineer.
- `dot`, `gather`, `pad` and `dynamic_update_slice` need multi-line constraints; keep StableHLO's wording where TAIDL's attributes match.

## Fail points

- A constraint written stricter than XLA accepts, noticed when an existing example fails the new front-end check.
- A constraint looser than the backend's Rust check, noticed by P7 examples that pass the oracle and fail the backend.

## Order and dependencies

- Depends on L; feeds P2, P7, P9.

## Verification

- Every S8 row has at least one `(C<n>)`; a script lists rows without one.
- Each existing example in taidl, act-oracle and act-backend passes the new checks unchanged.
