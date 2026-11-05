# P7 Executable examples per spec row

## What

- Every S8 operation row and every S9 rejection row gets one example, with expected output or expected error, stored once and run by the oracle and by the backend conformance harness.

- Boundary: P7 holds operation rows and rejection rows; accelerator sketches are the executable corpus of C11.

## Benefits

- The spec becomes executable; both generators prove the same rows.

## Caveats

- Backend examples need an accelerator that selects the operation; use one shared conformance accelerator.

## Fail points

- Oracle and backend disagree on an example, which is the point: it names a spec ambiguity to fix.

## Order and dependencies

- Depends on P1, P2, T; feeds P12.

## Verification

- A script counts S8 and S9 rows without an example and the count is zero.
