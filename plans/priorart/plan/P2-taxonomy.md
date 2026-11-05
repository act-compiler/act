# P2 Three-word taxonomy for every S1.3 item

## What

- Classify every S1.3 item as rejected (a front end refuses it), undefined (any behavior), or implementation-defined (each generator documents its choice).
- Move every decidable item to rejected and list it in S9 with its call and error.
- State for an unwritten region element whether reading it is undefined or yields an arbitrary value.

## Benefits

- Generators stop guessing; coverage.md rows in act-backend and act-oracle map one-to-one onto taxonomy words.
- The formal track gets a finite list of undefined cases.

## Caveats

- Some items are decidable only with shapes from P1; sequence after P1.
- "Arbitrary value" versus "undefined" affects the oracle's initial-memory fill and must be one owner decision.

## Fail points

- An item marked rejected that a case study needs, noticed when a case-study spec fails to load.
- An item left undefined that the backend silently relies on, noticed by a P7 example with divergent oracle and backend results.

## Order and dependencies

- Depends on P1, L; feeds P7, P8, P9.

## Verification

- S1.3 has no item without one of the three words.
- Both coverage.md files cite the word for every partial row.
