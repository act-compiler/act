# P2 Three-word taxonomy for every S1.3 item

## What

- Classify every S1.3 item as rejected (the front end refuses it at load), fault (a defined error at issue, named by a clause id), or defined value (the specification gives the result).
- Move every decidable item to rejected and list it in S9 with its call and error.
- Give an unwritten region element a defined value that the specification states; the oracle's initial-memory fill follows it.

## Benefits

- Generators stop guessing; coverage.md rows in act-backend and act-oracle map one-to-one onto taxonomy words.
- The formal track gets a finite list of faults and defined values.

## Caveats

- Some items are decidable only with shapes from P1; sequence after P1.
- Owner decision 16 makes every violation a fault or a defined value; each fault needs a check in every generator.
- The fill value of unwritten elements is one specification rule shared with the oracle.

## Fail points

- An item marked rejected that a case study needs, noticed when a case-study specification fails to load.
- An item left without a word that the backend silently relies on, noticed by a P7 example with divergent oracle and backend results.

## Order and dependencies

- Depends on P1, L; feeds P7, P8, P9.

## Verification

- S1.3 has no item without one of the three words.
- Both coverage.md files cite the word for every partial row.
