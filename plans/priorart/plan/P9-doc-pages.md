# P9 Generated instruction documentation pages

## What

- A generator in the act umbrella renders one page per instruction from an `Accelerator`: signature, data models read and written, attributes and constraints with `(C<n>)`, state updates, semantics block, faults and defined values, one example line with its oracle output.

## Benefits

- Case-study accelerators become readable without the Python source, in NKI's page style.

## Caveats

- Pages are derived; never edited by hand.

## Fail points

- A page that disagrees with the oracle output, noticed because the example is run when the page is built.

## Order and dependencies

- Depends on P1, P2, A.

## Verification

- Pages for every case-study accelerator build in CI and every example line executes.
