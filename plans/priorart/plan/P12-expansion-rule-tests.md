# P12 Named and oracle-tested expansion rules

## What

- Each hand-written `ir2ir` rule gets a name, a stated shape relation, and a test that checks both sides agree in the oracle on random shapes and element types.

## Benefits

- Soundness of the search rests on ten tested rules instead of ten strings.

## Caveats

- `(?x) => (dot_? ?x eye_?)` style rules need a generated identity; the test builds it.

## Fail points

- A rule unsound at some element type, noticed by the random test; narrow the rule's type set.

## Order and dependencies

- Depends on P3, P7.

## Verification

- Every rule name appears in a test; CI runs the random test with a fixed seed.
