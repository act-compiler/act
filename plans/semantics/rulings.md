# Rulings on the spec's open questions

The three questions the TAIDL language specification draft leaves open, each with the ruling, the choice among alternatives, and the generator that must change.
The full hole table is `holes.md`; these rows expand H1, H7, and H8.

## In-place marks (H1)

- Ruling: each same-buffer input-output pair has one relation (tied, apart, exact-or-apart, may overlap); an undeclared pair is exact-or-apart, so a partial overlap faults.
- Expansions: `set_inplace` is `same` plus the tie, `set_apart` is `apart`, `set_no_partial_overlap` is `same or apart`, `set_may_overlap` is empty (Q9.2).
- A permitted overlap gives the semantics on the pre-issue state; any failed constraint is the one fault before any write (Q9.3).
- Settled: `set_inplace` stays, as shorthand for the constraints it stands for (owner decisions 22, 30); the output reuses the marked input's storage (owner decision 17).
- Settled: a mark that pairs an input and an output in different buffers is rejected by the front end (owner decision 24).
- Settled: the language assumes no hazard model; the ISA author declares how an instruction behaves when inputs and outputs overlap (owner decision 31).
- Generator changes: act-oracle checks the marks' constraints at each call; act-backend consumes the mark in `fix_inplace` (inplace_fix.rs:52-74); act-oracle adds the `set_inplace` equality, the default, and the size term; act-backend applies each pair's relation when an input dies at the instruction (Q9.4).

## Reduction initial values (H7)

- Ruling: `reduce_<r>` starts from the identity of r at the element type: 0 for add, 1 for multiply, the type's least value for max, its greatest for min, all ones for and, 0 for or and xor.
- At a float type with an infinity the least and greatest values are -inf and +inf; at `f8e4m3fn` they are its largest finite value and its negation.
- A float sum starts from +0.0, so an all-negative-zero sum is +0.0; a reduction over an empty dimension set gives the element combined with the identity, and so does a dimension of size 1.
- This is exactly act-oracle's `reduce_init` (utils.py:439-451); it becomes normative text in S8.4 and the row leaves S1.3.
- Order of a float reduction is free inside the H6 bounds; integer reductions are order-free by wraparound.
- Alternative rejected: an explicit `init` operand per reduce line; it adds a word form the grammar lacks for a value the identity already fixes.

## Out-of-bounds regions (H8)

- Ruling: a region whose start is below 0, whose size is below 1, or whose start plus size passes the access dimension of its buffer is a fault, at every loop iteration, with an error naming the instruction, the buffer, the dimension, and the iteration; the fault writes no region and no state.
- For `d0` the bound is the size of the off-chip memory, which is an issue-time parameter of the program (act-oracle's `hbm`, coverage.md:13); the spec names it `|d0|` and makes it a declared kernel parameter rather than leaving it to each generator.
- A producer such as act-backend never emits an out-of-bounds region; its allocator already treats bounds as constraints on placement.
- This is act-oracle's `check_window` and `check_sizes` (utils.py:698-712, 651-665) made normative; the rows leave S1.3's undefined table.
- Alternative rejected: wraparound or clamping of starts; hardware differs, and a silent value would hide the ISA specification's error.
