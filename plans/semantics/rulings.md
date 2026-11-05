# Rulings on the spec's open questions

The three questions the spec draft leaves open, each with the ruling, the choice among alternatives, and the generator that must change.
The full hole table is `holes.md`; these rows expand H1, H7, and H8.

## In-place marks (H1)

- Ruling: `set_inplace` flag k asserts that input region k and output region 0 are one region: the same tensor buffer, and starts and sizes equal text for text as `keep_off` compares starts (instruction.py:214).
- The front end rejects a flag whose input differs from the output, and rejects a flag on an instruction with other than one output.
- The mark changes nothing about the computed value: the instruction still reads every input before it writes (instructions.md:136).
- At issue, an input region that overlaps an output region is rejected unless the two are equal and the input carries the mark.
- Alternative rejected: keep the mark implementation-defined (overview.md:45); it lets act-backend copy or alias storage (inplace_fix.rs:52-74) while act-oracle ignores the mark (coverage.md:20), so the same program means two things.
- Generator changes: act-oracle checks the overlap rule at each call; act-backend relies on the front end's equality check and drops its own copy logic for the marked case.

## Reduction initial values (H7)

- Ruling: `reduce_<r>` starts from the identity of r at the element type: 0 for add, 1 for multiply, the type's least value for max, its greatest for min, all ones for and, 0 for or and xor.
- At a float type with an infinity the least and greatest values are -inf and +inf; at `f8e4m3fn` they are its largest finite value and its negation.
- A float sum starts from +0.0, so a reduction over an empty dimension set gives the operand unchanged and over a dimension of size 1 gives the element itself.
- This is exactly act-oracle's `reduce_init` (utils.py:439-451); it becomes normative text in S8.4 and the row leaves S1.3.
- Order of a float reduction stays implementation-defined within the H6 tolerance; integer reductions are order-free by wraparound.
- Alternative rejected: an explicit `init` operand per reduce line; it adds a word form the grammar lacks for a value the identity already fixes.

## Out-of-bounds regions (H8)

- Ruling: a region whose start is below 0, whose size is below 1, or whose start plus size passes the access dimension of its tensor buffer is rejected at issue, at every loop iteration, with an error naming the instruction, the buffer, the dimension, and the iteration.
- For `d0` the bound is the size of the off-chip memory, which is an issue-time parameter of the program (act-oracle's `hbm`, coverage.md:13); the spec names it `|d0|` and makes it a declared kernel parameter rather than leaving it to each generator.
- A producer such as act-backend must never emit an out-of-bounds region; its allocator already treats bounds as constraints on placement.
- This is act-oracle's `check_window` and `check_sizes` (utils.py:698-712, 651-665) made normative; the rows leave S1.3's undefined table.
- Alternative rejected: wraparound or clamping of starts; hardware differs, and a silent value would hide the ISA description's error.
