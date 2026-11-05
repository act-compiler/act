# Panel 10 round 2: Opus

Lens: language designer and semantics author.
Binding: owner decisions 16, 30, 34, 35, 38, and 39.
Round-1 files read: Sonnet (no Q10.5 section), Fable.

### Q10.1

All three agree: option A as an optional layer 1, expanding to implicit range constraints under decision 30, default signed 64 bits, allocator domain widened from `MAKE_INT_VAR(0, INT64_MAX)`.

- Strongest opposing argument (Sonnet, step 4): a range cannot give the layout, since 0..16 derives 5 bits while the field may be 8, so width must be declared, not derived.
- Concession: correct, and it is why my layer 1 declares a type, not a range; my round-1 Q10.2 step 3 already ties a field's slice width to the declared `w`.
- Strongest opposing argument (Fable, step 3): spell it `bits('x', 8)` and expand to `0 <= @c.x and @c.x < 256`.
- Rebuttal: that spelling states no signedness, while decision 34 allows negative attributes and a two's-complement field holds them (Fable's own Q10.3 step 8); `s<w>` and `u<w>` name both facts and reuse the oracle's `s64` vocabulary (act-oracle/taidl_to/generic/utils.py:863-869).
- Strongest opposing argument (Sonnet, step 6): declare widths in a separate mapping such as `set_widths({'addr': 'u12'})`, since `check_name` forbids a suffix in the name (taidl/taidl/instruction.py:73-74).
- Concession on placement: a name suffix is wrong under decision 34; a separate mapping or a keyword on the declaration both work, and the choice belongs with the panel 3 typed model.
- Point I keep: `u64` is rejected at declaration, so every typed range fits in signed 64 (taidl/taidl/instruction.py:400) and attribute arithmetic keeps one meaning; neither other file addresses it.
- Final position: A as layer 1, types `s<w>` (1..64) and `u<w>` (1..63), each expanding to two implicit range constraints; unchanged.
- Confidence: high on meaning; medium on spelling.
- What still divides the panel: spelling only (`s<w>`/`u<w>` versus `bits(x, n)` versus a width mapping), and whether `u64` is legal.

### Q10.2

All three agree: option A, an optional per-instruction block with word width, fixed bits, and ordered slices per encoded attribute; front-end checks for full coverage and no overlap; a defined decode error for an unmatched word.

- Strongest opposing argument (Fable, step 5): the binary emitter is the same walk as `str_piece.txt:1` and belongs in act-backend.
- Rebuttal in part: the walk reads only the block and attribute values, not the allocator, so it is a generator in its own right under decision 39's easy-add rule (act-backend/act_backend/generator.py:79-86); which repository hosts it is a packaging choice, not semantics.
- Strongest opposing argument (Sonnet, step 6): the block is a pure bit placement with no scale or offset; arithmetic lives in constraints (Q10.3).
- Agreement: this matches my round-1 block and keeps one mechanism under decision 30; implied low bits (Fable step 4, RISC-V B-type `@ 0b0`) are a Q10.3 scaling, not a block feature.
- Point I keep: an encoded attribute with no declared type is rejected, since its default 64 bits rarely matches a field and the slice-width check needs `w`.
- Final position: A; assembler first, decoder from the same block second; unchanged.
- Confidence: high.
- What still divides the panel: the host of the assembler (act-backend or a new generator), and whether the decoder is a v3 deliverable (Q10.4).

### Q10.3

All three agree: the encoding block's field list is the only marking, every other attribute is derived, uniqueness is a front-end check, the link is affine with a nonzero integer coefficient, and option C is rejected.

- Strongest opposing argument (Fable, step 4): the link form is narrow, one encoded attribute and one derived attribute, `encoded == derived * k + c` or `derived == encoded * k + c`.
- Concession: my round-1 form `c * B + E1 == E2` let E1 and E2 range over several encoded attributes, and then the encode direction is not closed form: `@a.B == @a.hi * 4096 + @a.lo` gives many `(hi, lo)` pairs for one `B` unless widths are also used, which is a solver task, not evaluation.
- So I adopt Fable's narrower boundary: each defining equality is affine in exactly one derived and exactly one encoded attribute, which makes the link a bijection between the two on its image.
- A split field stays one encoded attribute with two slices (Q10.2), so the narrowing loses no case-study need I can cite.
- Rebuttal to Fable step 5: "a decoder computes derived from encoded by the forward map, which always succeeds" holds only for the `derived == encoded * k + c` form; for decision 35's own `A == B*8`, decoding `A = 12` has no integer `B`.
- Sonnet step 5 states this correctly: that case is a defined fault (decision 16), so the semantics needs one fault row "binary value has no derived value", raised on decode, and the encode-side row "derived value has no encodable field value", raised on assembly.
- Agreement with Sonnet step 9: a derived value past signed 64 bits is the same defined range fault as S1.3.10, so no new row is needed for it.
- Agreement with Fable step 6 and my step 11: the allocator holds the encoded attribute as an `IntVar` with its typed domain plus the link as one more linear constraint (act-backend/act_backend/generic/cpp/malloc/src/constraint.cc:47-58 posts equalities), so it never emits an unencodable value.
- Final position: B plus a front-end uniqueness check, with the link narrowed to one affine equality between one derived and one encoded attribute; changed (narrowed) from round 1.
- Confidence: high on the narrowed form.
- What still divides the panel: whether a lookup-table or log2 link form is added later (Sonnet and Fable both name it; all three would wait for a case study).

### Q10.5

Opus and Fable agree (Sonnet gave no position): option A for per-node data plus option C for per-run options; the front end stores metadata opaquely under a generator namespace, and only that generator validates them.

- Strongest opposing argument (Fable, step 5): a typo is caught by the one generator that cares.
- Rebuttal: a typo inside a namespace is caught, but a misspelled namespace (`'alo'`) is claimed by no generator and is silently ignored by decision 39's own rule.
- Remedy, kept from round 1: the front end lists all namespaces present, so an author sees an unclaimed one; no generator rejects foreign namespaces, which decision 39 forbids.
- Agreement with Fable step 6: generator metadata that two generators read is promoted to a shared layer with front-end validation; my round-1 test (generator metadata never changes what S5 to S8 mean) and Fable's promotion rule are the same line seen from two sides.
- Agreement with Fable step 3: TableGen `TSFlags` is a precedent beside MLIR's discardable attributes, so the shape has two independent sources.
- Spelling (`addons=` keyword versus `add_addon(generator, mapping)`) is a panel 3 typed-model choice, as in Q10.1.
- Final position: A plus C, values plain data, the member landing with the U2 typed contract; unchanged.
- Confidence: high on the layer-versus-metadata test.
- What still divides the panel: nothing of substance between Opus and Fable; Sonnet's view is unknown.

### Q10.4

- Strongest opposing argument (Sonnet, step 3): deferral for capacity contradicts the standing owner direction "Nothing is cut for capacity" and "dates move and scope does not" (plans/meeting/panel-8/decision.md:17-19).
- Concession: my round-1 step 5 rested on decision 11 capacity, and the panel 8 direction removes that ground; Fable's step 5 rests on the same assist-factor argument and falls with it.
- Strongest opposing argument (Sonnet, step 5): with no consumer, an encoding block risks being built untested.
- Agreement: the remedy is Sonnet's unit rule, the block lands only together with its assembler and decoder and a round-trip test `decode(encode(x)) == x` over a case study with real fields.
- Rebuttal to Fable step 3: an optional block is additive, so adding it after v3 would not be a second breaking change under decision 10; the reason to land it in v3 is decision 35 plus the standing direction, not decision 10.
- Agreement with all: layer 1 first, since it carries the allocator domain fix that decision 34 needs anyway (act-backend/act_backend/malloc_instructions_h_generator.py:456).
- Agreement with Sonnet step 2 and my round-1 step 1: Q10.3 never ships without Q10.2, so the agenda's option A is incoherent.
- Final position: B in order layer 1, then the encoding block with its front-end checks and the Q10.3 rule, then the assembler and decoder as one unit; if the schedule slips, dates move, and the encoding unit never ships in part; changed from D.
- Confidence: medium; the order is firm, the fit within two quarters is a dates question panel 8 owns.
- What still divides the panel: Fable keeps the binary consumers after v3; Sonnet and I put them in v3 as the last unit.

## Escalation to the owner

None.
