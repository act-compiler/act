# Layer corpus

T32 to T35 turn the panel 10 rule set into cases: the width layer, the encoding layer with the assembler round trip, derived attributes, and generator metadata.
Front-end rows are `case.toml` cases (T6); run-time rows are `program.toml` programs (T29); the round trip and the neutrality rows run in the generators' suites.
Each case cites the S id of its layer and, for a fault, S1.3.10 (a value outside its type) or S1.3.5 (a failed constraint).

## T32. Width layer cases

- What: accepted rows declare each of `s1`, `s8`, `s64`, `u1`, `u8`, and `u63` on an attribute; an untyped attribute is `s64` and a specification with no type behaves as the untyped corpus does.
- What: rejected rows, phase `build`: `u64`, `u0`, `s0`, `s65`, and a key naming no declared attribute; each cites its S9 clause and names the attribute.
- What: fault rows (`program.toml`) set an `s8` attribute to -129 and 128 and a `u8` attribute to -1 and 256; each expects `code`, clause S1.3.10, and unchanged buffers and state.
- What: boundary rows set `s8` to -128 and 127 and `u8` to 0 and 255, `s64` to its two ends; each is a value and no fault, and no value wraps, truncates, or sign-extends.
- What: an expression over typed attributes evaluates in signed 64 bits (`@a.x + 1` with `x = 127` on `s8` is 128 and no fault); the type binds the attribute value only.
- What: cross-check rows run a typed attribute and its hand-written pair of range constraints over a swept range and require the same fault set and values (decision 30).
- What: a region start `1024 - @a.addr` with a typed negative `addr` is a value; the type binds the attribute, the implicit constraints bind the evaluated start.
- What: backend rows (T13) check that a generated addressing attribute stays inside its declared range and that a `u8` attribute is never negative; a `s64` attribute may be negative.
- What: a declared range the solver cannot hold is a coded generation error, never a clamp; the row expects `unsupported` with the code in the backend expectations table.
- Benefits: the type is one implicit constraint, so a width that widens or narrows a range fails a named boundary row.
- Caveats: `u64` is the only rejected width inside the grammar; `s64` is the default, so no row declares `s64` as rejected.
- Fail points: a boundary row passed by an oracle that wraps; the neighbor values -129 and 128 rule it out.
- Order and dependencies: after T5, T6, T29, L37, and D16; Q1, first of the layer rows.
- Verification: every width in `s1..s64` and `u1..u63` is covered by a swept boundary property (T9 strategies); the named rows above exist.
- Breaking: no.
- Files: new cases under `taidl/conformance/`; the seed is `act-oracle/taidl_to/generic/utils.py:268-275` (`warn_width`), whose warning becomes the S1.3.10 fault row.

## T33. Encoding layer cases and the assembler round trip

- What: one corpus ISA specification `enc_mini` holds a 16-bit word with fixed opcode bits, a scattered immediate (RISC-V B-type shape), a signed field, and a two-slice descriptor.
- What: accepted rows: exact coverage, a signed field packed as two's complement, slices listed high to low, and a specification with no block.
- What: rejected rows, phase `build`, one per check: a word bit covered by no slice or fixed bit, a bit covered twice, a slice total different from the attribute's declared width, an encoded attribute with no declared type, a fixed-bit overlap with a slice.
- What: clash rows: two instructions whose fixed bits both match one word are rejected, naming both instructions and the word.
- What: a word width that is not explicit, or is variable, is rejected; multi-word encodings have no row until a case study needs one.
- What: encode rows hold hand-computed words for boundary attribute values (written by hand, never produced by the generator).
- What: decode rows: each hand-computed word decodes to its instruction and attribute values; a word that matches no instruction expects the defined decode error and its code.
- What: round-trip rows: `decode(encode(x)) == x` for every instruction at boundary values and at hypothesis-drawn values (fixed seed, T9), and `encode(decode(w)) == w` for every word of `enc_mini` that decodes (the 16-bit word space is run in full).
- What: backend rows: the text the backend prints for a kernel assembles with the generated assembler, and the disassembly equals the text.
- What: placement-stripped rows: the oracle's output and the backend's text are identical with the block's word width, fixed bits, and slices removed, since neither reads placement; the link constraints of T34 stay.
- Benefits: the three generation checks, the decode error, and the round trip are the acceptance test of the whole layer (Q10.4).
- Caveats: the block ships with the assembler and disassembler as one unit, so these rows land in one PR with T34.
- Fail points: a round trip that passes because encode and decode share one wrong table; the hand-computed words rule it out.
- Order and dependencies: after T32, T6, T29, L38, L39, and the assembler and disassembler generator; Q2, last of the layer rows.
- Verification: each check has one rejected row naming its attribute or instruction; the word-space run covers 65536 words.
- Breaking: no.
- Files: new `taidl/conformance/` cases and `act-backend/tests/test_assembler.py`; no existing file changed.

## T34. Derived attribute cases

- What: the rule under test is `E == k*D + c` or `D == k*E + c` with one encoded attribute `E`, nonzero integer `k`, integer `c`, and exactly one link per derived attribute.
- What: accepted rows: `A == B*8` (decision 35), `B == 16*A`, `k = -1`, `c = -7`, `k = 1` with `c = 0`, a negative derived value, and a typed derived attribute.
- What: closed-form rows: encode evaluates the link, and decode inverts it by one subtraction and one exact division; each row stores both directions by hand.
- What: partial-direction rows: for `A == B*8`, decode of `A = 12` has no integer `B` and expects the violated-constraint fault (S1.3.5); for `B == 16*A`, encode of a `B` that is not a multiple of 16 expects the same fault.
- What: field-range rows: a derived value whose encoded partner falls outside the field's type expects the violated-constraint fault.
- What: rejected rows, phase `build`, each naming the attribute: no link, two links, a nonaffine link (`D*D`), `k = 0`, a link naming another derived attribute, a link naming two encoded attributes, a chain, a state variable in a link, and a lookup link.
- What: a specification with no block and an attribute with no link is accepted, with no derived attribute.
- What: backend rows (T13): every assembly the backend emits for a kernel on `enc_mini` assembles, and its derived attributes satisfy their links.
- Benefits: the rule is closed-form, so each row has a hand value and no row runs a solver.
- Caveats: the oracle evaluates a link as any constraint; the allocator receives the encoded attribute with the type's domain and the link as a linear constraint.
- Fail points: a decode row that passes because the inverse divides inexactly; the `A = 12` row rules it out.
- Order and dependencies: after T33; same PR as T33.
- Verification: nine rejected rows and the two partial-direction rows exist; every row names its attribute.
- Breaking: no.
- Files: new cases under `taidl/conformance/`; seed `act-backend/act_backend/isel_ir2isa_rewrites_rs_generator.py:166-181` (`attribute_factor`).

## T35. Generator metadata cases

- What: generator metadata is a namespace-keyed plain-data mapping on the accelerator, each buffer, and each instruction; a metadata entry is one generator's block.
- What: accepted rows: an entry at each of the three levels, two namespaces on one node, and an entry in a namespace no generator reads.
- What: the front-end row: the typed model exposes each entry as plain data and `validate()` lists the namespaces present, as a note.
- What: neutrality rows: for every case carrying metadata, the generated oracle source and the backend's assembly are byte-identical with all metadata stripped, so another generator's entry changes no output.
- What: a generator-owned row: an unknown key inside the named generator's own namespace is rejected by that generator naming node and key; the front end accepts it.
- What: a misspelled namespace is ignored by every generator and listed by the front end.
- What: the template generator of U11 has an expectations table (T11) with a `supported` or coded `unsupported` row for every corpus case, and generated coverage.
- Benefits: decision 39 becomes executable: adding a generator's data cannot change another generator's output.
- Caveats: metadata never changes what S5 to S8 mean, so no row compares metadata against a language rule.
- Fail points: a generator that reads another namespace; the stripped comparison fails on the first byte.
- Order and dependencies: after T5, T6, T12, U2, and L40; Q1.
- Verification: the corpus holds at least one case per level; the neutrality run covers every such case.
- Breaking: no.
- Files: new cases under `taidl/conformance/`; the template generator's tests per U11.
