# Panel 10, round 2, Fable: rebuttals

Lens: compiler and assembler architecture.
Binding: decisions 16, 30, 34, 35, 38, 39.
All three round-1 files share one frame: layer 0 is the signed 64-bit integer of today, layer 1 is width and signedness, layer 2 is the encoding block with derived attributes.

### Q10.1

- Agreement: all three choose option A as optional layer 1, expanding to implicit range constraints under decision 30, with the allocator domain `MAKE_INT_VAR(0, INT64_MAX)` (act-backend/act_backend/malloc_instructions_h_generator.py:456) widened to the declared range.
- Strongest opposing point (Opus, step 3): the type is spelled `s<w>` or `u<w>`, and `u64` is rejected because its range does not fit the signed 64-bit layer 0.
- Concession: my round 1 wrote `bits('x', 8)` with no signedness in the spelling and no cap on unsigned widths; a `u64` field would admit a literal the front end already rejects (`INT64 = range(-(2**63), 2**63)`, taidl/taidl/instruction.py:400) and the oracle treats as overflow (act-oracle/taidl_to/generic/utils.py:285-287), so the cap `u` up to 63 and `s` up to 64 is right.
- Concession: Opus's step 8 reuses the oracle's own integer vocabulary (`'int64': 's64'`, utils.py:863-869), so one spelling serves attributes and the oracle; I adopt `s<w>` and `u<w>`.
- Second opposing point (Sonnet, step 6): the declaration is a separate call such as `set_widths({...})`, so the `list[str]` attribute lists stay unchanged.
- Rebuttal: a type beside the name in one declaration cannot drift from the name list, as TableGen's `bits<5> rd` sits in the operand list and not in a side table (llvm.org WritingAnLLVMBackend, cited by all three); a mapping call keyed by name is the small-scale version of the drift path every file rejects in Q10.2.
- Final position: option A as layer 1, type `s<w>` or `u<w>` declared with the attribute, absent means `s64`, one implicit range constraint pair, no wrap or truncation of the integer value (changed on spelling and the unsigned cap).
- Confidence: high on meaning; medium on syntax placement, which panel 3's typed model constrains.
- Still divides: whether the type rides on the attribute declaration or on a separate per-instruction call.

### Q10.2

- Agreement: all three choose option A, a per-instruction block of word width, fixed bits, and an ordered slice list per encoded attribute, checked by the front end for exact single coverage and for disjoint fixed-bit patterns; the assembler is the first consumer, a decoder the second, and the oracle never reads it.
- Strongest opposing point (Sonnet step 6, Opus step 3): the map is pure bit placement, arithmetic such as `*8` lives only in constraints, and an encoded attribute of type `s<w>` or `u<w>` must have slices covering exactly `w` bits.
- Concession: my round 1 let the field map carry "implied constant low bits" for the RISC-V B-type `@ 0b0`; that is a scaling in disguise, and Sail itself puts it in the mapping as `imm @ 0b0` only because its immediate is a bitvector, while a TAIDL attribute is an integer; so the block stays pure placement and `offset == field * 2` is a Q10.3 link.
- Concession: Opus's rule that an encoded attribute without a declared type is rejected closes a hole in my round 1, where a 64-bit default attribute could have been named in a 12-bit slice list with no rule on which 12 bits of 64 to take.
- Rebuttal to Sonnet's step 11 (fixed-length words only, variable length later): agreed on fixed-length words for the first version, but the block should carry an explicit word width now, since riscv-opcodes derives 32 versus 16 bits from the instruction name alone and admits the gap in its README (cited by Sonnet).
- Final position: option A; pure placement; word width explicit; an encoded attribute must carry a layer-1 type whose width equals its slice total; the decoder is a second reader of the same block (changed on implied low bits).
- Confidence: high on the block and its two checks; medium on word lists for multi-word encodings.
- Still divides: nothing of substance; the multi-word form is open for all three.

### Q10.3

- Agreement: all three choose option B with the uniqueness rule of A: the slice list is the only marking, every derived attribute has exactly one defining equality that is linear in it with a nonzero literal coefficient, the encode direction is evaluation, the decode direction is exact division, and a nonzero remainder or a value outside the field is a defined fault (decision 16).
- Strongest opposing point (Opus, step 4): the admitted form is `c * B + E1 == E2` with `E1` and `E2` over encoded attributes and literals, which is wider than my `encoded == derived * k + c`.
- Concession: Opus's form covers a derived address that combines two encoded fields, for example a base field and an offset field, while mine admits only one encoded attribute per link; the inversion is still one subtraction and one exact division, which the backend's `attribute_factor` (act-backend/act_backend/isel_ir2isa_rewrites_rs_generator.py:166) and the allocator's `floor_division` (act-backend/act_backend/generic/cpp/malloc/include/solver.h:26) already handle; I adopt Opus's form.
- Second opposing point (Sonnet step 8, Opus step 9): a derived attribute has no type of its own and stays `s64`.
- Agreement with it: my round 1 said the same by another route, so there is no dispute; a width on a derived attribute would be a second bound on a value the link already bounds.
- Rebuttal to Opus's step 12 (no chains until a case study needs them): agreed as a first version, and the front end should reject a defining equality that names another derived attribute with a message, rather than leave the case unspecified, since decision 16 covers specification errors as well as issue faults.
- Final position: option B plus uniqueness, Opus's linear form, closed-form inversion, derived attributes typed `s64`, no chains in the first version (changed on the admitted form).
- Confidence: high.
- Still divides: whether a lookup-table link form (log2 size fields) joins the first version; all three defer it to a case study.

### Q10.5

- Agreement: Opus and I choose option A for per-node data and option C only for per-run options; both reject a side file (option B) as a rename drift path, and both cite `generate_backend(..., cost)` as the existing run option (act-backend/act_backend/generator.py:79-86); Sonnet did not answer Q10.5.
- Strongest opposing point (Opus, step 7): metadata values are restricted to plain data (str, int, bool, list, dict) so the typed model of U2 can expose and serialize them, where my round 1 stored an opaque `Mapping`.
- Concession: an opaque object cannot be serialized or diffed by the conformance corpus, and the MLIR precedent Opus cites (discardable attributes with a dialect prefix, verified by that dialect) carries only attribute values, never host objects; plain data is the right bound.
- Rebuttal to Opus's step 9 (a misspelled namespace is silently ignored by design): decision 39 says other generators ignore a metadata entry, not that a typo is lost; the front end should expose the set of namespaces present, and a generator lists the ones it reads, so a one-line diff of the two sets shows an unclaimed namespace; Opus proposes the listing too, so this is a sharpening rather than a dispute.
- Agreement with Opus's step 1 and 2: the test between a layer and a metadata entry is meaning; widths, encodings, and derived attributes change which issues fault and what the oracle sees, so they are layers and never assembler metadatas.
- Final position: option A with plain-data values per generator namespace at accelerator, buffer, and instruction level, run options as entry-point keywords, a front-end listing of namespaces present, and promotion to a shared layer at the second generator that reads the same datum (changed on plain data).
- Confidence: high on the layer-versus-metadata test; medium on the keyword spelling, which panel 3 owns.
- Still divides: nothing between the two who answered; Sonnet's view is unknown.

### Q10.4

- Strongest opposing point (Sonnet, steps 2 to 5): option A as worded is incoherent because Q10.3's derived attributes exist only relative to an encoding block, the standing owner direction says nothing is cut for capacity and dates move rather than scope (meeting/panel-8/decision.md:17-19), and decision 35 says the plan covers encodings, so all of layer 2 lands in v3 with its assembler, disassembler, and a round-trip test.
- Concession: my round 1 put the Q10.3 rule in v3 and encodings after; under the Q10.3 position all three hold, the encoded set is defined by the slice list, so the derived-attribute check has nothing to check without a block, and Sonnet and Opus are right that Q10.2 and Q10.3 move together.
- Rebuttal on capacity: the standing direction forbids cutting scope, and nobody proposes a cut; layer 2 is written into the v3 specification with its packing rule, decode determinism, derived-attribute rule, and fault row, and its generators are scheduled as the first 3.x minor release, which is a date and not a scope change, and which decision 38 makes additive since a layer-0 or layer-1 specification is complete without it.
- Rebuttal on the first consumer: the backend prints text (templates/instructions.h.INSTRUCTION_CLASSES.str_piece.txt:1) and fsim's driver reads a text trace (oracle/fsim/architecture.md:26-27), and the roadmap names no assembler; a round-trip test over a case study, as Sonnet proposes, is the right acceptance test but it needs a case study with real fields, and that study is itself unscheduled work.
- Rebuttal on decision 10: layer 2 is additive, so landing it after v3 needs no breaking window; only its syntax must be fixed in the v3 spec so that the layer-1 type spelling does not change when the assembler arrives, which is Opus's step 6 and my round-1 step 3.
- Agreement with Sonnet's unit rule: layer 2 ships with both of its generators and the round-trip test or not at all, and the derived-attribute rule never ships alone.
- Final position: option D: layer 1 and the allocator domain fix in v3; layer 2 (Q10.2 and Q10.3 together) fully specified in the v3 spec and landing as the first 3.x minor release with assembler, decoder, and round-trip test as one unit (changed: the Q10.3 rule moves out of v3 to join layer 2).
- Confidence: medium; the split is a sequencing judgment, and a named first user of binary output would pull layer 2 into v3.
- Still divides: whether layer 2 lands inside v3 (Sonnet) or as the first 3.x release (Opus, Fable); the evidence difference is whether a scheduled minor release counts as "dates move" or as "scope cut".

## Escalation to the owner

- None.
