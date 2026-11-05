# Panel 10, round 1, Fable: attribute bits and binary encodings

Lens: compiler and assembler architecture (TableGen, Sail, binutils opcode tables, vendor assemblers).
Binding: decisions 16, 30, 34, 35, 38 (widths, signedness, and encodings are optional layers over a 64-bit integer default), and 39 (generator-specific metadata).

## Layered specification, the frame for every question

| Layer | Content | Default when absent |
| ----- | ------- | ------------------- |
| 0 | Attributes as today: integers, constraints, regions | Every attribute is a signed 64-bit integer |
| 1 | Width and signedness per attribute | None; layer 0 bounds apply |
| 2 | Per-instruction encoding block: opcode, fixed bits, field map | No binary form; text assembly only |

- Layer 0 is what exists: literals are checked against `INT64 = range(-(2**63), 2**63)` (taidl/taidl/instruction.py:400), the oracle warns past 64 bits (act-oracle/taidl_to/generic/utils.py:285-287), and the allocator's attribute domain is `MAKE_INT_VAR(0, INT64_MAX)` (act-backend/act_backend/malloc_instructions_h_generator.py:455).
- Every layer-1 and layer-2 fact becomes a constraint or a mapping in the mechanism of decision 30, never a new checking path.

### Q10.1

Position: D, layered: layer 1 is option A (width and signedness per attribute) and expands to one implicit constraint per attribute; ranges (option B) stay ordinary constraints; the default is the 64-bit integer of decision 38.

1. Every toolchain describes an immediate by width and signedness, and derives the range: TableGen's `bits<n>` is "a fixed-sized integer of arbitrary length n that is treated as separate bits" (llvm.org TableGen ProgRef, fetched), AArch64's operand classes are `class UImmScaledOperand<int Width, int Scale>` (llvm-project AArch64InstrFormats.td, fetched), and Sail types immediates as `bits('n)` (priorart/sail.md:7).
2. A range (option B) cannot recover width when a value is skipped, and a width always yields a range, so width is the primitive and range the derived form; an author who needs `1 <= rows <= 16` writes it as a constraint today (casestudies/amx.md:21-24 shows 17 such attributes).
3. Option A as an implicit constraint costs nothing new: `bits('x', 8)` expands to `0 <= @c.x and @c.x < 256`, the same expansion markers get (decision 30), so the backend's allocator domain, the backend's precondition checks, and the oracle's call-time check all enforce it without a fourth code path (act-backend/act_backend/isel_ir2isa_rewrites_rs_generator.py:220-260; act-oracle/taidl_to/generic/utils.py:274-287).
4. Option C (widths only inside an encoding) fails decision 16 for a spec without an encoding block: an AMX `ldtilecfg` row count is one byte of the 64-byte block (casestudies/amx.md:95), and an author who declares that byte should get a defined error from the oracle even before anyone writes the opcode.
5. Decision 34 is honored: a signed width declares the attribute's value range, while the implicit address constraints apply to the evaluated start; both are constraints and a violation of either is one defined error.
6. The allocator's present domain `(0, INT64_MAX)` contradicts decision 34 for a negative attribute, so layer 0's default must be stated as signed 64-bit and the generated domain widened to it, with layer 1 narrowing it.

Evidence: taidl/taidl/instruction.py:399-400; act-backend/act_backend/malloc_instructions_h_generator.py:445-457; act-oracle/taidl_to/generic/utils.py:274-287; casestudies/amx.md:21-24, :95; priorart/sail.md:7; llvm.org TableGen ProgRef (bits type); llvm-project AArch64InstrFormats.td (UImmScaledOperand).
Confidence: high.
Would change my mind: a case study instruction whose legal values form a non-interval set the author must spell out and that a width plus constraints cannot express compactly.

### Q10.2

Position: A, an optional per-instruction encoding block in the ISA specification (opcode and fixed bits, then attribute to bit-slice map), checked at generation time for complete and non-overlapping coverage; consumers are an assembler emitter in act-backend and a decoder for a trace driver; the oracle never reads it.

1. Every mature description keeps the encoding beside the instruction, not in a second file: TableGen sets `Inst{...}` slices in the instruction record and emits both the code emitter and the disassembler tables from it (llvm.org TableGen BackEnds, "CodeEmitter ... given a MachineInstr, returns the value of the instruction" and "disassembler table emitters", fetched); Sail's `mapping clause encdec` is written once and is bidirectional (riscv/sail-riscv ReadingGuide, fetched).
2. riscv-opcodes is the one counterexample of a separate file (option B), and it works because that repository is the source of truth and generates `encoding.out.h` for spike, pk, and the LaTeX tables (github.com/riscv/riscv-opcodes README and Makefile, fetched); TAIDL's source of truth is already the ISA specification (decision 32), so a second keyed file is a drift path with no owner.
3. The generation-time check comes from the same repository: "all bit positions must be accounted for (either as args or constant value fields)" and "values for the same bit positions should not be defined multiple times" (riscv-opcodes parse.py, fetched); TAIDL's front end applies the two checks to each block.
4. Split and implicit bits are needed on day one: a RISC-V B-type immediate is `imm7[6] @ imm5[0] @ imm7[5..0] @ imm5[4..1] @ 0b0` (mpact-sim generator.bin_fmt, fetched), so a field map is a list of slices per attribute plus implied constant low bits, and the implied low bits are a Q10.3 scaling.
5. First consumer, the assembler: the backend today prints each attribute with `str += "{{label}}" + std::to_string({{value}})` (act-backend/act_backend/templates/instructions.h.INSTRUCTION_CLASSES.str_piece.txt:1, rendered at malloc_instructions_h_generator.py:500-507); a binary emitter is the same walk over attributes writing slices instead of decimal text.
6. Second consumer, the decoder: the fsim NKI driver parses a text trace into instruction records and checks limits per record (oracle/fsim/architecture.md:26, :116); a binary trace needs the inverse map, which the same block gives, as Sail's one mapping gives both directions.
7. The oracle does not consume the block: it executes calls by attribute value (oracle/fsim/implications.md:34, "one instruction record"), so the block is an optional layer (decision 38) and a spec without it loses nothing the oracle does.

Evidence: llvm.org TableGen BackEnds (CodeEmitter, Disassembler); riscv/sail-riscv ReadingGuide (mapping clauses bidirectional); riscv-opcodes README, Makefile, parse.py checks; mpact-sim generator.bin_fmt (B-type overlay); act-backend templates str_piece.txt:1; act-backend/act_backend/malloc_instructions_h_generator.py:500-507; oracle/fsim/architecture.md:26, :116; oracle/fsim/implications.md:34.
Confidence: high on A and on the two checks; medium on the decoder consumer, since fsim's trace is text today.
Would change my mind: a vendor whose encodings are secret and documented only as an opcode table outside the ISA document, where a separate keyed file matches how the author receives the data.

### Q10.3

Position: B plus the uniqueness rule of A: the encoding block names the encoded attributes and every other attribute is derived; a derived attribute must be uniquely determined by the encoded ones through the constraints, checked at generation time in closed form for the affine case, and a value no encoding reaches is a defined assembly-time error.

1. The `A == B*8` case is the scaled immediate every assembler already owns: AArch64 `UImmScaledOperand<Width, Scale>` (llvm-project AArch64InstrFormats.td, fetched), the MIPS MSA rule that load and store immediates are "divided/multiplied by the size of the data format before encoding and after decoding" (llvm-project commit, searched), and binutils' `OPD_F_SHIFT_BY_2` flags with `operand_need_shift_by_two` (binutils include/opcode/aarch64.h, searched); none of them runs a solver, all invert a linear map.
2. act-backend already performs this inversion for computational attributes: `attribute_factor` recovers an attribute from a dimension of the form `@c.x * k` and divides by `k` in the rewrite (act-backend/act_backend/isel_ir2isa_rewrites_rs_generator.py:166-202); the derived-to-encoded step is the same function applied to the linking constraint.
3. No marking on the attribute (option B) keeps layer 0 unchanged and reads as the vendor documents read: the field table lists encoded names, and a name absent from the table is by construction derived; a mark would duplicate the table and could disagree with it.
4. Uniqueness is a generation-time check, not a run-time one: for each derived attribute the front end needs one linking constraint of the form `encoded == derived * k + c` or `derived == encoded * k + c` with integer `k != 0`; a derived attribute with no such constraint, or two, is a specification error, which matches Sail's exhaustiveness warnings (priorart/sail.md:21).
5. Solving direction follows the consumer: the backend chooses derived values (addresses) and the assembler computes the encoded value by the closed form, rejecting a value `k` does not divide or that leaves the field width as a defined error (decision 16), as every assembler rejects an unencodable immediate (meeting/panel-1/round1-fable.md:135); a decoder computes derived from encoded by the forward map, which always succeeds.
6. The allocator can take the encoded attribute as its own `IntVar` with the width domain and the linking constraint, so the search never produces an unencodable address (constraint.cc already adds linear equalities, act-backend/act_backend/generic/cpp/malloc/src/constraint.cc:47-103); the closed form then only re-derives what the solver already chose.
7. Option C (several solutions, one chosen) names no rule a decoder can apply in reverse, so round trips would not be identities; a non-affine link is refused at generation time, and the author rewrites the spec in terms of the encoded attribute.
8. Negative values (decision 34) and widths compose: a signed field of width `w` bounds the encoded value to `[-(2**(w-1)), 2**(w-1))`, the link maps it to the derived value, and the implicit address constraints bound the evaluated start; a derived attribute may be negative while its encoded form is a two's-complement field, exactly as a branch offset is.

Evidence: llvm-project AArch64InstrFormats.td (UImmScaledOperand); MIPS MSA commit note (searched); binutils aarch64.h `OPD_F_SHIFT_BY_2`, `operand_need_shift_by_two`; act-backend/act_backend/isel_ir2isa_rewrites_rs_generator.py:166-202; act-backend/act_backend/generic/cpp/malloc/src/constraint.cc:47-103; priorart/sail.md:21; meeting/panel-1/round1-fable.md:135.
Confidence: high on B plus uniqueness and the affine closed form; medium on routing the encoded attribute through the allocator rather than checking after allocation.
Would change my mind: a case-study instruction whose binary field is a non-affine function of the natural attribute (for example a log2 field or a lookup table), which would need either a table link form or a second link kind.

### Q10.5

Position: A plus C: a namespaced metadata block per generator inside the ISA specification, at accelerator, buffer, or instruction level, that the front end stores opaquely and the named generator validates; the generator's entry point takes the metadata through the typed model and keeps keyword arguments for per-run options only.

1. The precedent in both generators is already option C for per-run options: `generate_backend(accelerator, base_dir, cost)` takes a cost library name (act-backend/act_backend/generator.py:79-84) and `generate_oracle(accelerator, base_dir)` takes none (act-oracle/taidl_to/generator.py:21); a per-instruction datum cannot travel that way without a parallel dictionary keyed by instruction name, which is option B in disguise.
2. Option B is the drift path of Q10.2 step 2: a file keyed by instruction and buffer names has no owner when a name changes, while a block written beside `add_instruction` is renamed with it.
3. TableGen carries target-specific data inside the shared record as `bits<64> TSFlags` (llvm.org TableGen ProgRef, Appendix C sample record, fetched): the generic emitters ignore it and one target's code reads it, which is the shape decision 39 describes.
4. The front end's part is placement and naming only: `Accelerator.add_addon(generator, mapping)`, `Instruction.add_addon(...)`, and the buffer form store `mapping` under the generator's name, `validate()` (panel 3, Q3.5) publishes `addons: dict[str, Mapping]` on each typed node, and `CONTRACT_VERSION` (Q3.4) covers only the container, never the contents.
5. The generator's part is the schema: `generate_allo()` reads `node.addons.get('allo')`, validates it against its own typed schema, and raises a defined error naming the instruction and key, so a metadata entry another generator does not know is ignored by construction (decision 39) and a typo is caught by the one generator that cares.
6. Relation to decision 38: layers 1 and 2 are shared layers, since widths feed the oracle, the allocator, and an assembler, and the front end validates them; a metadata entry is a generator-private layer with the same optionality; the promotion rule is that a metadata entry two generators read becomes a shared layer with front-end validation, which is how Allo's own targets (AMD and Intel FPGAs, Ryzen NPUs; github.com/cornell-zhang/allo README, fetched) would share buffer-placement metadata if act-allo and a later HLS generator both read it.
7. What makes a generator easy to add: the typed model and `validate()` as the only input (panel 3), a per-generator expectation table against the shared conformance corpus (tests/generators.md, tests/corpus), and the entry-point shape `generate_<name>(accelerator, base_dir, **options)`; a new generator writes its schema, its expectation table, and its emitter, and touches neither taidl nor another generator.

Evidence: act-backend/act_backend/generator.py:79-91; act-oracle/taidl_to/generator.py:21; meeting/panel-3/decision.md:6-9, :64-79; shared/README.md:13-14; tests/generators.md; llvm.org TableGen ProgRef Appendix C (TSFlags); github.com/cornell-zhang/allo README.
Confidence: high on A for per-node data and C for per-run options; medium on the exact typed exposure through `validate()`, which panel 3 owns.
Would change my mind: a generator whose metadata entries are large tables (hundreds of lines per instruction), where a side file with a checked name cross-reference would read better than an inline block.

### Q10.4

Position: D, layered: the v3 specification defines all three layers so the breaking release carries the syntax (decision 10), the generators implement layer 1 (widths) and the Q10.3 rule in v3, and the layer-2 consumers (binary assembler, decoder) land after v3.

1. Layer 1 is small: a width expands to an implicit constraint (Q10.1 step 3), and the 64-bit default already exists in all three code bases (instruction.py:400, utils.py:285-287, malloc_instructions_h_generator.py:455), so the work is one declaration form plus the allocator domain fix of Q10.1 step 6.
2. The Q10.3 rule is a front-end check with an existing inversion routine (isel_ir2isa_rewrites_rs_generator.py:166-202), and it must exist in v3 because an ISA specification that holds `B` and encodes `A` is exactly what decision 35 admits.
3. Layer-2 syntax belongs in the v3 spec because reserving it later would be a second breaking change, which decision 10 batches into one; a block the generators ignore costs only the two coverage checks of Q10.2 step 3.
4. No generator consumes binary encodings today: the backend prints text (str_piece.txt:1) and fsim's driver parses a text trace (oracle/fsim/architecture.md:26), and the roadmap has no assembler or decoder item (grep of plans/roadmap for encod and assembler finds none; roadmap/q1-core.md:13 and q2-core.md:53 cover 64-bit checks only).
5. Decision 11 sizes the roadmap for 3-4 engineers over two quarters, and panel 8 found the plan already needs an assist factor of 4.3 to 5.7 against unassisted weeks (meeting/panel-8/decision.md:198), so the binary emitter and decoder, each a new generator output with its own tests, cannot join the v3 waves.
6. Layering (decision 38) makes the deferral safe: a spec without a layer-2 block is complete, and the first post-v3 release that emits binaries needs no spec change.

Evidence: taidl/taidl/instruction.py:400; act-oracle/taidl_to/generic/utils.py:285-287; act-backend/act_backend/malloc_instructions_h_generator.py:455, :500-507; act-backend/act_backend/isel_ir2isa_rewrites_rs_generator.py:166-202; oracle/fsim/architecture.md:26; roadmap/q1-core.md:13; roadmap/q2-core.md:53; meeting/panel-8/decision.md:198.
Confidence: high on layer 1 and the Q10.3 rule in v3; medium on specifying layer-2 syntax in v3 without an implementation, since unimplemented syntax can drift.
Would change my mind: a named first user (fsim binary traces or a vendor assembler) committed within the two quarters, which would pull the decoder or emitter into the v3 waves.

## Escalation to the owner

- None.
