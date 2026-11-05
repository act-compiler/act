# Panel 10 round 1: Opus

Lens: language designer and semantics author.
Every answer follows owner decisions 38 and 39: widths, signedness, and encodings are optional layers over signed 64-bit integers, and generator metadata sits beside them.
Layer 0 is today's language; layer 1 adds a range per attribute; layer 2 adds a binary encoding and derived attributes.

### Q10.1

Position: option A as an optional layer 1: an attribute may carry an integer type `s<w>` or `u<w>`, absent means `s64`, and the type expands to two implicit range constraints through the decision 30 mechanism.

1. Today an attribute is a signed 64-bit integer, and a value outside that range is its own fault row S1.3.10.
2. Decision 38 makes signed 64 the default, so S1.3.10 is already the `s64` case of a range check; one rule "an attribute value lies in the range of its type" replaces it with no new concept.
3. The type `s<w>` means the closed range from -(2^(w-1)) to 2^(w-1) - 1, and `u<w>` means 0 to 2^w - 1; w runs from 1 to 64 for `s` and 1 to 63 for `u`, so every range fits in signed 64 and `u64` is rejected at declaration.
4. The value stays a mathematical integer under S6.2: a type never wraps, truncates, or sign-extends anything, so arithmetic over attributes keeps exactly one meaning.
5. Decision 30 already says markers and region bounds are constraints with one violation rule; a type is one more shorthand of that family, so a value out of range is a fault exactly as S1.3.5 (decision 16: defined, never undefined).
6. Option B (min, max per attribute) adds nothing a written constraint `0 <= @a.x and @a.x <= 100` does not already say, so it fails minimality; a non-power-of-two bound such as AMX `rows <= 16` stays an ordinary constraint.
7. Option C (widths only in an encoding) leaves an ISA with no binary layer unable to state a field range, yet the allocator needs exactly that range as a variable domain.
8. Spelling the type with the vocabulary the oracle already uses for integers (`s64`, `u64`) avoids a second integer type syntax.
9. Decision 34 holds: a name never implies a type, and `u<w>` is how an author, not a tool, declares an attribute non-negative.
10. The backend gains a concrete win: the allocator today gives every addressing attribute the domain 0 to INT64_MAX, which both contradicts decision 34 (negative values) and gives the solver no finite bound; the declared range becomes the `MakeIntVar` domain.

Evidence:
- Step 1: act/plans/language/spec-draft/instructions.md:13, execution.md:12, overview.md:47.
- Step 3: taidl/taidl/instruction.py:400 (`INT64 = range(-(2**63), 2**63)`).
- Step 4: spec-draft/expressions.md:24-31.
- Step 5: owner decision 30; spec-draft/instructions.md:36-41.
- Step 6: act/plans/casestudies/amx.md:95.
- Step 8: act-oracle/taidl_to/generic/utils.py:863-869 (`'int64': 's64'`).
- Step 10: act-backend/act_backend/malloc_instructions_h_generator.py:456 (`MAKE_INT_VAR(0, INT64_MAX)`); act-backend/act_backend/generic/cpp/malloc/include/solver.h:16.
- External: Sail keeps integers (`int('n)`) apart from bitvectors (`bits('m)`), and "explicit zero/sign extension and truncation functions must be used" at the boundary; https://github.com/rems-project/sail/blob/sail2/doc/asciidoc/language.adoc.
- External: sail-riscv holds the I-type immediate as `bits(12)` and applies `sign_extend(imm)` only in `execute`; TAIDL attributes are integers, so signedness reduces to the range; https://github.com/riscv/sail-riscv model/extensions/I/base_insts.sail lines 143 and 159 at commit dc286e66.

Confidence: high on the meaning (range constraint, no wrap); medium on the `s<w>`/`u<w>` spelling.
What would change my mind: a case-study field whose legal values are not a contiguous range and that no single constraint can state, or an attribute whose value the hardware does wrap and whose wrapped value the semantics reads.

### Q10.2

Position: option A as optional layer 2: an encoding block on the instruction, giving a word width, fixed bits, and for each encoded attribute an ordered list of bit slices; the first consumer is an assembler generator, the decoder second.

1. An encoding is a property of one instruction, as TableGen puts `let Inst{31-30} = op` on the instruction record and Sail puts `mapping clause encdec` next to the instruction's union clause; a separate file (option B) keys two sources of truth by a name string and drifts.
2. Slices, not one contiguous range, are needed: RISC-V B-type scatters one immediate as `imm[11] @ imm[9..4] ... imm[3..0] @ imm[10]`, and Hopper splits a descriptor into three 14-bit fields.
3. Packing has one meaning: an encoded attribute of type `s<w>` or `u<w>` (Q10.1) is written as its w-bit two's complement or unsigned image, and its slices must cover exactly w bits; an encoded attribute without a declared type is rejected, because 64 bits rarely fits a field.
4. The front end checks, so no generator meets an undefined case: slices and fixed bits cover each bit of the word exactly once, and no two instructions' fixed-bit patterns can match one word, so decoding is deterministic.
5. Decode of a word that matches no instruction is a defined decode error, the analogue of a Sail mapping with no matching clause.
6. Consumers today: the backend prints text assembly with attribute names, and the fsim NKI driver reads a text trace; no current generator reads or writes binary, so the assembler is the first real consumer and decoding has no v3 consumer.
7. Option C (design study only) is weaker than needed, since decision 35 already commits the plan to encoding annotations; what remains open is when they land (Q10.4), not whether.

Evidence:
- Step 1: LLVM "Writing an LLVM Backend", `bits<5> rd;` with `let Inst{29-25} = rd;` and "an operand from the outs or ins list is expected to have a matching name"; https://llvm.org/docs/WritingAnLLVMBackend.html.
- Step 2: sail-riscv model/extensions/I/base_insts.sail:112-113 at commit dc286e66; PTX matrix descriptor bits 13-0, 29-16, 45-32 per https://forums.developer.nvidia.com/t/wgmma-matrix-start-address/339466 and https://reviews.llvm.org/D157382.
- Step 5: Sail mappings are bidirectional relations; https://github.com/rems-project/sail/blob/sail2/doc/asciidoc/language.adoc.
- Step 6: act-backend/act_backend/templates/instructions.h.INSTRUCTION_CLASSES.str_piece.txt:1; act/plans/oracle/fsim/architecture.md:26-27.
- Step 7: owner decision 35.

Confidence: high on per-instruction placement and slices; medium on fixed-length words only (multi-word encodings may need a word list).
What would change my mind: an accelerator in the case studies whose encoding depends on other instructions or on state (a mode bit), which a per-instruction static block cannot express.

### Q10.3

Position: option B with a decidable uniqueness rule: the encoding block's field list is the only marking, and each derived attribute needs exactly one defining equality, linear in that attribute with a nonzero integer coefficient, over encoded attributes and literals.

1. A separate encoded/derived mark (option A) restates what the encoding block's field list already says; minimality wants one source of truth, so an attribute is encoded iff a slice names it, and derived otherwise.
2. With no encoding layer (decision 38 default), no attribute is derived: every attribute is given at issue, exactly as today.
3. Generators need a closed form, not a search: the oracle given a binary must recover B from A, and the assembler must compute A from B; a general constraint set over `*`, `/`, `%` is nonlinear integer arithmetic, which has no general decision procedure.
4. The rule: a derived attribute `B` has one equality constraint of the form `c * B + E1 == E2`, with c a nonzero literal, and E1, E2 over encoded attributes and literals only (either side may be written first, as `@a.A == @a.B * 8` or `@a.B == 16 * @a.A`).
5. Encode direction is evaluation: E2 - E1 given B; then A must satisfy its type range (Q10.1) or the issue faults, which is the backend's ordinary range constraint.
6. Decode direction is exact division: B = (E2 - E1) / c when (E2 - E1) % c == 0; a nonzero remainder means the binary names no B, and that is one new defined fault row in S1.3, never undefined.
7. With c nonzero, B is an injective function of the encoded values, so "several solutions" (option C) cannot arise; option C would make each generator implement one tie-break identically, which a CP solver does not promise.
8. The front end rejects a derived attribute with no defining equality, with two, or whose equality is nonlinear in it; this is a syntactic check, decidable and cheap, and it is a front-end rejection, not a run-time case.
9. Negative values (decision 34): a derived attribute has no type of its own and may be negative; an encoded attribute's sign follows its `s<w>` or `u<w>`; the implicit region constraints still apply to evaluated starts, never to either attribute.
10. Sail encodes the same pattern: `BTYPE` holds `bits(13)` while the binary holds 12 bits, and `encdec` matches `imm @ 0b0`, so a byte offset with low bit set has no encoding; that is exactly step 6.
11. The allocator keeps both attributes as solver variables and posts the defining equality as one more linear constraint, which its CP solver already handles.
12. Chains (derived from derived) and state in the defining expression are left out until a case study needs them, per the second-use-site rule.

Evidence:
- Step 3: SMT-LIB separates QF_LIA, "Boolean combinations of inequations between linear polynomials over integer variables", from QF_NIA; https://smt-lib.org/logics.shtml; general Diophantine solvability is undecidable (Matiyasevich), https://en.wikipedia.org/wiki/Hilbert%27s_tenth_problem.
- Step 4: owner decision 35 (`A` and `B*8`); spec-draft/expressions.md:24 (operators).
- Step 6: owner decision 16; spec-draft/overview.md:28-33.
- Step 9: owner decision 34; spec-draft/instructions.md:29.
- Step 10: sail-riscv model/extensions/I/base_insts.sail:101 and 112 at commit dc286e66.
- Step 11: act-backend/act_backend/malloc_instructions_h_generator.py:286-396 (constraints pushed as solver constraints); generic/cpp/malloc/include/solver.h:16-17.
- Hopper descriptor as a second instance (address stored divided by 16): https://forums.developer.nvidia.com/t/wgmma-matrix-start-address/339466.

Confidence: high that uniqueness must be a front-end check and that option C breaks one-meaning; medium on the exact linear form as the boundary.
What would change my mind: a case-study encoding whose derived value needs a non-linear inverse (for example a log2 size field, `size == 2^k`) that is common enough to warrant a second form, such as a declared decode expression checked against the constraint.

### Q10.5

Position: option A: an optional `addons={'<generator>': {...}}` keyword on the accelerator, each buffer, and each instruction, stored opaquely by the front end and checked only by the generator that owns the namespace; run options stay entry-point keywords (option C's proper role).

1. The test that separates a layer (decision 38) from a metadata entry (decision 39) is meaning: a layer changes what every conforming generator must do (a range constraint, a decode fault), while a metadata entry may never change the meaning S5 to S8 give any instruction.
2. So widths, encodings, and derived attributes are core layers, not assembler metadata: a derived attribute changes which issues fault, and the oracle and the backend must agree on it.
3. A metadata entry carries only what one generator needs to produce its artifact, such as an Allo partition or pipeline hint per buffer; it is data for a target, not semantics.
4. MLIR draws the same line: inherent attributes define the operation and are verified by it, while discardable attributes carry a dialect prefix and "the dialect indicated by the dialect prefix is expected to verify these attributes".
5. Attaching the metadata to the declaration it qualifies (option A) avoids a second name-keyed file (option B) that drifts on a rename, the same argument as for encodings in Q10.2.
6. Option C cannot attach data to one buffer or instruction without restating names, but it is right for run choices: `generate_backend(..., cost=...)` is already such an option and is not an ISA fact.
7. The front end checks only that each namespace is a plain name and each value is plain data (str, int, bool, list, dict), so the typed model of U2 can expose `addons` as one member and serialize it later.
8. A generator's contract is then short: call `validate()`, read the typed model, check its own namespace and reject unknown keys there, honor S1.2 by naming every unsupported feature, and ship its expectations table (T11) and coverage rows.
9. A misspelled namespace is silently ignored by design (decision 39); each generator should list the namespaces it reads in its docs, and a front-end listing of all namespaces present lets an author spot an unclaimed one.
10. Adding `addons` is additive and small, and the U2 typed contract lands in v3, so the member should land with U2 to avoid a later contract bump.

Evidence:
- Step 1: owner decisions 38 and 39; spec-draft/overview.md:22-24 (a generator gives each feature the meaning S5 to S8 define, or rejects it).
- Step 4: https://mlir.llvm.org/docs/LangRef/ (inherent and discardable attributes).
- Step 3: Allo is "an Accelerator Design Language (ADL) and compiler" with its own schedule primitives; https://cornell-zhang.github.io/allo/ and https://github.com/cornell-zhang/allo.
- Step 6: act-backend/act_backend/generator.py:79-86; act-oracle/taidl_to/generator.py:21.
- Steps 7 and 10: act/plans/meeting/panel-3/decision.md:8-9 and 67; act/plans/shared/overhaul.md:31.
- Step 8: act/plans/tests/generators.md:3-4 and 21.

Confidence: high on the layer-versus-metadata test; medium on the keyword spelling and on plain-data values.
What would change my mind: a metadata entry that a second generator also needs, which per the second-use-site rule should be promoted to a core layer with one meaning rather than duplicated across namespaces.

### Q10.4

Position: option D: layer 1 (Q10.1) lands in v3 across front end, oracle, and backend; layer 2 (Q10.2 and Q10.3 together) is specified in the v3 plan but lands after v3 as an additive minor release.

1. Q10.3 has no meaning without Q10.2: under my Q10.3 position the encoding block is what makes an attribute derived, so the two must land together, which rules out option A as written.
2. Layer 1 is small: one optional declaration, two implicit constraints per typed attribute, one generalized fault row, and a variable-domain change in the allocator.
3. Layer 1 also carries a v3 correctness fix the backend needs anyway: the allocator's lower bound 0 on addressing attributes contradicts decision 34.
4. Both layers are optional with a 64-bit default (decision 38), so neither breaks an existing ISA specification; decision 10 batches breaking changes into v3, and an additive layer does not need that window.
5. No v3 generator consumes encodings: the backend emits text assembly and fsim reads a text trace, so shipping layer 2 in v3 spends capacity (decision 11) on a feature with no user until an assembler exists.
6. Writing the layer 2 semantics into the v3 spec now fixes its meaning (packing, decode determinism, the derived-attribute rule, its fault row) so that layer 1 choices, such as the `s<w>`/`u<w>` type, do not need to change when layer 2 lands.

Evidence:
- Step 3: act-backend/act_backend/malloc_instructions_h_generator.py:456; owner decision 34.
- Step 4: owner decisions 10 and 38.
- Step 5: act-backend/act_backend/templates/instructions.h.INSTRUCTION_CLASSES.str_piece.txt:1; act/plans/oracle/fsim/architecture.md:27; owner decision 11.
- External: LLVM keeps encoding as a separate backend (CodeEmitterGen, DisassemblerEmitter) over the same instruction records, so encoding can be added to existing records later without changing them; https://llvm.org/docs/TableGen/BackEnds.html.

Confidence: medium; the split is a capacity judgment once the dependency in step 1 is accepted.
What would change my mind: a named v3 deliverable that needs binary output (for example a hardware team that must assemble programs for a v3 demo), or a panel 8 roadmap slot that already sizes the assembler for the two quarters.

## Escalation to the owner

None.
