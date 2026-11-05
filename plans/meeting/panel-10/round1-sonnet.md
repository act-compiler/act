# Panel 10 round 1: Sonnet (ISA author who also needs an assembler and disassembler)

## Frame

- Owner decision 38 makes widths, signedness, and encodings optional layers on top of a default of 64-bit integers.
- Layer 0: every attribute and operand is a signed 64-bit integer; this is today's behavior (taidl/taidl/instruction.py:399-400, INT64 = range(-(2**63), 2**63)).
- Layer 1: an optional width and signedness per attribute.
- Layer 2: an optional per-instruction encoding block.
- Layer 3: derived attributes, meaningful only when layer 2 exists.
- A specification that stops at layer 0 or 1 behaves exactly as it does today; a layer adds checks and generators, never a change of meaning below it.
- Every answer below is argued against this frame and against decisions 16, 30, 34, and 35.

## Facts all answers rest on

- Attributes are bare name lists: `computation_attr` and `addressing_attr` are `list[str]` (taidl/taidl/accelerator.py:62-63, instruction.py:62-74), and names carry no meaning (decision 34).
- The backend declares each computational attribute as `int64` (act-backend/act_backend/malloc_instructions_h_generator.py:451).
- The backend declares each addressing attribute as `MAKE_INT_VAR(0, INT64_MAX)` (malloc_instructions_h_generator.py:456), which forbids negative values and contradicts decision 34 even at the default.
- Allocation is a constraint-solver problem over OR-tools integer variables (act-backend/act_backend/generic/cpp/malloc/include/solver.h:16).
- Backend output is text: `op_name(attr = value, ...)` built from `std::to_string` (templates/instructions.h.INSTRUCTION_CLASSES.class.txt:10, str_piece.txt:1).
- No generator today consumes or produces bits, and the roadmap tables have no item naming an encoding or an assembler (grep of plans/roadmap/*.md finds none).

### Q10.1: How does an ISA specification declare the bits of each attribute?

- Position: A, as layer 1, declared apart from the attribute name lists, with the 64-bit signed default when absent.
- Argument:
  1. A field has a bit count and a signedness; a value outside it cannot be encoded, so the check is a constraint on the attribute value.
  2. Decision 30 already says implicit constraints and markers are one mechanism; a width expands to `lo <= @x and @x <= hi`, and a violation is handled like any violated constraint, a defined error (decision 16).
  3. A range (option B) is already expressible as a constraint, for example `@c.r0 <= 16 and @c.cb0 <= 64` in the AMX sketch, so B adds no mechanism (plans/casestudies/amx.md:25-26).
  4. A range cannot give the layout: a range of 0..16 derives 5 bits, but the field may be 8; the width is a fact about the binary, not about the values, so it must be declared (option B loses it).
  5. Option C (widths only in the encoding annotation) puts a value check inside an optional layer, so the oracle and backend could not enforce widths without an encoding; widths in layer 1 let them check an attribute before any encoding exists.
  6. Names carry no meaning (decision 34), so a width cannot hide in a name; and `check_name` governs names (instruction.py:73-74), so a suffix like `addr:u12` is not a name; a separate mapping, such as a call `i.set_widths({'addr': 'u12'})`, keeps the existing lists and the default unchanged.
  7. Signedness is part of layer 1 because a signed field holds negatives in two's complement, which decision 34 allows; the declared type sets the allowed attribute range, and the allocator domain must follow it instead of the fixed `0..INT64_MAX` (malloc_instructions_h_generator.py:456).
  8. Decision 34 limits implicit constraints to the evaluated address; a width constraint is different, since it applies to the attribute value because the field holds that value, and it exists only if the author declared it.
  9. Case studies show the need is real but small: Hopper `wgmma` has five one-bit style immediates now pinned by constraints (plans/casestudies/hopper.md:108), and AMX `ldtilecfg` has byte and 16-bit fields in a 64-byte block (plans/casestudies/amx.md:95).
- Evidence for practice:
  - LLVM TableGen declares `bits<N>` fields and sets instruction bits from them (https://llvm.org/docs/WritingAnLLVMBackend.html, https://llvm.org/docs/TableGen/ProgRef.html).
  - The riscv-opcodes tables imply immediate width only from field names and bit positions, which the project README admits does not state width or signedness (https://github.com/riscv/riscv-opcodes); an explicit width avoids that gap.
  - Sail types bitvectors with a length, `bits('n)` (plans/priorart/sail.md, https://github.com/rems-project/sail).
- Confidence: high on A over B and C; medium on the exact syntax.
- What would change my mind: a target whose field width varies with state or mode, which a static declaration cannot state (then widths would become constraint-valued).

### Q10.2: How are binary encodings annotated, and which generators consume them?

- Position: A, layer 2: an optional per-instruction encoding block inside the specification, read by one assembler and one disassembler generated from the same map.
- Argument:
  1. An encoding is a fact about one instruction; a separate file keyed by name (option B) can drift from the instruction it describes, and a rename breaks it silently.
  2. LLVM keeps the encoding in the instruction record: `Inst` bits are assigned beside the operands, and an operand fills the bits whose name it matches (https://llvm.org/docs/WritingAnLLVMBackend.html).
  3. RISC-V's table is one line per instruction: name, field names, fixed bits `msb..lsb=value`; one source yields C, Rust, and HDL outputs (https://github.com/riscv/riscv-opcodes).
  4. Sail's `encdec` is one bidirectional mapping, so one definition both decodes bits and encodes an AST (https://github.com/rems-project/sail-riscv); the TAIDL block should likewise be one map with two readers, so encode and decode cannot disagree.
  5. The block lists: instruction word length, fixed bits (opcode, reserved bits with their value), and for each encoded attribute its bit ranges, high to low; a split field (RISC-V `bimm12hi` and `bimm12lo`) is two ranges of one attribute.
  6. The map is a pure bit placement with no scale or offset; arithmetic such as `*8` lives in constraints (Q10.3), so one mechanism covers arithmetic (decision 30).
  7. Decision 16 forbids undefined behavior, so every bit of the word is covered once or fixed; an uncovered bit is a generation error, and a word that matches no instruction decodes to a defined decode error.
  8. Overlapping opcodes between two instructions are a generation-time error, found by checking that no word satisfies two instructions' fixed bits and field ranges.
  9. The first consumer is the assembler and disassembler; the backend keeps printing text, which already carries every attribute (class.txt:10), and the assembler turns that text into words.
  10. The fsim consumer is not yet real: the NKI driver records a Python-level trace (plans/oracle/fsim/integration.md:27-30), and the seam to a trace emitter is the instruction record, not binary (plans/oracle/fsim/implications.md:34); a binary trace decoder would be a later reader of the same block.
  11. Fixed-length words only in the first version; variable-length encodings (x86, AIE-ML bundles) are an extension of the same block, not a reason to delay it.
- Confidence: medium-high on A; medium on the field list.
- What would change my mind: a case study whose instruction word depends on state (modes that change field meaning), which a static map cannot state.

### Q10.3: How are attributes that never appear in the binary handled?

- Position: A with a sharpening: when an encoding block exists, the block lists the encoded attributes and every other attribute is derived; every derived attribute must be uniquely determined by the encoded ones through constraints, checked at generation time.
- Argument:
  1. The distinction needs layer 2: without an encoding no attribute is "in the binary", so a mark on the attribute (option A as worded) would add an annotation with no meaning in layers 0 and 1; the block already names the encoded set, which makes option B's lack of a separate mark correct and removes a second source of truth.
  2. Uniqueness (option A over C): an assembler writes one word and a disassembler reads one word; if two values of a derived attribute satisfy the constraints, the decoded text depends on a tie-break rule, which decision 16 says must be a simple norm that is easy to explain, and no such rule is evident.
  3. Check form: for each derived attribute `D`, a constraint of the form `D == f(encoded)` or `c*D + k == g(encoded)` with a nonzero literal `c`, which is injective in `D`; the generator rejects any derived attribute without such a constraint, with a message naming it (a loud failure, as engineering.md asks).
  4. Solving is closed form: `D = (g - k) / c`; no solver runs in the assembler or disassembler; the integer division is exact, and an inexact division means the encoded values have no derived partner.
  5. No solution (for example `A == B*8` with `A = 12`) is a violated constraint, hence a defined error (decisions 16, 30); the divisibility requirement is thus a property of the encoded value, not undefined behavior.
  6. The backend allocator keeps solving over all addressing variables with the linking constraint included (solver.h:16 and the `floor_division` helper), so it needs the width domains from layer 1 and nothing new for derivation; the encoded value comes out of the same search.
  7. The oracle receives text-level attribute values, all of them; if it ever receives binary, the disassembler first computes derived values with the same closed form, so oracle and backend see one definition.
  8. Negative values: a signed width holds a negative attribute as two's complement; an unsigned width rejects it by the width constraint; a derived attribute has no width and stays 64-bit signed, so `B*8` of a negative `B` is fine, and decision 34's evaluated-address check still bounds regions.
  9. Overflow of a derived attribute past 64 bits is a defined checked error, matching the front end's 64-bit literal rule (instruction.py:399-435).
  10. Rewrite rules that reference a derived attribute are unaffected, since text keeps all attributes (isel_ir2isa_rewrites_rs_generator.py:565-623 recovers attributes by name).
- Evidence for practice: LLVM operands that are not encoded (printed or matched only) coexist with encoded operands in one record (https://llvm.org/docs/WritingAnLLVMBackend.html), and RISC-V states which operands are fields and which fixed bits (https://github.com/riscv/riscv-opcodes).
- Confidence: medium-high on uniqueness and closed form; medium on the exact allowed constraint shapes.
- What would change my mind: a real ISA whose encoded-to-derived link is non-affine and still one-to-one (for example a lookup table), which would need a table form.

### Q10.4: What lands in v3 and what is deferred?

- Position: B, with an internal order (layer 1, then layer 2, then the derived check) and a stated unit of deferral if capacity forces a cut: layer 2 with its two generators defers together; layer 3 never ships alone.
- Argument:
  1. Under decision 38 the layers are optional and additive: a specification at layer 0 or 1 is unchanged, so landing them costs no author who does not want binaries.
  2. The agenda's option A (Q10.1 and Q10.3 in v3, encodings later) is not coherent under the layering: Q10.3 defines derived relative to an encoding block (Q10.3 step 1), so shipping it without Q10.2 ships a check with nothing to check.
  3. Deferral for capacity contradicts the standing direction that nothing is cut for capacity and dates move instead of scope (plans/meeting/panel-8/decision.md, Standing owner direction).
  4. Decision 35 says the plan covers both widths and encodings; a plan that covers them must place them, and v3 is the batched breaking release (decision 10).
  5. The risk of building without a consumer is real, since no current generator reads bits (facts above), and the roadmap has no item for it; the remedy is to land the encoding with its assembler and disassembler as one unit, with a round-trip test, `decode(encode(x)) == x`, over a case study that has real fields.
  6. Layer 1 is the cheapest and removes one defect now: the allocator's `0..INT64_MAX` domain (malloc_instructions_h_generator.py:456) should follow the declared type; that edit sits in the backend work already open for v3.
  7. Option C (all deferred) leaves decision 35 unmet and leaves the ISA author with no binary path; option A defers the item that matters most to the assembler author.
  8. Order inside v3: the width layer touches the front end, the oracle checks, and the allocator domain; the encoding block touches only the front end and two new generators, so it can start once the width declaration exists.
- Confidence: medium; high that A as worded is incoherent, medium that the whole of layer 2 fits in v3.
- What would change my mind: a roadmap sizing that shows layer 2 and its generators exceeding the Q2 budget with the assist factor measured (panel 8 Q8.9), in which case the unit-deferral rule applies.

## Escalation to the owner

- None. All answers fit decisions 16, 30, 34, 35, and 38.

## Open points for round 2

- Whether the allowed constraint shapes in Q10.3 step 3 should include a lookup table.
- Whether the width mapping is a call or a keyword on `add_instruction`.
- Whether variable-length words belong in the first encoding block.
