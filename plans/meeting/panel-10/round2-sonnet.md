# Panel 10 round 2: Sonnet (ISA author who needs an assembler)

## Agreement across the panel

- All three agree on Q10.1: width and signedness per attribute (option A), expanded to implicit constraints, default signed 64-bit.
- All three agree on Q10.2: an optional per-instruction encoding block beside the instruction, with slices and generation-time coverage and overlap checks.
- All three agree on Q10.3: the encoding block names the encoded attributes, the rest are derived, and a derived attribute needs one linear defining constraint.
- All three agree on Q10.5: a namespaced metadata block attached to the declaration it qualifies, validated only by its generator.
- All three agree that the allocator domain `MAKE_INT_VAR(0, INT64_MAX)` contradicts decision 34 (act-backend/act_backend/malloc_instructions_h_generator.py:456).

### Q10.1: bits per attribute

- Strongest opposing argument (Opus): a range adds no mechanism, a type is only shorthand for two range constraints, and `u64` cannot exist because every value must fit signed 64.
- Rebuttal to the "range suffices" half: none needed, since Opus also picks A; the width stays a layout fact that a constraint cannot state (round 1 step 4).
- Concession: the type spelling `s<w>` and `u<w>` with `w` from 1 to 64 for `s` and 1 to 63 for `u`, and the rule that a type never wraps, is cleaner than my `set_widths` mapping.
- Concession: `u64` is rejected at declaration, because a value is a mathematical integer (spec-draft/expressions.md:24-31, as Opus cites) and the oracle holds `int64` (act-oracle/taidl_to/generic/utils.py:863-869).
- Open point I keep: the declaration must live apart from the name list, so `computation_attr: list[str]` (taidl/taidl/accelerator.py:62-63) and `check_name` (instruction.py:73-74) stay unchanged.
- Final position: A as layer 1, spelled `s<w>` or `u<w>`, optional, default `s64`, expanded through the decision 30 mechanism.
- Confidence: high on meaning, medium on spelling and on where the declaration sits.
- Still divides: the declaration site (Opus: on the attribute; Fable: `bits('x', 8)` style call; me: a mapping call).

### Q10.2: encoding annotation

- Strongest opposing argument (Fable): the encoding block is also the input to a trace decoder, and a RISC-V B-type immediate needs slices plus implied constant low bits, so the map is more than a list of ranges.
- Rebuttal: implied low bits are ordinary fixed bits inside the word, so my block already carries them; no scale enters the map, because `*8` lives in the Q10.3 constraint (round 1 step 6).
- Concession (Opus): an encoded attribute must carry a declared type, and its slices must cover exactly `w` bits; my round-1 block let slice width float free of the attribute, which would let encode truncate silently.
- Concession (Fable): the two riscv-opcodes checks, every bit accounted for and no bit defined twice, are the right generation-time pair (riscv-opcodes parse.py as Fable cites); my overlapping-opcode check stays as the third.
- Divergence kept: Fable puts the assembler emitter inside act-backend; I want assembler and disassembler as one new generator reading one block, so encode and decode cannot disagree, and the backend keeps printing text (templates/instructions.h.INSTRUCTION_CLASSES.str_piece.txt:1).
- Final position: A, optional layer 2 per instruction, typed encoded attributes, three generation-time checks, defined decode error for an unmatched word.
- Confidence: medium-high.
- Still divides: which package owns the assembler, and whether fsim ever reads binary (its trace is text today, plans/oracle/fsim/architecture.md:26).

### Q10.3: attributes absent from the binary

- Strongest opposing argument (Fable): the decoder always succeeds, and the fault belongs at assembly, because the allocator can post the width domain and the link so it never emits an unencodable address.
- Rebuttal: whether decode or encode is partial depends on the orientation of the link, so neither is always total.
- Example `A == B*8` (decision 35): encode `A = 8*B` is total, decode `B = A/8` is partial, so the word `A = 12` names no `B`; Opus step 6 states this, Fable step 5 states the opposite.
- Example `B == 16*A`: decode is total and encode is partial, which is Fable's case.
- Final rule: each orientation has exactly one partial direction, and a value in that direction with a nonzero remainder or a width overflow is a defined error, named at assembly or at decode (decision 16).
- Concession (Opus): chains of derived attributes and state in the defining expression are excluded until a case study needs them, which narrows my open question on constraint shapes.
- Concession (Opus): the front end rejects a derived attribute with zero or two defining equalities, so the check is syntactic and decidable.
- Concession (Fable): the allocator should see the encoded attribute as an `IntVar` with the width domain, so search never yields an unencodable value; the closed form re-derives only (act-backend/act_backend/generic/cpp/malloc/include/solver.h:16).
- Lookup-table links are out of v3; a non-affine link is refused at generation time and the author rewrites the specification in terms of the encoded attribute.
- Final position: B (block names the encoded set) plus the uniqueness rule, linear with nonzero literal coefficient, closed-form solving, no solver in the assembler.
- Confidence: medium-high.
- Still divides: whether the divisibility fault gets its own S1.3 row (Opus) or reuses the generic violated-constraint row (me).

### Q10.5: generator metadata

- Strongest opposing argument (Opus): a metadata entry must never change the meaning of any instruction, so the encoding block cannot be a metadata entry, and run options belong to the entry point.
- Strongest argument for option C alone: a per-run choice such as `cost` already travels as a keyword (act-backend/act_backend/generator.py:79-86), so a second channel looks like duplication.
- Fair rebuttal: a keyword cannot attach data to one instruction or one buffer without restating its name, which is option B in disguise (Fable, step 1).
- Position: A for data on a node, C for run options; both are already present in the code, so nothing is invented.
- Is the encoding block a metadata entry for an assembler generator.
  - It is not: the front end must validate it (bit coverage, opcode overlap, derived-attribute uniqueness), and a metadata entry is opaque to the front end.
  - Two readers (assembler and disassembler) already share it, which meets the second-use rule that promotes a datum to a layer.
  - Widths feed the oracle and the allocator, so they are shared by construction.
- What a metadata entry is: generator-private data, such as an Allo partition or pipeline hint per buffer, that never alters a fault row of S5 to S8.
- Front-end contract.
  - Each of accelerator, buffer, and instruction takes an `addons` mapping, namespace -> plain data (str, int, bool, list, dict).
  - `validate()` publishes `addons` on each typed node (panel 3, Q3.5), and `CONTRACT_VERSION` covers the container only (panel-3/decision.md:8-9, :64-79).
  - A namespace is a plain name; no tool infers meaning from it, in the spirit of decision 34.
- Generator declaration and check.
  - A generator names its namespace and ships a typed schema; it reads `node.addons.get('<name>')` and rejects an unknown key inside its own namespace, naming instruction and key (a loud failure).
  - Namespaces owned by other generators are ignored without a message, per decision 39.
  - A metadata entry refers to widths and encodings by reading layers 1 and 2 from the typed model, never by restating them, so one fact has one place.
- Misspelled namespace.
  - Silent ignore is required by decision 39, so the remedy is a listing: `validate()` reports the namespaces present and which installed generator claims each, as a note and never an error.
  - This is my addition to Opus step 9 and costs one registry list.
- Easy to add a generator.
  - Inputs: the typed model and `validate()` only (panel 3).
  - Entry point: `generate_<name>(accelerator, base_dir=None, **options)`, the shape of `generate_oracle` (act-oracle/taidl_to/generator.py:21) and `generate_backend`.
  - Tests: a per-generator `expectations.toml` with one row per corpus case, supported or unsupported with a coded error, and `coverage.md` rendered from it by the one T11 renderer (plans/tests/generators.md, T11 and T12).
  - Neutrality test: for every corpus case that carries metadata, the oracle and backend output must be byte-identical with the metadata stripped.
  - The neutrality test is the executable form of decision 39 and needs no knowledge of the new generator.
  - Metadata cases whose result only the owning generator can judge live in that generator's own tests.
- A new generator touches neither taidl nor another generator, except for registering its namespace.
- Final position: A for per-node metadata plus C for run options, with the neutrality test and the namespace listing.
- Confidence: high on the layer-versus-metadata test and on A plus C; medium on the listing.
- Still divides: Fable's trigger for a side file when metadata entries are large tables; I would add an include of a file path inside the block, checked by the same generator, only on first need.

### Q10.4: v3 scope

- Strongest opposing argument (Opus, Fable): no generator reads binary, the roadmap has no assembler item, and panel 8 found the plan needs an assist factor of 4.3 to 5.7 (panel-8/decision.md:198), so layer 2 spends capacity on a feature with no user.
- Rebuttal: the standing owner direction says nothing is cut for capacity and dates move, not scope (panel-8/decision.md:17-19), so capacity alone cannot defer an item the plan covers.
- Rebuttal: decision 35 says the plan covers encodings and decision 10 batches breaking changes, but layer 2 is additive, so Opus is right that it does not need the v3 window.
- Concession: Fable's no-consumer point is real, and I accept it as a scheduling rule: layer 2, the derived-attribute rule, the assembler, and the disassembler form one unit, scheduled last in the v3 plan with the round-trip test, so a cut of the unit by dates is clean.
- Concession (Opus): shipping Q10.3 without Q10.2 is incoherent, so Fable's "Q10.3 rule in v3, consumers later" is not a valid split.
- Concession (Fable): the v3 specification text defines all layers so the syntax is reserved once.
- I now accept deferral of the unit past v3 as a minor release if a named assembler user is absent when the Q2 wave is fixed; before this round I held that it ships in v3.
- Final position: layer 1 in v3 wave one; layer 2, 3 and both generators as one unit, in v3 only if a named user exists, otherwise the first additive minor release; the spec text covers all layers in v3.
- Confidence: low-medium; I moved from B toward D.
- Still divides: who counts as a named user, and whether "nothing is cut" forces the unit into v3 regardless.

## Escalation to the owner

- None.
