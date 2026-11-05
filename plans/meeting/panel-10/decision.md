# Panel 10 decision: attribute widths, binary encodings, generator-specific data

| Question | Decision (one line) | Confidence |
| -------- | ------------------- | ---------- |
| Q10.1 | A: optional type `s<w>` or `u<w>` per attribute, default `s64`, expanding to implicit range constraints | high |
| Q10.2 | A: optional per-instruction encoding block, pure bit placement, read by one assembler and disassembler generator | high |
| Q10.3 | B plus uniqueness: the block names the encoded attributes, every other attribute is derived through one affine link | high (rule), medium-high (allocator routing) |
| Q10.4 | B: all layers in v3, layer 1 first, layer 2 with its consumers as one last unit; dates move, scope does not | medium |
| Q10.5 | A plus C: namespaced plain-data block per generator at three levels; run options stay entry-point keywords | high (shape), medium (spelling) |

## Rule set

- Layers: layer 0 is today's language (signed 64-bit attributes); layer 1 adds a type per attribute; layer 2 adds an encoding block, and the derived-attribute rule exists only inside layer 2.
- Default: an untyped attribute is `s64`; a specification with no block has no derived attribute and behaves as today.
- Types: `s<w>` for w in 1..64 and `u<w>` for w in 1..63 mean a closed range; a value never wraps, truncates, or sign-extends.
- A type is an implicit constraint (decision 30); a value outside it is the violated-constraint fault, S1.3.10 generalized from "outside signed 64 bits".
- Block checks at generation time: every word bit is covered by exactly one slice or fixed bit; no word matches the fixed bits of two instructions; each encoded attribute has a declared type whose width equals its slice total.
- The block is pure placement: no scale, offset, or implied bit; signed fields pack as two's complement; the word width is explicit and fixed length.
- Derived-attribute rule: a slice names an encoded attribute; every other attribute is derived and has exactly one constraint `E == k*D + c` or `D == k*E + c`, E one encoded attribute, k and c integer literals, k nonzero.
- Refused with a message naming the attribute: no link, two links, a nonaffine link, a link naming another derived attribute or two encoded ones.
- Encode evaluates the link; decode inverts it by one subtraction and one exact division; a value with no integer partner or outside its field is a violated constraint; a word matching no instruction is a defined decode error.
- Generator-specific data: a namespace-keyed plain-data mapping on the accelerator, each buffer, and each instruction; the front end stores and lists it, only the named generator validates it, and it never changes what S5 to S8 mean.

## Q10.1: how an ISA specification declares attribute bits

Decision:
- Option A as layer 1, spelled `s<w>` and `u<w>`; `u64` is rejected at declaration.
- The type is declared in the instruction declaration as a name-keyed mapping beside the name lists; a key naming no declared attribute is rejected; the surface spelling belongs to the panel 3 typed model.
- The allocator domain of an addressing attribute becomes the declared range (default `s64`), replacing `0..INT64_MAX`.

Reasoning:
1. Toolchains describe a field by width and signedness and derive the range: TableGen `bits<n>`, AArch64 width-parameterized operand classes, Sail `bits('n)` (priorart/sail.md:7).
2. A range cannot recover layout (0..16 needs 5 bits, the field may hold 8), while a width always yields a range, so option B is rejected; option C leaves an ISA with no encoding unable to state a field range.
3. A type expands to two range constraints through decision 30, so oracle, allocator, and front end share one violated-constraint path (decision 16).
4. One fault row already covers out-of-range values (S1.3.10, spec-draft/overview.md:47), so the type generalizes a row.
5. `u64` cannot exist because every value fits signed 64, so attribute arithmetic keeps one meaning.
6. Decision 34 holds: no name implies a type; implicit address constraints still bind the evaluated start only.
7. A name-keyed mapping with unknown-key rejection cannot drift and leaves `list[str]` lists unchanged.

Evidence (verified):
- taidl/taidl/instruction.py:400 `INT64 = range(-(2**63), 2**63)`.
- act-oracle/taidl_to/generic/utils.py:268-275 `warn_width` only warns, while spec-draft/execution.md:13 says fault; layer 1 closes the gap.
- act-backend/act_backend/malloc_instructions_h_generator.py:456 emits `MAKE_INT_VAR(0, INT64_MAX)` for every addressing attribute, :451 emits computational ones as `int64`; the 0 lower bound contradicts decision 34.

Dissent:
- Fable's `bits('x', 8)` lacked signedness and was conceded; Sonnet's separate `set_widths` call is answered by the keyed mapping.
- The implementer confirms the solver holds the full `s64` domain with headroom (solver.h:33 already uses `INT64_MAX - 1`); a domain that cannot be held is a loud generation error, never a silent clamp.
- Confidence: high on meaning, medium on spelling.

## Q10.2: how binary encodings are annotated and consumed

Decision:
- Option A as layer 2: an optional block on each instruction with the word width, fixed bits (opcode and valued reserved bits), and per encoded attribute an ordered list of slices, high to low.
- One new assembler and disassembler generator reads the typed model; the oracle never reads the block; the backend keeps printing text.
- The generator is separate from act-backend because it needs no allocation; its package home is a roadmap packaging choice.

Reasoning:
1. An encoding belongs to one instruction; a keyed side file is a second source of truth that drifts on rename.
2. TableGen sets `Inst{...}` in the instruction record and derives emitter and disassembler from it; Sail writes one bidirectional `encdec`; riscv-opcodes is a side file only because it is its own source of truth.
3. Slices are needed because a RISC-V B-type immediate and a Hopper descriptor scatter one value; placement stays pure, and scales and implied low bits are links (Q10.3).
4. An encoded attribute needs a type whose width equals its slice total; a default 64-bit attribute names no rule for which bits enter a field.
5. Decision 16 forbids undefined cases: three generation checks plus the defined decode error close every gap.
6. Disjointness uses fixed bits only; variable-length and multi-word encodings join when a case study needs them.

Evidence (verified):
- Only text emission exists: str_piece.txt:1 renders `str += "{{label}}" + std::to_string({{value}});`, built at malloc_instructions_h_generator.py:500-507, and emitter.cc:507-509 writes each `instr->str()` as an `api.<text>` call.
- fsim's driver parses a text trace (plans/oracle/fsim/architecture.md:26-27); plans/roadmap has no encoding, assembler, or disassembler item (grep).

Dissent:
- Fable put the emitter in act-backend; ruled separate, since the walk reads no allocator state.
- Multi-word encodings remain open for all three debaters.
- Confidence: high.

## Q10.3: attributes that never appear in the binary

Decision:
- Option B plus option A's uniqueness: no mark on the attribute; the slice list defines the encoded set; each derived attribute has one affine link to one encoded attribute.
- Option C is rejected: several solutions need a tie-break that a decoder cannot invert.

Reasoning:
1. A mark would restate the slice list and could disagree with it; without a block no attribute is derived.
2. With k nonzero the link is a bijection on its image, so encode and decode are closed forms and no generator runs a solver.
3. The wider form over several encoded attributes was withdrawn in round 2: one derived value would map to many encoded tuples.
4. The partial direction depends on the link: for decision 35's `A == B*8` decode is partial (A = 12 names no B), for `B == 16*A` encode is; either way it is a violated constraint (S1.3.5, decisions 16, 30), so the only new row is the decode error "no instruction matches this word".
5. Decision 34 composes: signed encoded fields are two's complement, a derived attribute may be negative or typed, region constraints bind evaluated starts only.
6. The allocator gets the encoded attribute as a variable with the type domain and the link as a linear constraint, so search never emits an unencodable value.
7. Chains, state in a link, and lookup links (a log2 field) are refused until a case study needs one.

Evidence (verified):
- isel_ir2isa_rewrites_rs_generator.py:166-181 `attribute_factor` accepts only `@c.x` or `@c.x * <digits>`, so it precedes the inversion and does not implement it.
- solver.h:26-33 `floor_division` and constraint.cc:47-58 `MAKE_EQUALITY` show the solver carries quotients and linear equalities.
- malloc_instructions_h_generator.py:510 sends constraints naming an address to the solver and leaves the rest to the selector, so the implementation routes the link to the solver.

Dissent:
- Sonnet's round-1 mark was conceded; Opus's dedicated fault row is ruled generic per decision 30.
- Fable's claim that decode always succeeds holds for one orientation only.
- Confidence: high on the rule, medium-high on allocator routing.

## Q10.4: what lands in v3

Decision:
- Option B: layers 1 and 2 and the derived rule all land in the v3 set; the agenda's option A is incoherent because derived is defined relative to a block.
- Order: layer 1 first (with the allocator domain fix); then the block, the derived rule, the assembler and disassembler generator, and the round-trip test as one unit.
- The unit never ships in part; if time runs short the v3 date moves; layer 2 is not deferred to a 3.x release.

Reasoning:
1. Q10.3 has no meaning without Q10.2, so they move together; decision 35 puts encodings in the plan and decision 38 makes the layers additive.
2. The standing direction is that nothing is cut for capacity and dates move; the assist-factor argument (panel-8/decision.md:198) is the ground it removes, and Opus withdrew it.
3. A 3.x release for the unit lets the v3 tag close without planned scope; deferral for capacity is a cut, so it is unavailable.
4. The objection "no consumer today" is met by the unit rule: the block lands only with its consumers and a test, and enters the v3 contract (panel 3 Q3.4) once.
5. Acceptance: `decode(encode(x)) == x` on a corpus ISA specification with real fields (a small word, a scattered immediate, an `A == B*8` link), since no case study carries encodings.

Evidence (verified):
- panel-8/decision.md:17-19 standing direction; :13 Q8.9 "dates move, scope does not".
- plans/roadmap/*.md hold no encoding item, so the unit is new work for the roadmap owner to place; Fable's cites q1-core.md:13 and q2-core.md:53 hold items L5 and T25 and do not support the claim.

Dissent:
- Fable and Opus (round 2) wanted layer 2 built as the first 3.x release; Sonnet leaned that way if no user is named; overruled: a minor release after the tag reduces v3 scope, which the owner direction excludes.
- Confidence: medium; the order is firm, the fit within two quarters is a date question.

## Q10.5: generator-specific data and adding a generator

Decision:
- Option A plus C: a namespaced plain-data mapping on the accelerator, each buffer, and each instruction, validated only by the named generator; run options stay entry-point keywords.
- Layer test: a layer changes what every conforming generator must do; generator-specific data never changes what S5 to S8 mean.
- Widths, encodings, and derived attributes are layers; an Allo partition hint is generator-specific data.

Reasoning:
1. Data for one buffer or instruction cannot travel as a keyword without restating its name, which is a side file in disguise.
2. Option C already serves run options: `generate_backend(accelerator, base_dir, cost)` and `generate_oracle(accelerator, base_dir)`.
3. Plain values let the typed model expose and the corpus diff the data (TableGen `TSFlags`, MLIR dialect-prefixed attributes); `validate()` publishes it on each typed node and lists the namespaces present, as a note.
4. `CONTRACT_VERSION` covers the container; each generator owns its schema, rejects an unknown key in its own namespace naming node and key, and silently ignores other namespaces (decision 39).
5. A misspelled namespace is ignored by design; the front-end listing plus the namespaces each generator reports reading expose it, with no registry in taidl.
6. A new generator takes the typed model after `validate()`, exposes `generate_<name>(accelerator, base_dir=None, **options)`, ships an expectations table (supported, or unsupported with a coded error, one row per corpus case) with generated coverage, and touches neither taidl nor another generator.
7. Neutrality test, the executable form of decision 39: for every corpus case carrying generator-specific data, oracle and backend output is identical with the data stripped.
8. A datum two generators read is promoted to a layer with front-end validation; file includes and large tables wait for first need.

Evidence (verified):
- act-backend/act_backend/generator.py:79-86 and act-oracle/taidl_to/generator.py:21 (entry points).
- plans/tests/generators.md:4 and :11-12 (T11 expectations table with generated coverage, T12 runner).
- plans/shared/README.md:13-14 (U2 typed contract, U3 `validate()`); panel-3/decision.md Q3.4 and Q3.5.

Dissent:
- Fable's opaque object was conceded to plain data; Sonnet's registry check in `validate()` is ruled out to keep taidl independent of generators.
- Confidence: high on the layer test and shape, medium on the member spelling (panel 3).

## Escalation to the owner

- None; every ruling fits decisions 10, 16, 30, 34, 35, 38, 39 and the standing direction.
