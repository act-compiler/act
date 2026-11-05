# Panel 10 agenda: attribute bits and binary encodings

Binding owner decisions: 16 (no undefined behavior), 30 (markers and implicit constraints are one constraint mechanism), 34 (attributes may be negative; names carry no meaning), and 35.
Owner decision 35: the plan covers bit widths per attribute and binary-encoding annotations for an assembler; an ISA specification may hold attributes that never appear in the binary, tied to encoded attributes by constraints (for example addresses written as `B*8` with a constraint linking `B` to the encoded attribute `A`).

### Q10.1: How does an ISA specification declare the bits of each attribute?

- Context: today attributes are unbounded integers checked against signed 64 bits; real encodings have fixed widths and signedness, and a value outside the field cannot be encoded.
- Options: A. width and signedness per attribute, enforced as implicit constraints; B. a value range per attribute (min, max), with width derived; C. widths only inside an encoding annotation, nothing on the attribute itself; D. other, argued from evidence.
- Evidence: taidl/taidl/instruction.py and accelerator.py (attribute declarations); spec-draft S5; act-backend allocator constraints (cpp/malloc); priorart/llvm-tablegen.md, priorart/sail.md; casestudies (AMX `ldtilecfg`, Hopper `wgmma` immediates, NKI instruction fields).

### Q10.2: How are binary encodings annotated, and which generators consume them?

- Context: an assembler needs field positions, opcodes, and fixed bits; a disassembler or a trace decoder (for example fsim's NKI driver) needs the reverse mapping.
- Options: A. an optional per-instruction encoding block (opcode plus attribute-to-bit-field map); B. a separate encoding file keyed by instruction name; C. no encoding in v3, a design study only; D. other.
- Evidence: LLVM TableGen `Inst{...}` fields, Sail `encdec` mappings, RISC-V opcode tables, the act-backend assembly emission path, /home/user/act/plans/oracle/fsim/ (trace decoding).

### Q10.3: How are attributes that never appear in the binary handled?

- Context: an author writes addresses in terms of `B` because `B*8` is easier, while the binary holds `A`, and a constraint such as `A == B*8` links them; the backend must choose binary values, and the oracle may receive binary values.
- Options: A. mark each attribute as encoded or derived, and require every derived attribute to be uniquely determined by the encoded ones through constraints; B. no marking: the encoding annotation lists encoded attributes, and the rest are derived; C. allow derived attributes with several solutions and define which one is used; D. other.
- Questions within: how a generator solves for derived values (closed form, solver), what happens when no value or several values satisfy the constraints (decision 16), and how negative values and decision 34 interact with field widths.

### Q10.5: How are generator-specific metadata layered on the ISA specification, and what makes adding a generator easy?

- Context: owner decision 39: new generators will come (for example act-allo, generating an Allo specification), and an ISA specification may carry optional metadata for one generator at instruction, buffer, or accelerator level, ignored by other generators and passed to that generator's entry point (for example `generate_allo()`).
- Options: A. a namespaced metadata block per generator inside the ISA specification, validated by that generator; B. a separate metadata file per generator keyed by instruction and buffer names, passed to its entry point; C. metadata as keyword arguments to the generator's entry point only; D. other.
- Questions within: how the front-end contract (panel 3: typed model, `CONTRACT_VERSION`, `validate()`) exposes metadata; how a generator declares and checks its metadata; how the conformance corpus and coverage tables extend to a new generator; how bit-width and encoding layers (Q10.1-Q10.3) relate to metadata.
- Evidence: panel-3/decision.md, plans/shared/ (U2, U3), plans/tests/ (expectation tables per generator), Allo DSL documentation (github.com/cornell-zhang/allo), act-backend and act-oracle entry points (`generate_backend`, the oracle generator).

### Q10.4: What lands in v3 and what is deferred?

- Options: A. Q10.1 and Q10.3 in v3, encodings (Q10.2) after v3; B. all three in v3; C. all deferred to a design study; D. other.
- Evidence: the roadmap (plans/roadmap/), panel 8 decision, and which generators would use each part first.
