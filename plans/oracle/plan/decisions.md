# act-oracle architecture debate and decisions

Three positions were argued on the oracle's structure: an incremental simplifier, an overhaul architect, and an outside critic who judges both as a reference interpreter for an ISA.
This session had no sub-agent launcher, so the track lead wrote the three positions from the code read this session and then adjudicated; a later session can rerun them as separate agents.

## Positions

### Incremental simplifier

- Delete dead code first: the unreachable branch at `taidl_to/oracle_visitor.py:237-241`.
- Merge the six word tables in `oracle_visitor.py` (`TYPES`, `PRETTY`, `REDUCERS`, `INDICES`, `CHLO`, `op_to_template`) into one.
- Split `generic/utils.py` (989 lines) by role; do not rewrite.
- Keep `shapes.py`: its rules give errors that name the instruction and line, which the MLIR verifier would not.
- Risk is lowest because every step can be checked by a byte-identical generated `api.py`.

### Overhaul architect

- Lower TAIDL to jax directly: each word a `jax.lax` call, the visitor building a Python closure, no StableHLO text, no `templates.txt`, no `ssa_types`.
- Delete most of `utils.py`'s emitters (`sh_*`) and `template.py`; jax infers types, so most of `shapes.py` goes too.
- Loops become `jax.lax.fori_loop` over the closure, replacing the symbolic loop algebra in `generic/utils.py:18-260`.

### Outside critic

- A reference interpreter must compute exactly the spec's meaning; its own convenience may never change a result.
- Against the architect: `jax.lax` adds its own type promotion, decompositions and precision defaults (the oracle already fights them at `generic/decorator.py:22-23` and `generic/utils.py:604`); emitting StableHLO text names the reference operation of S8.1 directly, which is the strongest fidelity argument.
- Against the architect: `fori_loop` cannot express a value that changes per iteration in a shape, which the oracle supports by unrolling.
- Against the simplifier: merging tables does not address that the oracle holds a second copy of language rules (`TYPES` equals S8.3; `shapes.py` checks S7.4); those belong in the front end so every generator rejects the same specs.
- Against both: there is no independent check of meaning; the literal double rounding (`oracle_visitor.py:383`, `generic/utils.py:408-409`) survived because tests compare the oracle with jax, not with the spec.

## Round 2

Each position answered the others on the four contested points; the ruling follows each.

### Lowering target: StableHLO text or `jax.lax`

- Architect: text emission needs the `ssa_types` bookkeeping in every `sh_*` emitter (`generic/utils.py:459-850`), and a type error surfaces as a jax MLIR parse error that names no instruction; `jax.lax` does not promote types implicitly and takes precision per dot, so the critic's promotion objection is weak.
- Critic: `jax.lax` has no `mulhi` and no `batch_norm_inference` primitive, so two words would lower to compositions with their own rounding, not to their S8.1 reference operations (a recollection of the jax API, not checked this session); text names each reference operation one to one (`oracle_visitor.py:124`, `oracle_visitor.py:330-333`).
- Simplifier: either path keeps the per-iteration unrolling (`templates/templates.txt:101-133`), so the migration buys less than it costs.
- Ruling: keep StableHLO text.
- Ruling: a `jax.lax` path is reconsidered only if it emits the same reference operation for every word, measured by diffing module text.

### Shape rules: oracle, MLIR verifier, or front end

- Architect: the StableHLO verifier already checks shapes; `shapes.py` (543 lines) checks them again.
- Critic: the verifier checks StableHLO's rules, not TAIDL's; the presumed reshape (`generic/utils.py:741`) and the `compare` type fit (`oracle_visitor.py:276-283`) are language rules, and a verifier error names no instruction or line.
- Simplifier: moving rules across repos needs a TAIDL release; keep them in the oracle until then.
- Ruling: O7 moves literal-shape rules to the front end in Q2; the oracle keeps call-time evaluation; the verifier is never the only check.

### Call-time checks over every iteration

- Architect: `iterations` enumerates the product of all trip counts (`generic/utils.py:82-86`); affine starts need only their bounds.
- Simplifier: enumeration is short and exact; an affine fast path adds a second code path to keep in step.
- Critic: exactness comes first, but an oracle that takes minutes to compile a long loop will not be used.
- Ruling: keep enumeration; O11 adds the affine fast path only once a kernel's compile time is measured past an agreed threshold.

### Templates and generated code

- Architect: delete `templates/templates.txt` and generate calls only.
- Critic: when an oracle disagrees with hardware, the per-instruction function in `api.py` is what a reader debugs, so it must stay one function per instruction.
- Simplifier: the skeleton templates (`API_FILE`, `API_FUNC`) are fine; the per-word ones duplicate the `sh_*` emitter signatures.
- Ruling: O3 replaces the per-word templates by emitter calls and keeps the skeletons; O18 drops the line splitter, since readability comes from one function per instruction, not from line wrapping.

## Replacement justification

- No module is replaced from scratch.
- The per-word templates in `templates/templates.txt` are replaced (O3) because they are text fragments that ruff and tests do not check, each restating an `sh_*` emitter's arguments, and the visitor already holds every value they substitute.
- The `layout` splitter in `template.py:61-156` is removed (O18), not replaced, because it re-implements a formatter for generated code.

## Verdict

- Keep StableHLO text as the lowering target (critic over architect): it names each word's reference operation one to one.
- Take the simplifier's structural steps first (O1, O5, O4), each verified by a byte-identical generated `api.py`.
- Then remove one layer: string templates become emitter calls (O3) over one line record (O2).
- Move language rules out of the oracle (critic): S8.3 types (O6) in Q1, S7.4 shape rules (O7) in Q2, both with the L and B tracks.
- Add an independent NumPy reference interpreter for differential tests (O13), the critic's main point.
- Rationale: the generated oracle is the product, so every structural step must leave its text unchanged, and every behavioral step must change it for a stated spec reason.
