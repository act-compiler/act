# Panel 11, round 2, Fable

Lens: DSL and compiler-infrastructure architect.
Code paths are relative to /home/user; plan paths are relative to /home/user/act/plans.

### Q11.1

Strongest opposing argument: none against option A; the sharpest addition is Opus step 11, that `/` on a symbolic integer must raise and name `//`, because S6 `/` is floor division (spec-draft/expressions.md:28) while Python `/` is true division.

- Concession: I missed it, and it is decisive for "unambiguous meaning"; a silent `/` would give an architect a float where the spec gives a floor.
- The same rule covers `**`, `not`, `min`, and `max`, which S6.2 excludes (expressions.md:32); each raises at the author's line naming the spec rule.
- Concession to Sonnet step 13: a printer from the node model to HLO-like text is cheap and serves logs and the Q11.4 equivalence test; it is a tool, not a surface.
- Rebuttal to my own step 11: TAIDL bodies have no control flow, but an architect will still write `if n > 64:` on a traced attribute, so the `__bool__` hazard is the first error message the surface must get right (Opus step 11, Sonnet step 11).

Final position: option A, tracing, standard library only, with the hazard set of Opus step 11 plus the S6.2 exclusions as coded rejections.
Confidence: high.
What still divides the panel: nothing on the option; only whether the HLO printer ships or stays a test helper.

### Q11.2

Strongest opposing argument: Opus step 4, that a slice `d0[a:b]` reads as start and stop to every Python reader while a TAIDL region is start and size.

- Concession, with a stronger reason than readability: a region stores two expressions, start and size (taidl/taidl/instruction.py:115-129), and every expansion of S5.5 uses `R.start + R.size` (spec-draft/instructions.md, Expansions).
- A slice `addr_in : addr_in + n * 128` gives the size only as `(addr_in + n * 128) - addr_in`, and recovering `n * 128` needs the symbolic normalizer panel 3 Q3.6 ruled out (meeting/panel-3/decision.md Q3.6 row).
- Unsimplified, that size names an addressing attribute, which the backend's size checks must refuse (act-backend/act_backend/spec_checks.py:331-347 treats address-bearing terms specially); every slice region would then be backend-invalid.
- So the region is a method on the buffer with start and size as separate arguments, Sonnet step 3 and Opus step 3 agree on that shape, and the spelling is the moderator's.
- Concession on markers to Sonnet step 6: my `inplace=['bank_in']` named nothing, since the input was a slice, not a parameter; a marker is a keyword on `write` naming the input value, which is the pair `(In k, Out)` of S5.5 and keeps the one-output rule.
- Concession on state to Opus step 9 and Sonnet step 4: `update(ctr, ctr + 1)` is one recorded pair of S5.4's parallel assignment (instructions.md:46), while my `trn.rows = ...` makes a later read of `trn.rows` ambiguous.
- Concession on disjunction to Opus step 7: `acc == 0 | acc == 1` parses as `acc == (0 | acc) == 1`, since `|` binds tighter than comparison (docs.python.org, expressions, operator precedence), and the chained comparison then raises a baffling `__bool__` error; `any_of(...)` has no precedence trap, and `require(c1, c2)` is the conjunction.
- Concession on the encoding layer to Sonnet step 7: a separate `encode(instr, ...)` call keeps the layer-0 body unchanged when a layer is added (spec-draft/layers.md:4, :17), and S11.2 itself spells types as a name-keyed mapping (layers.md:22), so string field names are conformant.
- Rebuttal to Sonnet step 11, that accumulate needs a second instruction or a state flag because expressions have no conditional: S6.2:32 governs expressions, not the body; the body has `compare`, `select`, and `broadcast` (taidl/taidl/operations.py:97-98, :108), and the case study already spells the flag as `select` on `constant(@c.acc)` (casestudies/trainium2.md:109).
- Concession on the partial-unit write: my `[:, 0, 0:n]` sub-view is not a region; the bank is read whole and rewritten with `dynamic_update_slice` (trainium2.md:109), as today.

Corrected sketch of the two instructions:

```python
from taidl import Accelerator, Comp, Addr, require, any_of, update, ops
from taidl.types import u8, bf16, f32

trn = Accelerator('mini_trn')
d0 = trn.d0
d1 = trn.buffer('d1', access=[128], unit=[64], dtype=bf16)
sbuf = trn.buffer('sbuf', access=[128], unit=[], dtype=u8)
psum = trn.buffer('psum', access=[128, 8], unit=[512], dtype=f32)

@trn.instruction
def load_rm(n: Comp, addr_in: Addr, addr_out: Addr):
    raw = d0.read([addr_in], [n * 128])                       # u8[n*128]
    d1.write([addr_out], [n], ops.bitcast_convert(ops.reshape(raw, [n, 64, 2]), bf16))

@trn.instruction(meta={'allo': {'engine': 'tensor'}})
def nc_matmul(k: Comp, m: Comp, n: Comp, acc: Comp[u8], ws: Addr, mv: Addr, bank: Addr):
    require(k <= 128, m <= 128, n <= 512, any_of(acc == 0, acc == 1))
    w = ops.bitcast_convert(ops.reshape(sbuf.read([0, ws], [k, m * 2]), [k, m, 2]), bf16)
    x = ops.bitcast_convert(ops.reshape(sbuf.read([0, mv], [k, n * 2]), [k, n, 2]), bf16)
    whole = psum.read([0, bank], [m, 1])                      # f32[m,512]
    old = ops.slice(whole, [[0, m], [0, n]])
    p = ops.dot(w, x, lhs_contracting=[0], rhs_contracting=[0])
    isacc = ops.compare(ops.constant(acc), ops.constant(1), direction='EQ')
    base = ops.select(ops.broadcast(isacc, [m, n]), old, ops.constant(0, f32, [m, n]))
    psum.write([0, bank], [m, 1], ops.dynamic_update_slice(whole, base + p, 0, 0), inplace=whole)
```

Final position: regions as `buffer.read(starts, sizes)` and `buffer.write(starts, sizes, value, ...)`, attributes as annotated parameters, `require` and `any_of` for constraints, markers as keywords on `write`, `update` for state, `encode` as a separate call, and `meta=` on the three declaration levels.
Confidence: medium-high on the shape; medium on spelling.
What still divides the panel: whether result types are inferred with an optional `expect` (Opus step 6) or shown only in comments; whether `encode` sits in the body (Opus) or beside it (Sonnet, now Fable); whether `Addr[u16]` or `Addr('u16')` spells the width.

### Q11.3

Strongest opposing argument: Opus step 7, that a valid instruction the backend cannot model is dropped from the backend's instruction set with a recorded reason, and only an empty usable set rejects.

- Partial rebuttal: decision 40's "drops that candidate with a recorded reason" is about a kernel-compile candidate at run time, and decision 42 names the machine-state requirement a check; today's `check_state` raises (spec_checks.py:370-386).
- Partial concession: an ISA with one state-only semaphore instruction (trainium2.md, last table row) would otherwise force two specifications, which no owner decision asks for; the moderator weighs that against decision 42's word "check".
- Concession to Opus step 3 and Sonnet step 4: a list of check functions passed to the generator is a knob before a second use; the generator's entry calls `check` then its own functions, in code.
- Concession to Opus steps 1 and 2: `check(acc)` returns a frozen `Model`, the only type a generator entry accepts, so "runs once before any generator" holds by type; a raw `Accelerator` raises one coded rejection naming `check`.
- Concession to Opus step 8 over my step 7: the call site is recorded per node by the first frame outside taidl, which panel 3 Q3.5 wants as a typed member; `co_firstlineno` names only the function.
- Agreement with Opus step 4 and Sonnet step 8: `check` collects every rejection and raises once, each with code, section id, and the author's file and line.
- Agreement with Sonnet step 5: `python -m taidl check spec.py` is decision 42's standalone use.

Final position: option A as a type barrier; shared checks in `check`, backend checks as plain functions called from `generate_backend` after it, none for the oracle, unknown `meta` keys rejected by the generator that owns the namespace.
Confidence: high on the barrier and the move; medium on reject versus drop for a backend-unsupported instruction.
What still divides the panel: reject versus drop with a recorded reason for an instruction the backend cannot model.

### Q11.4

Strongest opposing argument: Opus step 4 and Sonnet step 4, that the converter ships in a final 2.x release rather than living as a development script, so v3 carries no ANTLR and private specifications migrate once.

- Concession: decision 10 batches breaking changes, and a 2.x minor release adding `taidl migrate` breaks nothing; decision 33 allows it as a public release pinning released versions.
- A script in the v3 repository would import v2 taidl to read its parse trees, which keeps ANTLR in the tree in all but name; Opus step 4 avoids that.
- Rebuttal to both on scope: the converter reads live `Instruction` objects (`instr_input_asts`, `constraint_asts`, `instr_semantics_ast`, instruction.py:44-55), so a loop that builds a family prints as its members, and a test that assembles semantics text at run time cannot be converted at all; those tests are rewritten by hand with the printer's output.
- Correction to Sonnet step 2: 25 Python files call `add_semantics` across the three repositories, not 40 (grep this session, including taidl/taidl/instruction.py itself and taidl/tests/test_grammar.py).
- Concession to Opus step 1 on order: `check` on the string front end before the tracer makes the surface swap front-end-only, and the tree-to-model translation it needs is reused by the converter and the equivalence test, so nothing is thrown away.
- Rebuttal to Sonnet step 5: equivalence is `Model` equality up to call sites (Opus step 5), which is the contract itself; printing to HLO text and re-parsing proves less and needs the printer first.
- Panel 3 Q3.12 reasoning 4 rejects a converter on capacity; it covers renamed API members, and this converter is one visitor over trees the parser already builds, so the capacity claim is the moderator's to weigh.

Final position: option A; the converter is `taidl migrate` in the last 2.x release, reading loaded specifications and printing the Q11.2 surface; order is node model, `check` from the string front end, generators on `Model`, tracer, converter and equivalence test, then ANTLR deletion in v3.
Confidence: medium-high.
What still divides the panel: whether the 2.x release is worth its cost against a script, which rests on evidence of private v2 specifications that none of the three has.
