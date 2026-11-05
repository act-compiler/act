# Authoring surface corpus

T38 turns the rule set of panel 11 (Q11.1 to Q11.3) into cases: what the tracer records, what it rejects at the author's line, what `check` collects, and what a backend excludes.
Every case is a `spec.py` in the authoring surface (T5); rows are `case.toml` cases (T6), and the exclusion rows also run in the backend suite (T12).

## T38. Surface, `check`, and exclusion cases (S4 to S8, S1.2)

- What: trace rows, phase `trace`, one per construct the tracer cannot record: `if` on a symbolic value, `and`, `or`, `not`, `&`, `|`, a chained comparison, `bool()`, `min`, `max`, `/`, `**`, and a slice `buffer[a:b]`; each expects the exception at the author's line and the spelling it names.
- What: tensor rows, phase `trace`: `/`, `//`, and `%` on a tensor value each expect a message naming `ops.divide` or `ops.remainder`; `+`, `-`, `*`, and unary `-` on tensors are accepted.
- What: expression rows: `+`, `-`, `*`, `//`, `%`, unary `-`, parentheses, and the six comparisons are accepted; `//` and `%` on negative operands store the S6 values (floor, divisor sign), so a StableHLO `divide` is never mistaken for them.
- What: family rows: a Python loop over plain integers at definition time yields one instruction per member, `@acc.instruction(name=...)` names a member, and the `Model` lists them in definition order.
- What: type rows: `Comp[u7]` and `Addr[u16]` set kind and width, an unannotated parameter is `s64`, and parameter order is attribute order; no row depends on a parameter's name.
- What: marker rows: `write(r, v, inplace=x)`, `apart=[y]`, `no_partial_overlap=[z]`, and `may_overlap=[w]` give the T30 relations; a marker naming a value of another buffer is rejected, and a misspelled marker value is a `NameError` at the author's line.
- What: state rows: `update(ctr, ctr + 1)` stores the pre-issue read; two updates of one variable are rejected at `check`.
- What: `require` rows: `require(c1, c2)` is a conjunction, `any_of` and `all_of` nest, and a comparison built twice from equal text is two constraints by identity.
- What: `check` rows: an ISA specification with five typos yields one `Rejection` with five entries, each with `code`, clause, instruction, and the author's file and line; a whole-instruction problem carries the function's first line.
- What: encode rows: `acc.encode(instr, ...)` with a field key naming no parameter of `instr` is rejected at `check`; the layer rows of T33 and T34 use this spelling.
- What: metadata rows: `meta={'allo': {...}}` at `Accelerator`, `acc.buffer`, and `@acc.instruction` reaches the `Model` as plain data (T35).
- What: printer rows: the HLO text of each `Model` reparses with the taidl 2.x parser to an equal `Model` for the 2.x-expressible subset, which tests the printer as much as the surface.
- What: type-barrier rows: a generator entry given an `Accelerator` calls `check` once (a counting wrapper), and a generator body that reads an `Accelerator` attribute fails a static test, since only `Model` reaches it.
- What: exclusion rows, one per backend limit: an instruction that reads state, updates state, has two outputs, sizes over an addressing attribute or state, holds a constraint the allocator cannot hold, or uses a tensor operator outside the supported table is excluded with its `code`, clause, and author line; the rest of the ISA specification generates.
- What: an ISA specification whose every instruction is excluded is rejected naming the empty set; a kernel that needs an excluded instruction is rejected at kernel compile time naming it.
- What: the oracle rows: the same ISA specifications exclude nothing in the oracle, and the oracle registers no check beyond `check` (decision 42).
- What: an unknown key in a backend's own `meta` namespace is rejected after `check` naming node and key; the same key is accepted by `check` and by every other generator.
- Benefits: each hazard the tracer closes by raising has a named row, so a silent meaning would fail a case; the exclusion rows make decision 40 executable for the backend.
- Caveats: the spelling of `any_of`, type objects, and keyword names is provisional (Q11.2 confidence), so slugs name the construct and a rename edits the cases and nothing else.
- Fail points: a trace row that passes because Python raised for another reason; the row asserts the exception text names the spelling and the author's line.
- Order and dependencies: after T5, T6, T12, U3, and U10; the surface spelling comes from the language track's panel 11 items and the printer rows wait for the printer of Q3.4; Q1.
- Verification: every construct of the Q11.2 rule set has a row; the trace rows number at least twelve, the exclusion rows one per limit.
- Breaking: no.
- Files: new cases under `taidl/conformance/`; exclusion rows in `act-backend/tests/test_conformance.py`.
