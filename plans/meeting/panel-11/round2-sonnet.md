# Panel 11, round 2: Sonnet

Lens: a hardware architect writing an ISA specification for their own accelerator.
Code paths are relative to /home/user; plan paths are relative to /home/user/act/plans.
Corrections to my round 1 are stated where they bind.

### Q11.1 Authoring surface

Strongest opposing argument, restated fairly:
- No debater picked B, so the strongest opposition is its case against A: a textual language is hermetic.
- Reading a text spec runs no code, its meaning does not depend on the host language, and a Rust or C++ consumer (the fsim-lib direction of D23) needs no Python.
- A traced Python file is a program: it can loop, import, read files, and differ between runs.

Rebuttal:
1. Decision 41 names "a Python meta-DSL" first among its examples and asks for "as few dependencies as possible" (minutes/owner-decisions.md:35); B keeps a parser and a runtime pin.
2. Hermeticity is recoverable: the traced result is a typed node graph (panel 3 decision Q3.4), and a serialized form of it is the answer for a non-Python consumer; panel 3 already set that second-consumer trigger.
3. The error line is the architect's decisive need: today `text_error` counts lines inside the string (taidl/taidl/instruction.py:379-383); a traceback frame names the architect's own line with no code.
4. Looping is a gain for the architect, not a hazard: mini_trn_double copies two instructions verbatim (taidl/examples/mini_trn_double.py:17-40), and a Python loop writes the family once.
5. Opus and Fable both close the tracing hazards with an error: `__bool__` on a traced value raises, so `a < b < c` and `if attr:` fail at the author's line.
6. That also answers Fable's own mind-changer about chained comparison: Python calls `bool()` on the first comparison, which raises, so it is an error and never a silent meaning.
7. Opus's rule that `/` on a traced integer raises and names `//` is right: S6 `/` is floor division (language/spec-draft/expressions.md:28).
8. The spec text then needs one table row stating that surface `//` is S6 `/`, so an architect reading the spec and the file meets one name per operation.

Concession: Opus is right that tracing beats source reading, and I drop my "medium" on it (my round-1 point 9 and its REPL caveat).
Final position: A, tracing, standard library only, one table mapping Python operators to S6.
Confidence: high.
Still divides the panel: nothing on Q11.1; the three agree.

### Q11.2 Regions, attributes, state, constraints, markers, layers

Strongest opposing argument, restated fairly (Fable):
- Slices on buffer handles, `d1[a : a + n]`, read as the architect writes NKI access patterns; arity is checked by Python; the implicit bounds constraint falls out of the slice.
- The accumulate flag becomes one `where`, and a partial-unit write is emitted by the tracer instead of typed.

Rebuttal on slices:
1. A TAIDL region is a start and a size per access dimension; Python slices carry start and stop, so every size becomes `stop - start`.
2. The declaration-time rules compare size text: "an input whose start and size texts all match the output's ... without `set_inplace`" is rejected (language/spec-draft/instructions.md:93-100).
3. With a stop, `(1024 - a + 64) - (1024 - a)` must be simplified to `64` before that comparison; without an algebra pass, the same spec is accepted or rejected depending on how it is spelled.
4. With `start, size` given as written, the comparison stays structural, and the rule needs only "equal expression trees" in place of "equal token text".
5. Fable's `psum[0:m, bank : bank + 1][:, 0, 0:n] = base + p` writes a sub-view of a region.
6. The tracer must then synthesize a hidden read-modify-write of the rest of the unit, which is an input the architect never wrote, and which changes the in-place and overlap relations (instructions.md:60-62, :78).
7. The case study says region sizes cannot cover a partial unit shape and the author writes `dynamic_update_slice` (casestudies/trainium2.md:109); hiding that is the opposite of "unambiguous meaning".
8. Fable's `inplace=['bank_in']` names an input by string, yet no input is named `bank_in` in his body; reads carry no name, so a string key returns to the text-keyed surface that decision 41 removes.
9. My form names the value: `write(region, value, inplace=old)`, where `old` is the read; a typo is a `NameError` at the author's line.

Concessions to the other two, binding on my sketch:
- Opus: a state update is the explicit `update(calls, calls + 1)`, not an implicit rule; it is greppable and matches the one-parallel-assignment rule of S5.4 (instructions.md, S5.4).
- Opus: a disjunction is `any_of(...)`, with `require(*conds)` as the conjunction; my `|` has a precedence trap, since `a == 1 | b == 2` parses as a chain and is caught only by luck of `__bool__`.
- Opus: `encode` is a call in the body whose fields name attribute objects; my round-1 sketch used strings `'addr_1'`, which a typo turns into a late error.
- Opus: result types are inferred by the S4 rules, with an explicit target only for reshape, broadcast, convert, bitcast_convert, iota, and constant (openxla operation semantics, Reshape and BroadcastInDim take dimensions).
- Fable: `where(cond, a, b)` is a legal value-level operation: it is `select` over `broadcast(compare(constant(acc), 1))` (casestudies/trainium2.md:49-53); my round-1 point 11 was true of attribute expressions (spec-draft/expressions.md:32), not of values, so the accumulate flag needs no second instruction.
- Fable: a second write to one output can be made to raise at trace time, so his would-change-mind is met.

Where I hold against Opus:
- A marker belongs on the write, naming the input, not on the read: the spec says a mark addresses output region 0 of a one-output instruction (instructions.md:84-85), and a mark on the write still names that pair when the multi-output rule is lifted.
- Dtypes are objects (`bf16`), so `Comp[u8]` and `Addr[u16]` subscript cleanly and an editor completes them.

Final sketch, accumulate with the corrections:

```python
@mini_trn.instruction(meta={'allo': {'engine': 'tensor'}})
def gemm_acc(acc_flag: Comp[u1], a1: Addr[u8], a2: Addr[u8], out: Addr[u8]):
    require(a1 != a2)
    a = read(d1.region(a1, 64))
    b = read(d1.region(a2, 64))
    old = read(acc.region(1024 - out, 64))
    prod = convert(dot(a, b, lhs_contracting=[1], rhs_contracting=[0]), f32)
    write(acc.region(1024 - out, 64), where(acc_flag == 1, old, 0) + prod, inplace=old)
    encode(32, fixed=[((31, 24), 0x12)], fields=[(a1, (23, 16)), (a2, (15, 8)), (out, (7, 0))])
```

Final position: tracing, typed parameters, `region(start, size)`, `write(..., inplace=value)`, `require` with `any_of`, explicit `update`, in-body `encode` with attribute objects, `meta=` on the decorator.
Confidence: medium-high on structure; medium on spelling.
Still divides the panel: slice syntax versus start-and-size; marker on read versus on write; strings versus objects for dtypes.

### Q11.3 The check function

Strongest opposing argument, restated fairly (Opus):
- A backend limit, such as an instruction with state or two outputs, is not an architect error under decision 40.
- The backend drops that instruction with a recorded reason, and only an empty usable set rejects the spec.
- Rejecting a valid spec because the backend cannot model one instruction makes the oracle and the backend disagree on what is valid.

Evidence on the limits (act-backend/act_backend/spec_checks.py):
- `check_outputs` refuses any instruction with other than one output (:22-29).
- `check_state` refuses any instruction that updates or reads state (:370-386).
- `beyond_allocator` and `honoured` refuse sizes and constraints the C++ allocator cannot take (:277-281, :331-347).
- Each of these is a model limit of the backend, and none is a defect in the architect's text.

Rebuttal and partial concession:
1. Decision 42 names "the backend's machine-state requirements" as an example of the generator's own check (owner-decisions.md:36), and today that is `check_state`; the owner expects some backend check to be able to say no.
2. Decision 40 says the backend drops "that candidate" with a reason; its words are about a kernel-search candidate, and it bars stopping on one failed candidate, not on a spec the backend cannot represent.
3. From the architect's side, silent dropping is the worst outcome: the generated compiler quietly never uses their tensor instruction, and the first sign is slow code.
4. Opus's position holds only with a visible report: each dropped instruction, its code, its spec section id, and its author line, printed at generation time and stored in the generated artifact.
5. I therefore split the backend's checks into two kinds.
6. Kind one, errors: the spec contradicts the backend's stated input (for example state declared but never initialized, a name the allo or backend metadata namespace does not know); the check raises, and the architect can fix it.
7. Kind two, model limits: the instruction is valid but outside the backend; it is dropped with the report of point 4, and a spec whose usable set is empty raises.
8. A strict option turns kind two into errors, for an architect who wants a loud failure.
9. This respects decision 42 (the check exists and can reject) and decision 40 (no stop for a correctly written spec).

Concessions on form:
- Opus: `check` returns a frozen `Model` and generators accept only that type; "always runs once" becomes a type fact.
- Opus and Fable: no registry and no subclassing; the generator entry calls its own plain functions on the model.
- Opus: `check` collects every error and raises one `CheckError`; each entry is `file:line: instruction: section-id: message`.
- Fable: the line comes from the call site each node records, with `__code__.co_firstlineno` as the fallback for a whole-instruction problem.

Final position: A with the type barrier, the common check covers everything in the front end, backend checks are errors or reported drops, the oracle adds none.
Confidence: high on the form; medium on error versus drop for model limits.
Still divides the panel: whether a model limit is ever an error.
What would change my mind: the owner reading decision 42's "machine-state requirements" as a hard reject of the whole spec, which makes kind two moot.

### Q11.4 Migration and order of work

Strongest opposing argument, restated fairly (Fable and panel 3 Q3.12):
- The converter is a development script, not a shipped feature.
- Panel 3 ruled "no converter", and gave capacity as the reason: a converter rewrites Python programs and decision 11 leaves no capacity (meeting/panel-3/decision.md:16, :182).
- The corpus is small and a stray `add_semantics` call gets one coded rejection.

Facts:
- I overstated the corpus: the real count is 25 Python files with 176 `add_semantics` calls across taidl, act-oracle, and act-backend (6, 9, and 10 files), and 7 more files with 45 calls under /home/user/act; my "40 files" was wrong.
- Most of the call sites are tests, not examples, and the backend tests use state and constraints that the examples do not (casestudies/gaps.md:45 is about examples only).

Rebuttal and concession:
1. The converter must cover the full old grammar, since tests exercise state, constraints, and updates; a visitor over the existing tree is small (Opus counts about 165 grammar lines, IDLV2.g4:6-25).
2. Panel 3's "no converter" covers renamed members with a one-line replacement; a whole-surface change has no such replacement.
3. Still, Fable is right that public shipping is a support commitment; a `taidl migrate` command in 2.x promises behavior to outside users, and no plan names such users.
4. Final: the converter is written and used on the 32 corpus files either way; shipping it as a 2.x command is a time-boxed add-on of one small item, dropped first if capacity is short.
5. The v3 coded rejection of `add_semantics` points at the migration route in either case.
6. I adopt Opus's order, which beats my round-1 order: build the typed `Model` and `check` over today's string front end, port every generator to read only the `Model`, then add the traced surface, then the converter.
7. This makes the surface swap front-end only, and the equivalence test is `Model` equal to `Model` up to call sites; my HLO-print-and-reparse step is dropped, with the printer kept only for logs and the architect's "show me what you understood" view.
8. Order: operation table as data; typed `Model` and `check` over the string front end; generators on `Model`; tracer; converter; corpus converted and diffed; ANTLR deleted last.

Final position: A, converter plus per-spec equivalence, string form removed in v3, Opus's ordering.
Confidence: medium-high.
Still divides the panel: ship the converter in 2.x or keep it a repository script.
What would change my mind: a partner list showing outside ISA specifications, which argues for shipping; none exists in the plans.
