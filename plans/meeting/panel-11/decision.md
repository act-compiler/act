# Panel 11 decision: authoring surface and the check function

| Question | Decision | Confidence |
| -------- | -------- | ---------- |
| Q11.1 | A: a Python embedded DSL by tracing one call of a decorated function on symbolic attributes; no source parsing, no third-party dependency, ANTLR removed | high |
| Q11.2 | Typed parameters; `buffer.region(starts, size=sizes)` objects with `read(r)` and `write(r, value, ...)`; markers as `write` keywords naming the read value; `require`, `any_of`, `all_of`; `update`; `Comp[u7]` and `Addr[u16]` type objects; `encode` beside the instruction; `meta=` at three levels | medium-high on shape, medium on spelling |
| Q11.3 | A with a type barrier: `check(acc)` returns a frozen `Model`, the only type a generator body reads; backend checks are plain functions after it; an architect-correctable fault rejects, a backend model limit excludes the instruction with a recorded, printed reason; the oracle adds none | high on form, medium-high on reject versus exclude |
| Q11.4 | A: a converter script on the taidl 2.x line, run once on the corpus, with a `Model` equivalence test per file; string form and ANTLR gone in v3; order is node model, `check` from the string front end, generators on `Model`, tracer, converter, deletion | medium-high |

Cited code was re-read in /home/user/taidl, /home/user/act-backend, and /home/user/act/plans; uncited claims carry no weight.
Code paths are relative to /home/user; plan paths are relative to /home/user/act/plans.

## Rule set

- An ISA specification is a Python file; an instruction is a function decorated by `@acc.instruction`, run once by the tracer on symbolic attributes to build the typed node model of Q3.4.
- The tracer never reads function source; Python control flow on plain values runs at definition time and builds instruction families, with `@acc.instruction(name=...)` naming a member.
- Attributes are parameters; the annotation `Comp` or `Addr` gives the kind, `Comp[u7]` or `Addr[u16]` adds the width layer, the default is `s64`, parameter order is attribute order, and no name carries meaning (decision 34).
- A region is `buffer.region(starts, size=sizes)`, one start and one size per access dimension, both lists, `size` keyword-only; a slice `buffer[a:b]` is rejected.
- `read(r)` yields the region's value with its type; `write(r, value, inplace=x, apart=[y], no_partial_overlap=[z], may_overlap=[w])` writes it, and each marker names a read value of the same buffer; the panel 9 expansions apply unchanged.
- Integer expressions use Python `+`, `-`, `*`, `//`, `%`, unary `-`, parentheses, and the six comparisons; `/`, `**`, `not`, `and`, `or`, `&`, `|`, chained comparison, `min`, `max`, `bool()`, and `if` on a symbolic value raise at the author's line and name the spelling to use; S6 spells its floor division `//`.
- Tensor values overload `+`, `-`, `*`, and unary `-` only; `/`, `//`, and `%` on a tensor raise naming `ops.divide` and `ops.remainder`; every other operation is `ops.<word>` as S8 names it.
- Result types are inferred by S4; a target shape or type is written only where the operation takes one (`reshape`, `broadcast`, `convert`, `bitcast_convert`, `iota`, `constant`); no `expect`; the HLO-text printer of the `Model` is the "show me what you understood" tool and a log format, never an input form.
- Constraints are `require(c1, c2, ...)` as a conjunction, with `any_of(...)` and `all_of(...)` for nesting; a comparison hashes by identity.
- State is `ctr = acc.state('ctr', 0)`; a read in a body is the pre-issue value; `update(ctr, ctr + 1)` records one member of the parallel assignment of S5.4; two updates of one variable reject.
- The encoding layer is `acc.encode(instr, word=32, fixed=[((31, 24), 0x12)], fields={'a1': [(23, 16)]})` beside the instruction; a field key names a parameter of `instr`, and an unknown key rejects.
- Generator metadata is `meta={'allo': {...}}` on `Accelerator(...)`, `acc.buffer(...)`, and `@acc.instruction(...)`.
- Every node records the first frame outside `taidl` as its call site (Q3.5); a whole-instruction problem uses the function's first line.
- `taidl.check(acc)` runs every front-end rule, freezes the accelerator, and returns a `Model`; it collects every problem and raises one `Rejection` whose entries each carry code, section id, instruction, and the author's file and line in the Q3.8 form; `python -m taidl check spec.py` is the standalone use of decision 42.
- A generator entry takes a `Model` or an `Accelerator`; given an `Accelerator` it calls `check` once; its body reads only the `Model`; no registry and no checker subclass.
- Backend checks after `check`: an unknown key in its `meta` namespace rejects; an instruction that reads or updates state, has other than one output, a size over an addressing attribute or state, a constraint the allocator cannot hold, or a tensor operator outside the supported table is excluded from the backend's instruction set with a reason (code, section id, author line) printed at generation and stored in the generated crate; an empty usable set rejects; a kernel that needs an excluded instruction is rejected at kernel compile time naming it.
- S1.2 gains the exclusion outcome beside rejection; both name the feature and neither gives it another meaning.
- The oracle registers no check (decision 42).
- The converter reads loaded v2 `Instruction` objects and prints the surface above; it is a script on the taidl 2.x line, run under taidl 2.x, not a `taidl` subcommand; the v3 removed-names rejection of `add_semantics` names it (Q3.12).
- Equivalence per corpus file: `Model` built from the v2 string form equals `Model` traced from the converted file, up to call sites; a test that assembles semantics text at run time is rewritten by hand from the printer's output.

## Q11.1 Authoring surface

Decision: option A by tracing; standard library only.

Reasoning:
1. Decision 41 names a Python meta-DSL first and asks for as few dependencies as possible; today's one runtime dependency is the parser pin (taidl/pyproject.toml:15), and the generated parser modules are most of the package.
2. Options C and D put a C++ toolchain in front of an architect who writes 100 lines of Python today; ODS expands to C++ at compiler build time (mlir.llvm.org, DefiningDialects/Operations), and decision 28 admits no such dependency.
3. Option B keeps a parser, and a parser is why today's errors count lines inside a string (`text_error`, taidl/taidl/instruction.py:379-383); a traced body fails inside the author's own frame.
4. Tracing beats source reading: a traced function has Python's meaning on every line, and source reading defines a language that looks like Python but is not, which NKI abandoned for tracing (round1-opus.md step 6).
5. The hazards of tracing are closed by raising, never by a silent meaning: `__bool__` on a symbolic value raises, which catches `if`, `and`, `or`, and chained comparisons in one place.
6. S6 `/` is floor division and `%` takes the divisor's sign (spec-draft/expressions.md:28-29), which is Python `//` and `%` exactly; Python `/` is true division, so it raises and names `//`.
7. StableHLO `divide` on integers discards the fractional part and `remainder` takes the dividend's sign (openxla.org/stablehlo/spec, fetched), the opposite of S6 on negative operands; a tensor `/` or `%` therefore raises and names the tensor operator, so one glyph never has two meanings in one file.
8. Panel 3 fixed the contract as a typed model plus a typed node model with trees unstable (meeting/panel-3/decision.md:62-74); the tracer builds that model directly and removes the tree-to-model step from the shipped front end.

Dissent: none on the option; Sonnet's round-1 "medium" on tracing over source reading was withdrawn in round 2.
Confidence: high.

## Q11.2 Regions, attributes, state, constraints, markers, layers

Decision: the rule set above; the accumulate sketch below is normative for shape, not for every spelling.

```python
from taidl import Accelerator, Comp, Addr, read, write, require, update, ops
from taidl.types import u1, u8, bf16, f32

acc = Accelerator('mini_trn')
d1 = acc.buffer('d1', access_dims=[128], base_element_dims=[64], element_type=bf16)
accum = acc.buffer('accum', access_dims=[64], base_element_dims=[64], element_type=f32)
calls = acc.state('calls', 0)

@acc.instruction(meta={'allo': {'engine': 'tensor'}})
def gemm_acc(flag: Comp[u1], a1: Addr[u8], a2: Addr[u8], out: Addr[u8]):
    require(a1 != a2)
    a = read(d1.region([a1], size=[64]))
    b = read(d1.region([a2], size=[64]))
    target = accum.region([1024 - out], size=[64])
    old = read(target)
    prod = ops.convert(ops.dot(a, b, lhs_contracting=[1], rhs_contracting=[0]), f32)
    base = ops.select(ops.broadcast(ops.compare(ops.constant(flag), ops.constant(1), direction='EQ'), [64, 64]), old, ops.constant(0, f32, [64, 64]))
    write(target, base + prod, inplace=old)
    update(calls, calls + 1)

acc.encode(gemm_acc, word=32, fixed=[((31, 24), 0x12)],
           fields={'a1': [(23, 16)], 'a2': [(15, 8)], 'out': [(7, 0)]})
```

Reasoning:
1. A region stores a start and a size per access dimension (taidl/taidl/instruction.py:115-129), and every S5.5 expansion and declaration-time rule is written over those two texts (spec-draft/instructions.md:75-76, :98-100); a slice gives a stop, so the size would be `stop - start`, and recovering `64` from `(1024 - a + 64) - (1024 - a)` needs the symbolic normalizer Q3.6 ruled out.
2. A named region object is read and written once in the accumulate case, where every slice or call-per-read spelling writes the same start twice (round1-sonnet.md:76-78); the `size=` keyword stops a reader taking the second argument for a stop.
3. Lists for starts and sizes keep one form for `d0`, `d1`, and a two-dimensional PSUM, and the arity check of today (instruction.py:123-129) stays one rule.
4. A marker on `write` naming the read value pairs one input with one output by object, which is the S5.5 pair `(In k, Out)` panel 9 kept (meeting/panel-9/decision.md:60-71), and a typo is a `NameError` at the author's line; a string key in a decorator named a value no sketch had (round1-fable.md:55-57).
5. Fable's sub-view write `psum[...][:, 0, 0:n] = ...` hides a read-modify-write the architect never wrote and changes the overlap relations; the case study spells it `dynamic_update_slice` on the whole unit (casestudies/trainium2.md:109), and so does the surface.
6. `and` and `or` cannot be overloaded, and `|` binds tighter than a comparison, so `acc == 0 | acc == 1` parses as a chain and fails with a `__bool__` message that names the wrong thing; `any_of` and `all_of` have no precedence and read as S6.3's words; `&` and `|` on a condition raise naming them.
7. `update(ctr, ctr + 1)` is one recorded pair of the parallel assignment (instructions.md:46); an attribute assignment `acc.rows = ...` would make a later read in the same body return the new value by Python's meaning.
8. Type objects `u1` to `u63` and `s1` to `s64` complete in an editor and fail as `NameError` at the author's line, where a string fails late; a module defines them in a loop, so the count of 126 is no cost.
9. The encoding block is read by no oracle and no backend (spec-draft/layers.md:40) and a layer adds to the one below it (layers.md:4), so the block sits beside the body and never edits layer-0 text; its field keys name parameters, which `check` matches against the signature, and matching a declared name is not inferring meaning from one (decision 34).
10. Result types are inferred as JAX does on tracers; the duplicated `parameter(k)` type line and `check_values` reconciliation (instruction.py:142-161) disappear, and the printer answers "what did you understand".
11. The `where` shorthand is not added: `select` over `broadcast(compare(...))` is the body's own idiom (trainium2.md:54, :109) and a second spelling for one operation is what decision 41's "nothing fancy" excludes.

Dissent: Sonnet holds in-body `encode` with attribute objects, ruled out by reasoning 9; Fable's slices were conceded in round 2; Opus holds `&` and `|`, ruled out by reasoning 6.
Confidence: medium-high on shape; medium on `any_of` versus operators and on type objects versus strings.

## Q11.3 The check function and generator checks

Decision: option A with the type barrier; reject versus exclude as the rule set states.

Reasoning:
1. Only `check` constructs a `Model`, so decision 42's "always runs once before any generator" holds by type; a generator entry given an `Accelerator` calls it once, so the architect's one-call flow stays.
2. `check` subsumes `validate()` of Q3.2 and Q3.5 (meeting/panel-3/decision.md:37-40, :76-88) and today's front-end duplicates in the backend (act-backend/act_backend/spec_checks.py:394-418), which Q3.2 already deletes.
3. Collecting every problem before raising lets an architect fix five typos in one run; a trace-time error such as `__bool__` is a Python exception in the author's frame and stops at once, which is the right behavior for a construct the tracer cannot record.
4. Decision 40 limits rejection to problems the architect can correct, such as a typo or an invalid ISA specification; an instruction that updates state (`check_state`, spec_checks.py:370-386), has two outputs (:22-29), or sizes over an address (:277-290) is a valid ISA the architect cannot "correct" without changing the hardware.
5. Rejecting the whole specification on one such instruction forces two specifications for an ISA with one semaphore instruction (casestudies/trainium2.md:115) and makes the oracle and the backend disagree on what is valid.
6. Decision 42's "machine-state requirements" are still a check: it runs after `check`, it says no to an uninitialized or non-integer state the architect can fix, and it excludes what the backend cannot model with a reason the architect sees.
7. Silent dropping is the worst outcome for the architect, so the report is printed at generation and stored in the crate; S1.2 already has a `partial` status for "works with stated limits" (spec-draft/overview.md:28), and the backend's docs row for state carries it.
8. A strict option that turns exclusions into errors is a knob before a use site; the printed report and the expectations table of S12.4 give the loud signal.
9. A function list, a registry, or a checker base class is abstraction before a second use; an explicit call from the generator's entry meets S12.4's "touches neither taidl nor another generator" (spec-draft/generator_metadata.md:35).

Dissent: Fable reads decision 40's "drops that candidate" as kernel candidates only and would reject on an unsupported word; reasoning 4 and 5 outweigh it, and Fable's own semaphore case concedes the cost.
Confidence: high on the form; medium-high on exclude over reject.

## Q11.4 Migration and order of work

Decision: option A; converter as a 2.x-line script; the order in the rule set.

Reasoning:
1. Decision 10 batches breaking changes into one release, so dual acceptance (option B) is a second surface with two meanings to keep equal for no v3 gain.
2. Manual migration (option C) is unsafe over 25 Python files in taidl, act-oracle, and act-backend plus 7 under act (tutorials and `tests/test_end_to_end.py`, grep this session), many of which exercise state, constraints, and updates that the examples do not.
3. The converter is one visitor over trees v2 already holds (`instr_input_asts`, `constraint_asts`, `instr_semantics_ast`, taidl/taidl/instruction.py:44-55), so Q3.12's capacity reason for "no converter" (meeting/panel-3/decision.md:182) does not apply; Q3.12's mechanism, one coded rejection naming the route, applies to `add_semantics` unchanged.
4. No plan names an outside owner of a v2 specification, so a `taidl migrate` command in a public 2.x release is a support commitment with no known user; a script on the 2.x line migrates the corpus and any specification that appears, and a release can wrap it later if one does.
5. Building `check` and the `Model` over the string front end first, and porting every generator to read only the `Model`, makes the surface swap a front-end-only change and keeps each port testable on unchanged inputs; the tree-to-`Model` builder is then the converter's front half and the equivalence oracle, not throwaway.
6. `Model` equality up to call sites tests the contract itself; printing HLO text and reparsing with the 2.x parser tests the printer as much as the surface.

Order of work:
1. Operation table as data and the typed node model (Q3.4).
2. `check` producing the `Model` from the string front end; every generator ported to read only the `Model`.
3. The tracer producing the same `Model`.
4. The converter on the 2.x line, run on the corpus; converted files committed.
5. Equivalence test per corpus file; run-time-assembled tests rewritten by hand.
6. String surface, `taidl/antlr4/`, grammar tests, and the dependency deleted in v3.

Dissent: Opus recommends shipping the converter in a 2.x release; Sonnet calls it a time-boxed add-on; both agree it is not required, and reasoning 4 settles it.
Confidence: medium-high.

## Escalation to the owner

- None; decisions 10, 28, 34, 38, 39, 40, 41, and 42 cover every split, and no ruling changes direction, scope, or an external commitment.
