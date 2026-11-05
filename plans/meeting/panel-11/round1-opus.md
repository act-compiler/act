# Panel 11 round 1: Opus

Lens: language designer and front-end maintainer.
Code paths are relative to /home/user; plan paths are relative to /home/user/act/plans.

### Q11.1

Position: option A, a Python embedded DSL built by tracing (one call of a decorated function on symbolic attributes), never by parsing the function's source.

1. Today an error in a semantics block names a line of the string, not of the author's file: `parse_semantics` and `text_error` report "semantics line {line}, column {column}" counted inside the triple-quoted text.
2. Today one value is written twice and checked for agreement: the region gives `['@c.n * 128']` and the block's `parameter(0)` line restates `u8[@c.n * 128]`, which `check_values` must reconcile.
3. Option B (own textual language) keeps a parser and its runtime pin, and gives no editor, debugger, or completion support until someone writes them.
4. Option C requires TableGen and a C++ build, since ODS records are expanded into `mlir::Op` C++ specializations at compiler build time; that contradicts "as few dependencies as possible" (decision 41).
5. Option D puts a C++ toolchain in front of an architect for a file that today is plain Python.
6. Within option A, source parsing (Exo's procedure bodies, TorchScript-like subsets) defines a language that looks like Python but is not; NKI moved its front end from a "limited subset of Python" to tracing.
7. Tracing gives Python's own meaning to every line: Python control flow runs once at definition time, as in JAX ("Python loops are inlined during tracing"), so a loop over variants is ordinary Python.
8. Each recorded node takes its call site from the first frame outside `taidl` (stdlib `sys._getframe`; `Accelerator.__init__` already walks a frame), so every error points at the author's file and line, which panel 3 Q3.5 wants as a typed member.
9. The tracer builds panel 3's typed node model directly (Q3.4), so generators read the same contract with no ANTLR tree in between.
10. Dependencies fall to zero third-party packages: `antlr4-python3-runtime==4.9.2` leaves taidl.
11. Tracing has three hazards, each closed with an error at the author's line rather than a silent meaning:
    - a symbolic condition used by `if`, `and`, `or`, `not`, a chained comparison, or `max()` calls `__bool__`, which raises (torch.fx refuses the same case: "symbolically traced variables cannot be used as inputs to control flow");
    - `/` on a symbolic integer raises and names `//`, since Python `/` is true division while S6 `/` is floor division;
    - a Python integer literal outside signed 64 bits raises at the expression that holds it.
12. With that, S6 arithmetic is Python's own: `//` floors and `%` takes the divisor's sign in both, so `a == (a // b) * b + a % b` holds unchanged.

Evidence:
- taidl/taidl/instruction.py:264-266, :375-383 (string-relative line); :142-160 (`check_values`); examples/mini_trn.py:19-25.
- taidl/pyproject.toml:15; taidl/taidl/accelerator.py:25.
- https://mlir.llvm.org/docs/DefiningDialects/Operations/ (ODS expanded to C++ at build time).
- https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/deep-dives/nki-compiler.html (NKI 0.7 tracing; pre-0.7 "limited subset of Python").
- https://docs.jax.dev/en/latest/601/jaxpr.html (Python control flow inlined at trace time).
- https://docs.pytorch.org/docs/2.15/fx.html (TraceError on symbolic control flow).
- https://docs.python.org/3/reference/expressions.html (chained comparison expands to `and`; `//` floors; `%` sign of divisor).
- act/plans/language/spec-draft/expressions.md:28-29; act/plans/priorart/exo.md:6.
- Confidence: high on A; medium-high on tracing over source parsing.
- Would change my mind: a corpus specification whose meaning needs a construct tracing cannot record (a loop over a symbolic bound inside one instruction), which source parsing could.

### Q11.2

Position: attributes are typed function parameters, regions are buffer method calls with keyword `size=`, everything that names an attribute lives in the body, and result types are inferred except where an operation needs a target.

1. Attributes are function parameters whose annotation gives the kind (`Comp`, `Addr`) and, as layer 1, the type; parameter order is the attribute order, and no name carries meaning (decision 34).
2. The `@c.`, `@a.`, and `@s.` prefixes disappear from the surface; the kind is a declaration, and a state variable is a Python object returned by `state()`.
3. A region is `buffer.read(starts, size=...)` or `buffer.write(starts, size=..., value=...)`; a start is any expression, such as `1024 - addr_out` (decision 34).
4. Python slice syntax is refused for regions, since `d0[a:b]` reads as start and stop to every Python reader, while a TAIDL region is start and size.
5. A read's value has the type the region gives (element type, sizes, then the unit dims), which removes the duplicated `parameter(k)` line of point 2 in Q11.1.
6. Reshape, broadcast, convert, bitcast_convert, iota, and constant take their target shape or type, as XLA's builder does; every other result is inferred by S4's shape rules, and `expect(value, type, dims)` asserts one where the author wants a check.
7. Constraints are `require(c1, c2, ...)`, a conjunction; disjunction is `any_of(...)`, since Python's `and` and `or` cannot be overloaded.
8. A marker is a keyword on the input read (`mark=INPLACE`, `APART`, `NO_PARTIAL_OVERLAP`) and expands to constraints exactly as today (decision 30); the one-output rule of today's marks stays.
9. A state update is `update(ctr, ctr + 1)` and all updates of one issue form one parallel assignment (S5.4).
10. The width layer is the annotation argument, `Addr('u16')`, defaulting to `s64` (decision 38); the encoding layer is one `encode(...)` call in the body, so fields name attribute objects, not strings.
11. Generator metadata is the `meta=` keyword on `Accelerator`, `buffer`, and the instruction decorator, stored unread (S12.2, decision 39).
12. Annotations are read with `inspect.get_annotations(eval_str=True)`, so a file with `from __future__ import annotations` means the same thing.

One mini_trn instruction, load_rm (examples/mini_trn.py:18-27):

```python
from taidl import Accelerator, Comp, Addr, ops

mini_trn = Accelerator('mini_trn')
d0 = mini_trn.d0
d1 = mini_trn.buffer('d1', [128], [64], 'bf16')
d2 = mini_trn.buffer('d2', [64], [64], 'bf16')

@mini_trn.instruction
def load_rm(n: Comp, addr_in: Addr, addr_out: Addr):
    raw = d0.read([addr_in], size=[n * 128])      # u8[n*128]
    a = ops.reshape(raw, [n, 64, 2])              # u8[n,64,2]
    d1.write([addr_out], size=[n], value=ops.bitcast_convert(a, 'bf16'))
```

One accumulate instruction, with a marker, a constraint, the width and encoding layers, a derived attribute, state, and metadata:

```python
from taidl import INPLACE, require, update, encode, ops

calls = mini_trn.state('calls', 0)

@mini_trn.instruction(meta={'allo': {'engine': 'tensor'}})
def gemm_acc(n: Comp('u7'), a1: Addr('u8'), a2: Addr('u8'),
             out: Addr('u6'), out_b: Addr):
    require(n >= 8, n % 8 == 0, out_b == 2 * out)   # out_b is derived (S11.4)
    lhs = d1.read([a1], size=[n])                   # bf16[n,64]
    rhs = d1.read([a2], size=[64])                  # bf16[64,64]
    acc = d2.read([out_b], size=[n], mark=INPLACE)  # bf16[n,64]
    prod = ops.dot(lhs, rhs, lhs_contracting=[1], rhs_contracting=[0])
    d2.write([out_b], size=[n], value=ops.add(acc, prod))
    update(calls, calls + 1)
    encode(32, fixed=[((31, 29), 0b101)],
           fields=[(n, [(28, 22)]), (a1, [(21, 14)]), (a2, [(13, 6)]), (out, [(5, 0)])])
```

Evidence:
- taidl/taidl/instruction.py:96-113, :163-226 (regions and marks today); act/plans/language/spec-draft/instructions.md:30, :61-62, :78-79.
- act/plans/language/spec-draft/layers.md:15-30, :34-39, :50-59; generator_metadata.md:8-19.
- https://openxla.org/xla/operation_semantics (Reshape and BroadcastInDim take target dimensions).
- https://docs.python.org/3/reference/expressions.html (`and` and `or` test truth, no operator hook).
- act/plans/casestudies/trainium2.md:109 (today's accumulate needs `select` and `dynamic_update_slice` workarounds).
- Confidence: medium-high on regions, constraints, and markers; medium on inferred result types.
- Would change my mind: a corpus operation whose result shape the S4 rules cannot infer without an author-written type.

### Q11.3

Position: option A with a type barrier: `taidl.check(acc)` returns a frozen `Model`, the only input any generator entry accepts, and each generator then runs its own plain function over that model; no registry.

1. Only `check()` constructs a `Model`, so "always runs once before any generator" (decision 42) holds by type, not by convention.
2. `check()` subsumes panel 3's `validate()`: it attaches derived facts and call sites as typed members (Q3.5), and the `Accelerator` is frozen once checked.
3. A generator entry `generate_x(model, **options)` calls `x_checks(model)` first; a registry or a checker base class (option B) is abstraction before a second use site needs it.
4. `check()` collects every error, each with code, section id, and the author's file and line (decision 40, Q3.8), and raises one `CheckError` listing them; first-error-only hides the second typo.
5. Most of `spec_checks.py` duplicates front-end rules (semantics form, sizes, literals, regions) and moves into `check()`.
6. What stays backend-only are machine and allocator limits: one output per instruction (`check_outputs`), no state read or update (`check_state`), constraints the backend can hold (`honoured`), and sizes free of address attributes and state (`beyond_allocator`).
7. Those limits are not architect errors under decision 40: a valid instruction the backend cannot model is dropped from the backend's instruction set with a recorded reason, and only an empty usable set rejects.
8. The oracle adds no check (decision 42).
9. The `allo` generator's own check rejects unknown keys in its namespace (S12.3).

Evidence:
- act/plans/meeting/panel-3/decision.md Q3.2, Q3.4, Q3.5, Q3.8.
- act-backend/act_backend/spec_checks.py:22-29, :277-281, :331-347, :370-386 (backend-only); :179-243, :284-308, :394-418 (front-end duplicates).
- act/plans/language/spec-draft/generator_metadata.md:24-27.
- https://docs.jax.dev/en/latest/601/jaxpr.html (a traced function yields one closed jaxpr that later stages consume, the same staging as `Model`).
- Confidence: high on the barrier and the move; medium on dropping over rejecting backend-unsupported instructions.
- Would change my mind: an owner reading of decision 42 that the backend's machine-state requirement must reject the whole specification.

### Q11.4

Position: option A, a one-time converter shipped in a final v2.x release (it imports v2 taidl and its ANTLR trees), with v3 carrying no string form and no ANTLR.

1. Order of work, each step independently testable:
   - step 1: the typed node model (panel 3 U2), independent of any surface;
   - step 2: `check()` producing it from today's string front end, and every generator ported to read only `Model`;
   - step 3: the traced surface producing the same `Model`;
   - step 4: the converter, run on taidl, act-oracle, and act-backend examples and tests;
   - step 5: an equivalence test per corpus specification, string-built `Model` equal to converted-surface `Model` up to call sites;
   - step 6: delete the string surface and the ANTLR grammars in v3.
2. Step 2 before step 3 makes the surface swap a front-end-only change, since no generator reads a parse tree afterwards.
3. Conversion is mechanical: each v2 `Instruction` already holds parsed regions, constraints, updates, and a semantics tree (`instr_input_asts`, `constraint_asts`, `instr_semantics_ast`).
4. Hosting the converter in v2 keeps ANTLR out of v3; owners of private specifications run it once before upgrading (decision 33 pins released versions).
5. Option B ships two surfaces with two meanings to keep equal, against decision 10's single breaking release.
6. Option C hand-rewrites every test and example and has no equivalence oracle.
7. Panel 3's "no converter" (Q3.12) covers renamed API members, where a coded rejection names the replacement; a whole-surface change has no one-line replacement to name.

Evidence:
- taidl/taidl/instruction.py:44-55 (parsed members a converter reads).
- taidl/taidl/antlr4/IDLV2.g4:6-25 and IDLV2Expr.g4:6-25 (the full surface a converter covers, about 165 lines).
- act/plans/meeting/panel-3/decision.md Q3.4, Q3.12.
- https://docs.python.org/3/reference/expressions.html (S6 operators map one to one onto Python `//` and `%`, so expressions convert textually).
- Confidence: medium-high.
- Would change my mind: evidence that a 2.x release with an added tool breaks decision 10 or 33, in which case the converter lives in a repository script run on the corpus only.
