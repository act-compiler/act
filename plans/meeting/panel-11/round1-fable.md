# Panel 11, round 1, Fable

Lens: DSL and compiler-infrastructure architect.

### Q11.1

Position: option A, a Python embedded DSL built by tracing, with the standard library as its only dependency and the ANTLR runtime removed.

1. Today an instruction is one Python call plus three string grammars, and every string is parsed at the call by `antlr4-python3-runtime==4.9.2` (taidl/taidl/instruction.py:4-10, :246-258, :331-364; taidl/pyproject.toml, `dependencies`).
2. The generated parser modules are 3,200 of the package's 4,360 lines (`wc -l taidl/taidl/antlr4/*.py`), so the string surface costs more code than the language it carries.
3. The expression grammar is a tiny integer language: `+ - * / %`, comparisons, `and`, `or`, and three reference kinds (taidl/taidl/antlr4/IDLV2Expr.g4:14-25, :34-36); Python's own operators cover it exactly, with `__add__`, `__eq__`, `__and__` overloads on an attribute object.
4. The semantics body is a pure dataflow block of named values, one word per line with a declared result type (taidl/examples/mini_trn.py:21-27); that is precisely what JAX's `make_jaxpr` records when it runs a Python function on tracers and emits a typed first-order dataflow graph (docs.jax.dev, jaxpr page).
5. NKI gives the same shape to an architect: `nki.isa.nc_matmul(dst, stationary, moving, ..., accumulate=None)` is a Python call with keyword flags, and its scan instruction's semantics are a NumPy-style loop (awsdocs-neuron, nki.isa.nc_matmul and nki.isa.tensor_tensor_scan pages).
6. Exo's `@instr` puts name, typed arguments with memory annotations, applicability asserts, and the body in one decorated function (exo-lang.dev, docs/instructions.md); plans/priorart/exo.md:19 already names this as the shape to borrow.
7. Option C costs an LLVM build: Allo requires "the latest LLVM project" built from source with MLIR Python bindings and offers no wheel (cornell-zhang.github.io/allo, setup page); ODS needs `mlir-tblgen` and emits C++ (mlir.llvm.org, DefiningDialects/Operations); both fail decision 41's "as few dependencies as possible" and decision 28, which admits only cbindgen, hypothesis, and mutmut.
8. Option D needs a C++ toolchain to read a spec, and Sail needs opam and OCaml (github.com/rems-project/sail README); neither is comfortable for an architect who today writes 100 lines of Python.
9. Option B keeps a parser, and a parser is the thing that cannot point at the author's line without the `token.line - 1` arithmetic of `text_error` (instruction.py:366-383); a traced Python body fails inside the author's own frame, so the traceback names file and line for free.
10. Panel 3 fixed the contract as a typed Python model plus a typed node model with ANTLR trees marked unstable (plans/meeting/panel-3/decision.md:62-74); tracing builds that node model directly, so option A also deletes the tree-to-model translation step.
11. The one real cost of tracing is Python control flow on traced values, which JAX resolves by forbidding it and offering `lax.cond` (docs.jax.dev, jaxpr page); TAIDL's bodies have no control flow, only `select`, so the trap does not arise, and an `if` on an attribute object raises at the author's line.

Evidence: steps above, each cited inline.
Confidence: high.
What would change my mind: a count showing most case-study sketches need a construct Python expressions cannot overload cleanly (for example chained comparisons `a < b < c`, which Python evaluates as `and` of bools and an overload cannot see), or an owner statement that the spec must be readable without Python installed.

### Q11.2

Position: regions are `buffer[start:start+size, ...]` slices on buffer handles, attributes are typed parameters of a decorated function, state and updates are attributes on an accelerator object, constraints are a `require(...)` call, markers are keyword flags, and the width, encoding, and generator-metadata layers are optional keyword arguments on the same decorator.

1. Decision 34 makes attribute names plain variables, so attributes become function parameters and `@a.`/`@c.` prefixes disappear; the addressing/computational split becomes a parameter annotation (`Addr`, `Comp`), which the checker reads, not the name.
2. A region today is `[buffer, starts, sizes]` with per-buffer arity (instruction.py:115-129); a slice on a buffer handle carries the same three facts, Python checks arity at trace time, and `buffer[s:s+n]` reads as the architect writes it in NKI (`sbuf[0:128, 0:512]` style in every nki.isa example).
3. Markers stay shorthand per decision 30; `inplace=` and `apart=` keyword lists name inputs by parameter name and expand to the constraints `keep_off` writes today (instruction.py:190-231).
4. Decision 38's width layer is an annotation argument, `Comp[u8]` or `Addr[s16]`, default `s64`; decision 39's metadata is one `meta={'allo': {...}}` argument that the checker stores and only `generate_allo()` reads.
5. The accumulate case is Trainium2 `nc_matmul`, where today a flag becomes `constant(@c.acc)`, `compare`, `broadcast`, `select`, and `dynamic_update_slice` across 16 lines (plans/casestudies/trainium2.md:31-57, :109); in the surface below it is one `where`.

```python
from taidl import Accelerator, Addr, Comp, u8, bf16, f32, ops

trn = Accelerator('mini_trn')
d0 = trn.buffer('d0', u8)
d1 = trn.buffer('d1', bf16, access=[128], unit=[64])
d2 = trn.buffer('d2', bf16, access=[64], unit=[64])

@trn.instruction
def load_cm(n: Comp, addr_in: Addr, addr_out: Addr):
    src = d0[addr_in : addr_in + n * 128]            # u8[n*128]
    a = ops.reshape(src, [n, 64, 2])
    b = ops.bitcast_convert(a, bf16)                   # bf16[n,64]
    d1[addr_out : addr_out + n] = ops.transpose(b, [1, 0])
```

```python
psum = trn.buffer('psum', f32, access=[128, 8], unit=[512])
sbuf = trn.buffer('sbuf', u8, access=[128], unit=[])

@trn.instruction(inplace=['bank_in'], meta={'allo': {'engine': 'tensor'}})
def nc_matmul(k: Comp, m: Comp, n: Comp, acc: Comp[u8],
              ws: Addr, mv: Addr, bank: Addr):
    trn.require(k <= 128, m <= 128, n <= 512, (acc == 0) | (acc == 1))
    w = ops.bitcast_convert(ops.reshape(sbuf[0:k, ws : ws + m * 2], [k, m, 2]), bf16)
    x = ops.bitcast_convert(ops.reshape(sbuf[0:k, mv : mv + n * 2], [k, n, 2]), bf16)
    old = psum[0:m, bank : bank + 1][:, 0, 0:n]          # f32[m,n]
    p = ops.dot(w, x, lhs_contracting=[0], rhs_contracting=[0])
    base = ops.where(acc == 1, old, ops.zeros_like(p))
    psum[0:m, bank : bank + 1][:, 0, 0:n] = base + p
```

6. The second sketch keeps every v3 rule: the implicit bounds constraint comes from the slice, `acc == 1` is a traced predicate so `where` is `select` on `broadcast(compare(constant(acc), 1))` in the node model, and the partial-unit write is the `dynamic_update_slice` idiom the case study already uses, now emitted by the tracer rather than typed by the architect.
7. A state variable is `trn.state('rows', 0)` read as `trn.rows` inside a body and updated by `trn.rows = ...` at the body's end, which the tracer records as the update list of S5.4; Sail's `register` plus `execute` clause is the same split (github.com/rems-project/sail README).

Evidence: steps above.
Confidence: medium-high on the shape, medium on the exact spelling of the partial-unit write.
What would change my mind: a demonstration that slice assignment to a buffer handle cannot be made to raise on a second write to the same output, or that the `[:, 0, 0:n]` sub-view confuses the implicit bounds constraint.

### Q11.3

Position: option A, `taidl.check(acc)` returns the validated typed model, and a generator passes a list of plain check functions that run on that model; no subclassing.

1. Panel 3 settled that `validate()` publishes derived facts as typed members that raise before it runs (plans/meeting/panel-3/decision.md:76-88); `check` is that function under decision 42's name, and the typed model is its return value.
2. `check` owns what today sits in `parse_semantics`, `parse_text`, `check_values`, and `keep_off`: names declared, one ROOT, operand counts and attribute forms per word (taidl/taidl/operations.py:41-80), 64-bit literals, region arity, value-to-region pairing, and marker consistency (instruction.py:142-161, :260-319, :331-364).
3. It adds the v3 facts the tracer makes cheap: a region start is any expression over attributes, bounds constraints are implicit (decision 34, 30), the width layer makes an attribute's declared width check its literals (decision 38), and metadata keys name a known generator (decision 39).
4. Backend extra checks are the machine-state requirements decision 42 names and the supported-operation table, today in `spec_checks.py:394-418`, which panel 3 already said becomes "call `validate()` first" (decision.md:37-40); they stay in act-backend as functions `(model) -> list[Rejection]`.
5. The oracle registers none, as decision 42 states; `act_oracle.generate(check(acc))` is the whole call.
6. A function list beats subclassing: MLIR attaches verifiers per op as traits and a flat `verify()` and orders them structurally first, generated invariants second, custom last (mlir.llvm.org, DefiningDialects/Operations); the same order falls out of "shared check, then the generator's list" with no class hierarchy to version.
7. Errors are the `Rejection` class panel 3 named (decision.md Q3.10 row) with a code, the instruction name, and the author's file and line; the line comes from the decorated function's `__code__.co_firstlineno` or from the traced call's frame, not from counted backticks (contrast instruction.py:379-383).
8. Decision 40 limits rejections to what the architect can correct, so a backend check that finds an unsupported operation word rejects with the word and the line, and a cost-model gap is a recorded reason at selection, never a check failure.

Evidence: steps above.
Confidence: high.
What would change my mind: a generator that must run a check before the shared one (for example to accept a word taidl rejects), which would argue for an ordered hook instead of a trailing list.

### Q11.4

Position: option A, a one-time converter run once on the examples and tests, with the string form gone in v3.

1. Decision 10 batches every breaking change into one v3 release, so option B's dual acceptance is a second surface to document and test for no v3 gain.
2. The converter is cheap because the parser exists: the ANTLR tree already yields each line's word, operands, attributes, and backtick expressions (instruction.py:246-258), and a visitor prints the Python form of Q11.2 line for line.
3. The public corpus is small: two examples (taidl/examples/mini_trn.py, mini_trn_double.py), the generator tests, and the case-study sketches under plans/casestudies/, none of which uses state, constraints, or updates (plans/casestudies/gaps.md:45).
4. The converter is a development script, not a shipped feature: panel 3 ruled that removed names get one coded rejection naming the replacement and no shipped converter (decision.md Q3.12 row); the same rejection catches a stray `add_semantics("""...""")` call in v3.
5. Order of work: the tracer and node model first, `check` on the node model second, the converter third, examples and tests converted and diffed against the old parse fourth, ANTLR deleted last in the same release.
6. Step four is the safety net: converting each example and comparing the traced node model with the ANTLR-parsed one proves the new surface loses nothing before the parser is removed.

Evidence: steps above.
Confidence: medium-high.
What would change my mind: a count of private ISA specifications at the owner's partners that cannot be reconverted by script, which would argue for option B for one minor release.
