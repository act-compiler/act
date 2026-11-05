# Panel 11, round 1: Sonnet

Lens: a hardware architect writing an ISA specification for their own accelerator.
Evidence tags: T = /home/user/taidl, O = /home/user/act-oracle, B = /home/user/act-backend, S = spec-draft under /home/user/act/plans/language/spec-draft, D = owner decisions.
External sources were fetched this session; where a host was blocked or a page was thin, the step says so.

### Q11.1 Authoring surface

Position: A, a Python embedded DSL built by tracing ordinary function calls, with no AST rewriting and no new dependency.

1. Decision 41 asks for few dependencies and architect comfort (D41); `taidl` today depends on `antlr4-python3-runtime==4.9.2` (T pyproject.toml, `dependencies`), and Python 3.14 is already required (T pyproject.toml, `requires-python`).
2. A Python surface removes the one runtime dependency and the generated parser modules (T taidl/antlr4/, 18 files); a tracing DSL needs only the standard library.
3. Option B repeats today's cost: error positions inside a string are recomputed by hand, with a line and column offset from the backtick token (T taidl/instruction.py:366-400).
4. Option B also has no editor support, no debugger, and no way to loop; an ISA family such as load_01 and load_03 is copied text today (T examples/mini_trn_double.py:17-40).
5. Option C needs `mlir-tblgen` built first and a C++ and CMake or Bazel toolchain; ODS is a build-time C++ generator (mlir.llvm.org/docs/DefiningDialects/Operations/, fetched).
6. Option D has the same toolchain cost, and an architect edits a spec then waits for a compiler; decision 28 allows new dependencies only when an item needs them, and nothing here does.
7. Precedent for A is wide: Exo defines instructions as decorated Python procedures with asserts for applicability (exo-lang.dev; plans/priorart/exo.md "What it is"), and NKI instructions are Python calls with an `accumulate` argument (awsdocs-neuron.readthedocs-hosted.com, nki.isa.nc_matmul, fetched).
8. JAX records operations by running the function on tracer stand-ins that carry shape and dtype (docs.jax.dev/en/latest/key-concepts.html, fetched); the same trick gives TAIDL shape inference at the author's line with no parser.
9. Tracing beats AST parsing (Triton, Exo, Allo decorators read the function source) for TAIDL: a traceback of a tracing error names the author's own line for free, and no restricted-Python subset needs documenting.
10. The traced result is a typed operation graph, which is exactly what panel 3 wants generators to read instead of unstable ANTLR trees (panel 3 decision Q3.4; S object_model.md "Trees"); the oracle's expression writer is an ANTLR visitor today (O taidl_to/expression_writer.py:3-9) and would read typed nodes.
11. Known costs of tracing: `if`, `and`, `or`, and chained comparisons on traced values must raise a clear error ("use & and |; Python control flow on an attribute is not allowed"); `==` is overloaded so nodes hash by identity.
12. Python control flow on plain ints stays legal and is a feature: it runs once at specification time and builds instruction families.
13. The HLO text stays as a printed form of the graph (for logs, the oracle's `.hlo` if a generator wants it, and migration equivalence in Q11.4); I have not verified which generator needs the text.
14. Rejected ground: no JAX or Triton import; I borrow the idea, not the package.

Confidence: high for A over B, C, and D; medium for tracing over AST reading (the second needs `inspect.getsource` and breaks in a REPL or exec).

Would change my mind: evidence that a generator must consume the semantics from a file with no Python at all (the fsim-lib direction of D23), which would argue for a serialized graph (the panel 3 second-consumer trigger), not for a text language.

### Q11.2 Regions, attributes, state, constraints, markers, layers

Position: one decorated function per instruction; attributes are function parameters; a region is a value returned by `read` and consumed by `write`.

1. Decision 34 says attribute names are plain variables; Python parameters are exactly that, with no `@a.` or `@c.` text (D34).
2. The kind (computational or addressing) is a parameter annotation, `Comp` or `Addr`, and the width layer is the same annotation with a type, `Addr[u16]` (D38, S layers.md S11.2).
3. A region is `buffer.region(start, size)`; inside the body, `read` returns the value `%In<k>` and `write` consumes `%Out<k>`, so the numbering and the "semantics has no %In1" check disappear by construction (T instruction.py:142-161).
4. State is declared with the buffers; in a body a state object always reads the pre-issue value, so S5.4 parallel assignment holds in any statement order (S instructions.md S5.4).
5. A constraint is `require(cond)` with Python operators; `&` and `|` stand for and and or, so `same or apart` is `|` (S S5.3).
6. A marker is a keyword on `write`, naming the input value: `inplace=x`, `apart=[x]`, `no_partial_overlap=[x]`, `may_overlap=[x]`; it expands to constraints exactly as D30 and S5.5 say, and a mark can no longer be reset by a later `set_inputs` (S S5.5 "Mark calls").
7. The encoding layer is a separate call keyed by instruction, `spec.encode(gemm_acc, ...)`, so layer 0 text never changes when a layer is added (D38).
8. Generator metadata is `meta={'allo': {...}}` on `Accelerator`, `buffer`, or `instruction`; the namespace is the key (D39, S generator_metadata.md S12.1).

mini_trn `load_rm` and `gemm`, today (T examples/mini_trn.py:18-27, 79-88) and proposed:

```python
from taidl import Accelerator, Comp, Addr, read, write, require
from taidl.ops import reshape, bitcast_convert, dot, convert, bf16, f32

mini_trn = Accelerator('mini_trn')
hbm = mini_trn.hbm                               # byte buffer, access dim = hbm size
d1 = mini_trn.buffer('d1', access=[128], unit=[64], dtype=bf16)
d2 = mini_trn.buffer('d2', access=[64], unit=[64], dtype=bf16)

@mini_trn.instruction
def load_rm(n: Comp, addr_in: Addr, addr_out: Addr):
    x = read(hbm.region(addr_in, n * 128))            # u8[n * 128]
    y = bitcast_convert(reshape(x, (n, 64, 2)), bf16) # bf16[n, 64]
    write(d1.region(addr_out, n), y)

@mini_trn.instruction
def gemm(addr_1: Addr, addr_2: Addr, addr_out: Addr):
    a = read(d1.region(addr_1, 64))
    b = read(d1.region(addr_2, 64))
    write(d2.region(addr_out, 64), dot(a, b, lhs_contracting=[1], rhs_contracting=[0]))
```

An accumulate instruction, with a float32 accumulator buffer, a start such as `1024 - addr`, a width layer, a marker, and a constraint:

```python
acc = mini_trn.buffer('acc', access=[64], unit=[64], dtype=f32)

@mini_trn.instruction(meta={'allo': {'engine': 'tensor'}})
def gemm_acc(addr_1: Addr[u16], addr_2: Addr[u16], addr_acc: Addr[u16]):
    a = read(d1.region(addr_1, 64))
    b = read(d1.region(addr_2, 64))
    old = read(acc.region(1024 - addr_acc, 64))       # start is any expression
    prod = convert(dot(a, b, lhs_contracting=[1], rhs_contracting=[0]), f32)
    write(acc.region(1024 - addr_acc, 64), old + prod, inplace=old)
    require(addr_1 != addr_2)

mini_trn.encode(gemm_acc, word=32, fixed={'[31:24]': 0x12},
                fields={'addr_1': '[23:16]', 'addr_2': '[15:8]', 'addr_acc': '[7:0]'})
```

9. `old + prod` is the operation `add`; the Python operators `+ - * / %` map to the S8 operation table, and every other operation is a function named as in StableHLO (D14, S operations.md).
10. An overlap that the architect declares is visible on the line of `write`; the expansion `same(In k, Out)` is generated, never written (S5.5 "Expansions").
11. Accumulate by NKI is a flag `accumulate=True` on one call, with undefined behavior on an uninitialized PSUM location (awsdocs-neuron.readthedocs-hosted.com, nc_matmul, fetched); TAIDL has no undefined behavior (D16), so the sketch uses a plain read-modify-write, and "first write" is a separate non-accumulating instruction or a state flag, because expressions have no conditional (S expressions.md line 32).
12. The `Addr[u16]` annotation is evaluated lazily on Python 3.14, which is already required (T pyproject.toml).
13. Loop-generated families use `@mini_trn.instruction(name=...)`, since the function name is only the default.

Confidence: medium-high on the shape of `read`/`write`; medium on the exact spelling (`region`, `Addr[u16]`, keyword markers), which a style pass can change cheaply.

Would change my mind: a case study whose instruction has data-dependent region counts that `read` and `write` in a body cannot order statically (check plans/casestudies/ for multi-output cases).

### Q11.3 The check function

Position: A, with one amendment: no registry, each generator's entry point calls the common check and then its own.

1. Panel 3 already fixed `validate()` as the publisher of derived facts on a typed model (panel 3 decision Q3.2, Q3.5); decision 42 only names the call `check` and orders it first (D42).
2. `check(spec)` returns a `CheckedSpec`; `generate_oracle` and `generate_backend` accept only that type, and given a raw `Accelerator` they call `check` themselves, once; "always runs once before any generator" becomes a type fact, not a convention.
3. B (subclassing one checker) couples every generator to taidl's class layout and makes the order of base and extra checks implicit; A keeps the common check a function.
4. A registry (global `register_check`) adds import-order state; the rule "touches neither taidl nor another generator" (S generator_metadata.md S12.4) is met equally by an explicit call from the generator's entry point.
5. A command `python -m taidl check spec.py` runs only the common check, which is the standalone use that D42 asks for.
6. What the common check covers: names unique and identifiers (T names.py:8-14); every `read` and `write` region fits its buffer when literal; every `write` has a value of the region shape and type; markers rejected at declaration (S S5.5 "Rejected at declaration"); literal 64-bit and divisor-0 faults (T instruction.py:403-438); layer checks of S11.3 and S11.4 (slice coverage, no two opcodes matching one word, one link per derived attribute); metadata is plain data (S S12.2).
7. Errors raised while tracing already carry the author's traceback; errors found by `check` after tracing carry the call site that each node records at construction, which panel 3 listed as a typed member (panel 3 decision Q3.5).
8. `check` collects every problem and raises one `SpecError` listing `file:line: instruction: section-id: message`, so an architect fixes five typos in one run; each entry is correctable by the architect, per decision 40 (D40, S overview.md S1.1).
9. Backend extra checks: state is initialized and integer (machine-state requirements), each state variable the search must model is supported, and every operation in the semantics is in the backend's supported list, rejected naming the feature and its S id (S overview.md S1.2; B tests/test_refusals.py:89, 408 show state-related refusals today).
10. Oracle extra checks: expected none (D42); its unsupported operations come from the same S1.2 table, if any appear after the corpus run.
11. A generator's extra check receives the `CheckedSpec`, never the raw trace, so it cannot re-derive what the common check published.

Confidence: high on A over B; medium on registry versus explicit call (a stylistic point, not binding).

Would change my mind: a third generator that needs to add checks before the common check can finish (for example a pre-trace option), which would justify a hook.

### Q11.4 Migration and order of work

Position: A, a one-time converter, run on every example and test spec, and the string form removed in v3.

1. Decision 10 batches all breaks into one release (D10); accepting both forms (B) keeps ANTLR and two code paths alive through v3 and doubles the corpus of tests for no user benefit.
2. C (manual) is unsafe: 40 Python files call `add_semantics` across taidl, act-oracle, and act-backend (grep over /home/user, this session), and a hand edit can silently change a shape.
3. The old parser already produces the trees a converter needs (T taidl/antlr4/IDLV2Parser.py), so the converter is a visitor printing the new surface; no new parsing work.
4. Ship the converter inside the last 2.x release (`taidl migrate`), so v3 itself has zero ANTLR code and zero dependency on it (D33 pins released versions; D28).
5. Equivalence proof per converted spec: trace the new file, print the graph as HLO text, parse the print with the 2.x parser, and compare to the old tree modulo whitespace; then run the oracle on the old and new forms and compare outputs on the existing corpus.
6. The converter output is formatted by the project formatter, and comments on the old `.add_semantics` lines carry over as comments (engineering.md comment rules).
7. Order of work:
   - Item 1: operation table as data, one source for the S8 text and the Python functions (D14).
   - Item 2: tracer, typed graph, and `CheckedSpec` with the panel 3 contract.
   - Item 3: `check` with the common checks.
   - Item 4: HLO-text printer from the graph (for step 5 and for logs).
   - Item 5: converter in 2.x, run on the 40 files; commit the converted examples and tests.
   - Item 6: move generators from ANTLR visitors (O taidl_to/expression_writer.py:9-66) to typed nodes.
   - Item 7: delete `taidl/antlr4/`, the grammar tests (T tests/test_grammar.py), and the dependency.
8. Items 1 to 4 are parallel to the generator work; item 6 is the widest change and sets the critical path.
9. Capacity: roughly one engineer on items 1 to 4, one on item 5, one or two on item 6, within the 3-4 engineers and two quarters of D11.

Confidence: high on A over B and C; medium on shipping the converter as a 2.x command rather than a script in the v3 repo (the second keeps ANTLR in a tools folder).

Would change my mind: an external user base with large private ISA specifications that cannot all migrate in one release, which is evidence for B; no such evidence is in the plans.
