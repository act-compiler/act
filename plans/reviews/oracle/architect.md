# act-oracle plan review: architect

An independent target for TAIDL-TO, formed from the code before reading `work/oracle/plan/`, then set against the plan step by step.
Paths are relative to `/home/user/act-oracle/taidl_to/` unless they name another repo.

## Own target architecture

- Lowering stays StableHLO text compiled by XLA, since S8.4 defines each word as its StableHLO operation (`taidl/docs/operation_meaning.md:7`) and `backend.compile_and_load` takes text today (`generic/decorator.py:47-61`).
- Every `start_loop` unrolls at trace time, so a loop variable is a plain Python `int` at each call.
- That deletes the symbolic loop algebra (`generic/utils.py:15-260`), the `Unroll` retrace with `deepcopy` (`templates/templates.txt:107-125`, `generic/utils.py:750-761`), `stablehlo.while` emission (`templates/templates.txt:177-237`) and `held`/`fixed` (`expression_writer.py:336-339`).
- Unrolling is already the fallback path: any loop around a state update or a varying shape unrolls (`templates/templates.txt:101-104`, `templates/templates.txt:137-140`).
- Every call-time check already enumerates each iteration (`generic/utils.py:68-72`, callers `generic/utils.py:698-806`), so tracing per iteration adds no asymptotic cost before XLA.
- The runtime is real Python under `runtime/`: about 260 lines of fixed code inside the `API_FILE` template (`templates/templates.txt:26-293`) move out, leaving per-ISA data only (`{{state}}`, data models).
- Generated `api.py` is one thin function per instruction that passes its arguments and a static record to one runtime `call`, so tracebacks still name the instruction.
- The record is data built once by the visitor: regions, lines (word, result, type, dims, operands, attributes), constraints, updates, each expression as Python source.
- With a one-name body, the reserved-name machinery (`api_generator.py:61-86`, `instruction_codegen.py:13-20`, `instruction_codegen.py:209-224`) shrinks to keyword and dunder checks.
- One emitter in `runtime/emit.py` writes MLIR generic form for every word, which every dialect reads (`generic/utils.py:490-495`); only `reduce`, `dot` widening and `constant` need their own function.
- One word table, split by owner: language facts (arity, attributes, types, index operands) in the front end beside `OPERATIONS` (`/home/user/taidl/taidl/operations.py:41`); lowering facts (op name, dialect, attribute mapper, reducer) in the oracle.
- Static checks live in the front end; the oracle keeps only call-value checks (window, tile, constraint, division by zero, expression shapes) and "not supported by the oracle".
- No undefined behavior (ruling 16): addressing values are checked as errors at each call, and element results are fixed by guards in the emitted module.
- Fidelity: literals rounded once from the decimal, no process-wide jax configuration, and a differential run against an independent reference on every word.

## Focus points

- Lowering target: StableHLO text, agreeing with the plan (`decisions.md` round 2).
- `jax.lax` would remove `ssa_types` and type strings, but lacks a `batch_norm_inference` primitive and adds its own decompositions; text names the reference operation directly (`oracle_visitor.py:124`, `oracle_visitor.py:330-335`).
- The StableHLO reference interpreter evaluates the very module text the oracle compiles, so it is the cheapest independent check of XLA's execution against S8.4; it needs CHLO legalized first and a new dependency (ask the owner).
- A NumPy reference interpreter (plan O13) re-derives about 77 words by hand, so it checks lowering bugs like D1 but costs L and can misread gather or `dot_general` the same way the oracle does.
- Recommendation: StableHLO interpreter for differential runs, plus golden module-text tests for lowering; a NumPy interpreter only for words the StableHLO interpreter lacks.
- Splitting `utils.py`: agree on four role files, but delete the loop algebra first; a pure move of 260 lines that a later step deletes is wasted review.
- Six word tables: merge, but the types column and `INDICES` belong to the front end (D7), and `PRETTY` plus `op_to_template` vanish once every word uses generic form.
- Templates vs emitter calls: the plan keeps `API_FILE` as a template; it is mostly fixed runtime code (`templates/templates.txt:26-293`) that ruff and tests cannot see, so it moves first.
- Front-end checks: region type and element count (`semantic_generator.py:41-56`, S7.2 at `/home/user/taidl/docs/semantics.md:43-47`), S8.3 types (`oracle_visitor.py:251-264`), compare type fit (`oracle_visitor.py:276-283`), literal shape rules (`oracle_visitor.py:285-289`) and constant range (`oracle_visitor.py:373-389`).
- The front end skips all of these today (`/home/user/taidl/docs/conformance.md:59`).
- Numerics: D1 rounding (`generic/utils.py:408`) and the `f16` solve widening (`generic/utils.py:498-512`) are real; D11, D19 and D20 come from 64-bit loop arithmetic in emitted text (`generic/utils.py:214-252`) and disappear with unrolling.
- `jax_default_matmul_precision` (`generic/decorator.py:23`) has no effect on a module compiled from text, which carries `HIGHEST` per dot (`generic/utils.py:604`); x64 is needed only for `jax.device_put` (`generic/decorator.py:188`).

## Rulings 16 (no undefined behavior) and 17 (in-place is aliasing)

- One norm proposed for ruling 16: an address, size, attribute, state or expression value that breaks a rule is a defined error at the call; a tensor element value always has a defined result.
- The split follows what the oracle can see: addressing values are concrete Python ints at trace time once loops unroll, while element values exist only inside the compiled module.
- Unrolling becomes close to required: under `stablehlo.while` every check must be proven for all iterations symbolically, which is the source of D11 and D17-D20.
- Element norms cannot be left to XLA CPU: where a reference operation leaves a result open, the emitter writes explicit guards (`select`, clamp) so the module itself fixes the result.
- The lowering table gains one column: the guard each word needs at each type, empty for most words.
- A differential reference cannot settle a norm, since the StableHLO interpreter makes its own choice; each norm case gets a golden test.
- Ruling 17: storage reuse means the marked input's window and its output's window are the same bytes of the same buffer.
- The oracle checks that at the call (same buffer, same starts and sizes, same element count) and raises a defined error otherwise.
- The oracle already loads every input before it stores any output (`instruction_codegen.py:169-181`), so reads see pre-call values; that load-then-store order is the norm for aliasing.
- Open for L: which output a marked input pairs with, since `set_inplace` takes one flag per input only (`/home/user/taidl/taidl/instruction.py:163-172`).

| S1.3 case (`overview.md`) | Oracle today | Target |
| --- | --- | --- |
| i64 overflow in an expression (line 34) | `UserWarning` only (`generic/utils.py:268-271`, callers 287, 659, 706) | Error at the call; the expression writer emits checked ops (`expression_writer.py:253-261`) |
| Region outside its data model (35) | Error (`generic/utils.py:698-712`) | Keep |
| Region size 0 or less (36) | Error (`generic/utils.py:651-666`) | Keep |
| Divisor 0 (37) | Error (`generic/utils.py:75-80`, `generic/utils.py:292-301`) | Keep |
| Shapes or types break the reference op (38) | Oracle checks (`oracle_visitor.py:251-293`) | Front-end error; expression shapes at the call |
| Reference op leaves result open, e.g. float `convert` overflow (39) | Whatever XLA CPU gives | L picks the norm; emitter guards it |
| Region line count or type (40) | Oracle check (`semantic_generator.py:41-56`) | Front-end error |
| Addressing attribute outside a start (41) | Not rejected | Front-end error |
| Two outputs overlap (42) | Not checked in `taidl_to/` | Error at the call from concrete windows |
| Constraint broken (43) | Error (`generic/utils.py:796-804`) | Keep |
| `bitcast_convert` byte order (44) | XLA's | L fixes it; golden test |
| `reduce_<reducer>` initial value (45) | Identity per type (`generic/utils.py:439-451`) | Propose this as the norm |
| Initial state outside i64 (46) | Not rejected | Front-end error |
| In-place meaning (48) | Ignored (D10) | Ruling 17 alias check |

- Not in S1.3 but open under ruling 16: float `reduce` and `dot` association order, which StableHLO leaves to the backend; ask L whether results may differ in the last bits.

## Plan steps

| Plan step | Verdict | Reason | Proposed change |
| --- | --- | --- | --- |
| O1 split `utils.py`, rename `runtime/` | agree, reorder | Role split is right; 260 of 989 lines are loop algebra a later step can delete. | Do after the unroll spike; split what survives; move `API_FILE` runtime code into `runtime/trace.py` in the same step. |
| O2 line record | agree, enlarge | One record is the core simplification; it also covers regions, constraints and updates, not only lines. | Make it the per-instruction record the runtime consumes; absorb O5. |
| O3 emitter calls | agree, move earlier | It is the step that removes the template layer, yet sits last and is sized L in Q2. | Q1 after O2; generic form for all words shrinks it to M; keep one function per instruction. |
| O4 one word table | partly agree | Merging six tables keeps `PRETTY` and `op_to_template`, which O3 deletes; the types column is S8.3 (D7). | Fold into O3: language columns to the front end, lowering columns to `runtime/emit.py`. |
| O5 dead branch, double visit | agree, merge | The double visit fix is O2 itself (plan's own caveat in `checks.md`). | Delete the dead branch (`oracle_visitor.py:237-241`) now; the visit fix lands with O2. |
| O6 `ValueError`, S8.3 from front end | agree | A spec error raised as `NotImplementedError` misstates the contract (`oracle_visitor.py:254-264`). | Keep; pair with front-end S8.3 export in the same major release. |
| O7 shape rules to front end | agree, widen | Region checks of S7.2 and literal ranges are also static language rules. | Move them too; runtime imports the front-end rule module instead of vendoring a copy. |
| O8 literals rounded once | agree | Double rounding is a silent wrong result (D1). | Keep in Q1; reuse for `epsilon`. |
| O9 `triangular_solve` at `f16` | agree | Widening on CPU deviates from the reference operation (`generic/utils.py:498-512`). | Keep, after measuring. |
| O10 in-place marks | agree, now defined | Ruling 17 fixes the meaning: true aliasing; nothing reads `set_inplace` today (D10). | Implement the alias check and storage reuse of the rulings section; breaking. |
| O11 reject overflowing starts | disagree, widen | Ruling 16 makes every i64 overflow an error, not only starts; symbolic per-step checks over `while` loops are the hard way. | Unroll, then checked i64 arithmetic on concrete values everywhere; `warn_width` becomes an error. |
| missing: S1.3 case audit | add | Ruling 16 turns each row of `/home/user/taidl/docs/overview.md:34-48` into an error or a norm the oracle must enforce. | One row per case in the rulings table below, each with a test. |
| O12 scoped jax config | over-engineered | Scoping x64 relies on an experimental context; the module does not read matmul precision. | Drop the precision flag; transfer buffers without `jax.device_put` so x64 is not needed; spike S. |
| O13 NumPy interpreter | disagree on target | L-sized, and its ground truth is hand-written for 77 words. | StableHLO reference interpreter on the emitted module (M), NumPy only for gaps. |
| O14-O17 docs, coverage, names, logs | agree, move later | Renaming identifiers that O2-O3 delete is churn. | After O3; O16 log lines stay early (S). |
| O18 drop line splitter | agree | `template.py:61-156` re-implements a formatter for generated text. | Keep early; delete `tests/test_templates.py` layout cases with it. |
| missing: unroll every loop | add | Removes about 400 lines of loop algebra, retrace and `while` emission, and D11, D17-D20 by construction. | Spike S: force `loop['unrolled'] = True` (`templates/templates.txt:82`), measure compile time on the examples; land if within an agreed bound. |
| missing: generic form for all words | add | `sh_pretty`, `sh_cast`, `sh_transpose`, `sh_broadcast`, `sh_iota` restate one pattern (`generic/utils.py:477-571`). | One generic emitter plus per-word attribute mapper (`oracle_visitor.py:132-173` already exists). |
| missing: semantic checks to front end | add | `semantic_generator.py:41-56` checks S7.2, a language rule. | Part of widened O7; then `semantic_generator.py` deletes instead of merging. |
| missing: per-generation C++ build | open | `generator.py:72-84` runs CMake at every generation for the debug print handler. | Build `xla-debug/` once per install; generation only copies it. |

## Good parts the plan keeps and must not lose

- Atomic generation into a staging directory (`generator.py:57-87`).
- Every refusal names the instruction and the failing iteration (`generic/utils.py:698-712`).
- Exact integer expressions with floored `/` (`expression_writer.py:20-21`).
- Per-type reduce identities including `f8e4m3fn` (`generic/utils.py:439-451`).
- Non-finite literals by bits (`generic/utils.py:411-420`).
- Byte-identical `api.py` as the gate for every structural step.

## Good parts the plan would lose

- None of substance; the risk is the reverse: the plan keeps too much (loop algebra, `API_FILE` as text, `PRETTY` form) and polishes it before deleting it.
- With unrolling, the `while` loop's small module size goes; that is the one real cost and the spike measures it.

## Revised order

Sizes as the plan defines them (S under a week, M one to three weeks, L more); about four engineers, two quarters.

| # | Step | Size | Quarter | Gate |
| --- | --- | --- | --- | --- |
| 1 | O18 drop splitter; delete dead branch (O5 part) | S | Q1 | `api.py` equal up to line breaks |
| 2 | Spike: unroll every loop, measure compile time | S | Q1 | Agreed bound on examples and attention tests |
| 3 | Move `API_FILE` runtime code to `runtime/trace.py` | S | Q1 | `api.py` equal but its head |
| 4 | Unroll every loop; delete loop algebra, `Unroll`, `while`, `held` (replaces O11) | M | Q1 | Suite green; attention goldens bit-exact |
| 5 | O1 split what remains of `utils.py` by role | S | Q1 | Moves only |
| 6 | O8 literals once; O9 `f16` solve | S each | Q1 | Tie-case tests |
| 6a | Ruling 16 call-time errors: checked i64, overlapping outputs, `warn_width` to error | S | Q1 | One test per S1.3 row |
| 6b | Ruling 17: in-place alias check (O10) | S | Q1 | Alias and non-alias tests |
| 7 | O2 instruction record (absorbs O5 double visit) | M | Q1 | `api.py` byte-identical |
| 8 | O3 with O4: runtime `call`, generic-form emitter, oracle-side lowering table | M | Q2 | Module text diff limited to form; suite green |
| 9 | O6 then widened O7: S7.2, S8.3, compare, ranges, literal shapes to front end | M | Q2 | Same verdict per test spec, old and new |
| 9a | Element norms: guard column in the lowering table, golden tests (after L rules) | M | Q2 | One golden per norm case |
| 10 | StableHLO-interpreter differential runs (replaces O13) | M | Q2 | Every S8.1 word covered or listed as a gap |
| 11 | O12 as backend buffers without jax config; debug build once per install | S | Q2 | Caller jax config untouched |
| 12 | O14-O17 docs, coverage test, names, log lines | S | Q2 | Coverage table checked by test |
