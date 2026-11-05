# act-oracle architecture decisions

Three positions were argued on the oracle's structure: an incremental simplifier, an overhaul architect, and an outside critic who judges both as a reference model for an ISA.
Each decision below states the position taken and the evidence; rulings still before the panels are listed in the review response, not here.

## Positions

- Simplifier: delete dead code, merge tables, split `generic/utils.py` by role, gate every step on a byte-identical generated `api.py`.
- Architect: unroll every loop, move fixed runtime code out of templates, one generic-form emitter, static checks in the front end.
- Critic: a reference model computes exactly the spec's meaning; where StableHLO leaves a result open, the oracle needs a stated norm, not XLA's choice.

## Lowering target: StableHLO text

- Decision: keep StableHLO text.
- Evidence: S8.4 defines each word as its StableHLO operation, and text names it one to one (`oracle_visitor.py:124`, `oracle_visitor.py:330-335`).
- Evidence: `jax.lax` adds its own decompositions and type promotion and lacks a `batch_norm_inference` primitive.
- A `jax.lax` path is reconsidered only if it emits the same reference operation for every word, measured by diffing module text.

## Loops: unroll or enumerate symbolically

- Decision: spike unrolling (O19); adopt it (O20) if compile time stays within a bound the owner fixes first.
- Evidence: unrolling is already the path for any loop around a state update or a varying shape (`templates.txt:101-104`, `templates.txt:137-140`).
- Evidence: every call-time check already enumerates the iterations a value depends on (`utils.py:68-72`), so tracing per iteration adds no asymptotic cost before XLA.
- Evidence: the symbolic algebra (`utils.py:15-260`) is the source of D11, D17, D19 and D20.

## Static checks: oracle, MLIR verifier, or front end

- Decision: static rules move to the front end (O7); the oracle keeps call-time evaluation; the MLIR verifier is never the only check.
- Evidence: the verifier checks StableHLO's rules, not TAIDL's, and its errors name no instruction; the front end skips these checks today (`taidl/docs/conformance.md:59`).

## Templates and generated code

- Decision: fixed runtime code leaves `API_FILE` first (O1); per-word templates become emitter calls (O3); one function per instruction stays in `api.py`.
- Evidence: `API_FILE` holds about 260 lines of fixed code (`templates.txt:26-293`) that ruff and tests do not see.
- Evidence: when an oracle disagrees with hardware, the per-instruction function is what a reader debugs.

## Fidelity before structure

- Decision: O8 and O21 land before the unroll spike and the structural steps.
- Evidence: they are S items with no structural dependency, and the byte-identical gate needs a fixed behavioral baseline.

## Differential reference

- Decision: factor the tests' NumPy references into one module (O13); the StableHLO reference interpreter is an option that needs owner approval as a new dependency.
- Evidence: the tests already compare with NumPy references (`tests/test_elementwise.py:7`, `tests/test_reduce.py:82-91`); jax appears only for a `bf16` bitcast (`tests/conftest.py:274`).
- A reference cannot settle a norm, since it makes its own choice; each norm row gets a golden test (O22).

## Run-time violations (owner decision 16)

- Decision: an addressing value that breaks a rule raises at the call (O24); an element value that StableHLO leaves open gets a norm the emitter enforces (O22), or an error flag computed in the module (O23).
- Evidence: addressing values are concrete at trace time, while element values exist only in the compiled module (checks before tracing, `utils.py:651-806`).

## Replacement justification

- No module is replaced from scratch.
- Per-word templates are replaced (O3) because each restates an `sh_*` emitter's arguments as unchecked text.
- The `layout` splitter (`template.py:61-156`) is removed (O18), not replaced, because it re-implements a formatter for generated code.
- The loop algebra is removed (O20), not replaced, because unrolled values are plain integers.
