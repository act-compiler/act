# act-oracle architecture decisions

The decisions in force on the oracle's structure and fidelity, each with its source and evidence.
Panel decisions are cited by question, for example Q5.1; owner decisions by number.

## Lowering target: StableHLO text

- Decision: keep StableHLO text, one thin function per instruction; a `jax.lax` path is rejected (Q5.9).
- Evidence: S8.4 defines each word as its StableHLO operation, and text names it one to one (`oracle_visitor.py:124`, `oracle_visitor.py:330-335`).
- Evidence: `jax.lax` adds its own decompositions and type promotion and loses the module text the goldens pin.

## Execution platform

- Decision: the oracle stays on Python and jax in this plan; a later move to fsim-lib is a study (O29) under owner decision 23.
- Decision: CPU is the normative platform; GPU is held to stated bounds, each deviation a coverage row (Q5.10).

## Loops: unroll every loop

- Decision: unroll every loop and delete the loop algebra (O20), after a spike against a fail point the owner sets first (O19, Q5.1).
- Decision: if the spike fails, keep `while` emission with an equivalence test against unrolling (O11).
- Evidence: unrolling is already the path for any loop around a state update or a varying shape (`templates.txt:101-104`, `templates.txt:137-140`).
- Evidence: every call-time check already enumerates the iterations a value depends on (`utils.py:68-72`).
- Evidence: the symbolic algebra (`utils.py:15-260`) is the source of D11, D17, D19 and D20.

## Static checks: front end, shape module copied

- Decision: static rules run in the front end at generation (O7, Q5.8); the MLIR verifier is never the only check.
- Decision: the generator copies the front end's shape module into each oracle, stamped with its taidl version (owner decision 25, Q3.7).
- Decision: the move is shared item U4; O6's table import is its exit test, and O7 deletes the oracle's copies (Q3.1, Q8.4).
- Evidence: the verifier checks StableHLO's rules, not TAIDL's, and its errors name no instruction.
- Evidence: the generator already copies `generic/` into each oracle (`generator.py:59-64`).

## Order of work

- Decision: fidelity items first, then loop deletion, then the split of `utils.py`, then the record, then emitter calls only if needed (Q5.9).
- Evidence: the byte-identical gate needs a fixed behavioral baseline, and no deleted code should be moved.
- Decision: the C++ debug handler is built once per install and jax version (O28); the `cmake` call takes no ISA input (`generator.py:72-84`).

## Numerics

- Decision: the oracle widens narrow float `dot` and `reduce` to `f32` in the module and rounds once (O27, Q5.2).
- Decision: XLA flags set per compile: no excess precision, no flush to zero, IEEE min and max (O21, Q5.10, Q1.6, Q1.7).
- Decision: every literal is parsed exactly, rounded once to nearest even, emitted as hex bits (O8, Q5.5, Q1.10).
- Evidence: XLA CPU computes `bf16` ops in `f32` in its own passes, so the oracle's value today depends on a pass.

## Differential reference

- Decision: NumPy references over the instruction record with exact literals, plus module-text goldens (O13, Q5.7).
- Decision: the StableHLO interpreter is not added (owner decision 28).
- Evidence: the tests already compare with NumPy references (`tests/test_elementwise.py:7`, `tests/test_reduce.py:82-91`).
- Evidence: an interpreter of the emitted text passes a lowering bug, and D1 and D3 are lowering bugs.

## Run-time violations (owner decision 16)

- Decision: every addressing intermediate is a checked signed 64-bit integer, evaluated as parsed, with one error class (O24, Q5.3).
- Decision: expressions read the pre-state; updates commit with the outputs (O25, Q1.1).
- Decision: a read of an unwritten on-chip element is an error (O26, Q1.4).
- Decision: tensor data never faults; integer divide and remainder at zero, `INT_MIN / -1`, and float-to-integer convert out of range or NaN take fixed values (O23, joint decision of panels 1 and 5).
- Evidence: the backend input is an HLO file with no tensor values, so a data fault would make a correct compile of a legal kernel raise.
- Decision: the oracle checks every constraint of the front end's list, whether implicit, marker, or written, before any load (O10, Q9.3).
- Decision: a failed constraint is the one fault, before any write; a permitted overlap gets the semantics on the pre-issue state (O10, Q9.3, Q9.4).
- Decision: the undeclared same-buffer input-output pair is exact-or-apart; markers are shorthand for their expansions (owner decisions 30 and 31, Q9.2).
- Evidence: `instruction_codegen.py:162-181` runs the constraints first and loads every input before any store; the oracle needs no code per relation.

## Replacement justification

- No module is replaced from scratch.
- Per-word templates are replaced (O3) because each restates an `sh_*` emitter's arguments as unchecked text.
- The `layout` splitter (`template.py:61-156`) is removed (O18), not replaced, because it re-implements a formatter for generated code.
- The loop algebra is removed (O20), not replaced, because unrolled values are plain integers.

## Failure policy (owner decision 40)

- Decision: a generated oracle stops with an error only on invalid assembly code or binary input; the error names the instruction and the clause; otherwise it runs correctly (O31).
- Decision: generation rejects only what the architect can correct, such as a typo, a reserved name or an invalid ISA specification (O6, O7).
- Decision: a path that raises or computes wrong on valid input is a bug with a plan item (D28, D29; D11, D19, D21).
- Evidence: `bugs.md` lists today's paths; O31 audits the remaining raise sites.
- Open: O26 stands only if the language states an unwritten read as a violation (decision 31 allows hardware that reads it).

## Layers and generator metadata (panel 10, owner decisions 38 and 39)

- Decision: the width layer adds implicit range constraints, and the oracle checks them like any constraint (O32, Q10.1, owner decision 30).
- Decision: the oracle ignores the encoding layer and derived attributes, except through a future binary decoder (O33, Q10.2, Q10.3).
- Decision: generator metadata is never read by the oracle unless it is the oracle's own entry; none is defined (O33, Q10.5).
- Evidence: a derived attribute is an ordinary attribute plus a written constraint, so the constraint list carries it (Q10.3).
- Evidence: the neutrality test (O33) is the executable form of owner decision 39.
