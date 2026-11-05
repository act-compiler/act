# Response to the round-1 reviews

Dispositions: accepted, partly, rejected, or pending panel (depends only on a tech-lead ruling, which the panels settle).
Evidence is the review's own `file:line`, re-read where the response rests on it.
`V` is the verification review, `C` the critic, `A` the architect.

## Verification review

| Finding | Disposition | Reason | Changed |
| --- | --- | --- | --- |
| V C1-C5, C7-C13, C15, C17-C25, C27, C29 confirmed | accepted | No change needed beyond vocabulary. | coverage-table.md |
| V C6 incomplete: data model after an instruction accepted | accepted | `accelerator.py:33-44` accepts it. | coverage-table.md (partial, L11) |
| V C14 overclaim: warning narrower; state updates and constraints never warn | accepted | `U:287,659,706` warn; `T:353-363` do not. | coverage-table.md, D23, O24 |
| V C16 shallow: `ROOT` is a front-end rule | accepted | `semantic_generator.py:58-59` only reads it. | coverage-table.md |
| V C28 unverifiable: no XLA comparison | accepted | No differential run exists. | coverage-table.md (partial), O13 |
| V D1-D3, D5-D9, D12, D14, D15, D19, D20 confirmed | accepted | Evidence matches. | bugs.md |
| V D4 GPU part unverifiable | accepted | CPU difference measured; GPU claim stays an assertion until measured. | O9 (measure first) |
| V D10 premise stale | accepted | Owner decision 17 fixes the meaning: storage reuse. | D10, O10 |
| V D11 divergence by reading only | accepted | Stated in the row. | D11 |
| V D13 path unreachable | accepted | `shapes.py:98-103,136-140` reject first. | D13, O5 |
| V D16 "latent" unverifiable | accepted | Wording reduced to the mechanism. | D16 |
| V D17 wrong: enumerates only variables a value reads | accepted | `U:68-72`, product at `U:71`. | D17 |
| V D18 fix not free: ruff check in tests | accepted | `tests/conftest.py:278-282`. | D18, O18 |
| V integer literal above `2^53` rounds twice | accepted | `U:408` calls `float(value)` on an `int`. | D21, O8 |
| V S1.2 status vocabulary has no `supported` | partly | The planning table keeps `supported` rows for review; the shipped table keeps only `partial` and `rejected`. | coverage-table.md, O17 |
| V `bitcast_convert` byte order missing | accepted | Not stated anywhere. | coverage-table.md, O22 |
| V float `convert` overflow missing | accepted | XLA's value today. | coverage-table.md, O22 |
| V initial state outside 64 bits | accepted | `api_generator.py:43-44` stores it exactly. | coverage-table.md, D23, O24 |
| V overlapping output regions | accepted | No check under `taidl_to/`. | coverage-table.md, D22, O24 |
| V `@a.` outside a start or constraint | accepted | `expression_writer.py:100` reads it anywhere; L10 rejects it. | coverage-table.md |
| V constraint arithmetic overflow | accepted | Exact evaluation at `T:353-360` contradicts S6.2. | coverage-table.md, O24 |
| V computed divisor of 0 | accepted | `U:78-80`, `T:359-360`. | coverage-table.md |
| V region value type or count mismatch | accepted | `semantic_generator.py:44-56`. | coverage-table.md, O7 |
| V `constant` operand forms | accepted | `OV:376-380`. | coverage-table.md |
| V compare `type` rule | accepted | `OV:277-283`. | coverage-table.md, O7 |
| V `dot`, `broadcast`, `pad` forms | accepted | Lowering table `OV:19-97`. | coverage-table.md |
| V `add_constraints`, `update=`, `constraints=` | accepted | Read at `IC:32-44`. | coverage-table.md |

## Critic review

| Finding | Disposition | Reason | Changed |
| --- | --- | --- | --- |
| C fidelity rests on XLA where StableHLO is silent | accepted | The StableHLO spec leaves accumulation, order, divide by zero and `convert` range open. | O22, coverage-table.md |
| C fidelity fixes before structure | accepted | They have no structural dependency and set the gate's baseline. | overhaul.md, O8, O21, O19 |
| C wrong premise: tests compare with jax | accepted | `tests/test_elementwise.py:7`, `tests/test_reduce.py:82-91` use NumPy. | decisions.md, O13 |
| C O1, O2, O4, O5, O6, O7, O17, O18 keep | accepted | Kept; O4 folds into O3 on A's evidence. | architecture.md, checks.md |
| C O3 conditional on O2, one function per instruction | accepted | Plan's own debugging argument. | O3 |
| C O8 emit decimal verbatim if MLIR reads it exactly | partly | Pinned jaxlib not checked; hex bits from `Fraction` work regardless. | O8 |
| C O9 decide norm first, CPU is reference | accepted | `README.md:163` holds CPU bit-exact. | O9, O14 |
| C O10 settled by decision 17 | accepted | Owner decision 17. | O10 |
| C O11 fold D19, D20 | accepted | Same code `U:214-252`. | O11 |
| C O12 test whether x64 is needed | accepted | x64 serves only `decorator.py:188`. | O12 |
| C O13 smaller, NumPy from tests, `Fraction` literals | accepted | Existing references in tests. | O13 |
| C O14 keep test-bounds table | accepted | It is the accuracy contract. | O14 |
| C step order | accepted | As above. | overhaul.md |
| C lowering and enumeration rulings keep | partly | Lowering kept; enumeration replaced by unrolling on A's evidence. | decisions.md |
| C `xla_allow_excess_precision` | accepted | No XLA flag set; softmax chains at `bf16` (`mini_trn.py:101-104`). | D24, O21 |
| C numerics norm table owned by L | accepted | Decision 16 needs each case defined. | O22; L needs an item |
| C `dot` relies on XLA upcast; test asserts single rounding | partly | Fact accepted (`tests/test_mini_trn.py:91-92`); widening waits on the norm. | O22 |
| C in-module detection of data-dependent errors | accepted | Checks run before tracing (`U:651-806`). | O23 |
| C normative-platform statement | accepted | No such statement exists. | O14 |
| C version pin as fidelity contract | accepted | `pyproject.toml:18-19` pins jax. | O13 |
| C placement marks over every iteration | rejected | Already done: marks append constraints, checked per iteration (`IC:36`, C12). | none |
| C literal rounding is a spec gap | accepted | S8.4 has no rounding rule. | O8 (depends on L16) |

## Architect review

| Finding | Disposition | Reason | Changed |
| --- | --- | --- | --- |
| A unroll every loop (spike, then adopt) | accepted | `templates.txt:101-104,137-140`; checks already enumerate (`U:68-72`). | O19, O20 |
| A runtime code out of `API_FILE` first | accepted | `templates.txt:26-293`. | O1 |
| A one thin function per instruction, one record | accepted | Matches the debugging argument. | O2, O3 |
| A generic form for all words | accepted | `U:477-571` restate one pattern. | O3 |
| A language columns of the word table to the front end | accepted | `TYPES` equals S8.3 (D7). | O3, O6, O7 |
| A StableHLO interpreter over NumPy | partly | NumPy references exist in tests; the interpreter is a new dependency needing owner approval. | O13 |
| A split `utils.py` after deleting loop algebra | accepted | Avoids moving deleted code. | O1 after O20 |
| A O3 earlier, size M | accepted | Generic form shrinks it. | O3 |
| A O4 fold into O3 | accepted | `PRETTY`, `op_to_template` go with O3. | O4 |
| A O5 merge double visit into O2 | accepted | Plan's own caveat. | O5, O2 |
| A O6 pair with front-end S8.3 export | accepted | Already its dependency. | O6 |
| A O7 widen: region, ranges, compare, literal shapes | accepted | `semantic_generator.py:41-56`, `OV:276-289,373-389`. | O7 |
| A O8, O9 keep | accepted | | O8, O9 |
| A O10 alias check at the call | partly | Storage reuse per decision 17; error only on count or type mismatch; pairing is L5's. | O10 |
| A O11 replaced by unrolling plus checked i64 | accepted | Contingent on O19. | O11, O20, O24 |
| A S1.3 case audit | accepted | Decision 16. | O24, coverage-table.md |
| A O12 over-engineered | accepted | `decorator.py:23` moot given `U:604`. | O12 |
| A O14-O17 later, O16 early | accepted | Avoids renaming deleted code. | overhaul.md |
| A O18 keep early | accepted | | overhaul.md |
| A element norms guarded in the emitter | accepted | | O22 |
| A addressing errors at the call, elements always defined | partly | Element cases may also be errors, detected in the module. | O23 |
| A per-generation C++ build | partly | `generator.py:72-84` confirmed; not on the v3 path, left open. | none |

## Rulings

| Ruling | Disposition | Reason | Changed |
| --- | --- | --- | --- |
| R28, R30, R31, R32, R33, R35, R37 | accepted | Applied on the A and C evidence above, not on the ruling. | as above |
| R38 i64 overflow at the call | accepted | A's S1.3 row and decision 16. | O24 |
| R29 differential reference choice | pending panel | NumPy first on C's evidence; interpreter optional. | O13 |
| R36 keep `kernel` | pending panel | No change planned either way. | none |
| R3, R4, R10, R12 | pending panel | The oracle already behaves so; no change made. | none |
| R5 alias by identical region text, drop `set_inplace` | pending panel | Plan follows decision 17 only. | none |
| R8, R56, R57 numerics and literal norms | pending panel | O22 and O8 enforce whatever norm L states. | none |
| R39, R44, R55, R58, R62, R6 | pending panel | Import form, clause ids, the word "fault", mark removal, item merge, pin. | none |

## Objections for the panels

- R8: a per-word tolerance makes the oracle a band, while the CPU goldens are bit-exact today (`README.md:163`); an oracle-side widening that rounds once would keep it exact.
- R56: "a final value outside 64 bits faults" conflicts with S6.2's "an overflowing constraint does not hold"; one rule must win.
- R62: one M item for O6, O7, L3 and L12 undercounts the widened O7, which alone touches `shapes.py` (543 lines) and three visitor checks.
- R55: "writes no region and no state" needs a commit-after-check run for data-dependent errors (O23), a cost the ruling does not size.
