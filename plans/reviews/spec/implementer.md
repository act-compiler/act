# Implementer review: NumPy simulator from the spec alone

Role: an engineer who writes a NumPy functional simulator reading `docs/` and the two examples first, then the front end.
Code paths are under `/home/user/taidl/taidl/`; `inst` is `instruction.py`, `ops` is `operations.py`.
"Unanswered" means neither the spec nor the code says.

## Questions

| Question | Spec section consulted | Answer from code (file:line) or unanswered | Proposed spec addition |
| --- | --- | --- | --- |
| Q1. Which Python attributes of `Accelerator` hold the data models, state, and instructions? | S4.2 only names `generate_oracle(acc)` | `data_model`, `state`, `instructions`, `name`, `base_dir` (accelerator.py:11-15) | S4.2 table of the object model: attribute names and types |
| Q2. Which attributes of a data model give name, dimensions, unit shape, element type? | S4.3 uses `dimensions`, `unit_dim`, `var_type` (call parameters) | `var_name`, `access_dim`, `unit_dim`, `var_dim`, `var_type` (data_model.py:33-50); `var_dim` is the full shape | Same table; note the field names differ from the S10 terms |
| Q3. How is `d0` represented in the accelerator list? | S4.3 | First entry, `DataModel('d0', [], [], 'u8')` (accelerator.py:31) | State that `d0` is `data_model[0]` with empty lists |
| Q4. How do I read an instruction's name and its two attribute lists? | S5.1 | `instruction`, `comp_attr`, and `parameters` for the addressing list (inst:34-35) | Object-model table; flag that `parameters` is the addressing list |
| Q5. How are regions stored: raw or parsed, ints or strings? | S5.2, S6 "Parsed trees" | `instr_inputs` keeps the caller's ints or strings (inst:459-461); `instr_input_asts` holds `[name, start trees, size trees]`, each text parsed via `str()` (inst:131-140) | Say a start or size may be `int` or `str` in raw form and always a tree in the AST form |
| Q6. What node type is a tree: which parse rule's context? | S6 "Parsed trees", S3.2 | `parse_text` returns the first child of the entry rule: `expression`, `condition`, or `update` context (inst:364) | Name the context class returned per place |
| Q7. How do I find the backtick expressions and the shape dimensions in the semantics tree? | S6 "Parsed trees", S7.4 | `instr_expression_asts` maps each `EXPRESSION` token node to its tree (inst:249-252); `dim: INT or EXPRESSION` (IDLV2.g4:23) | Say lookup is by token-node identity and covers shapes, operands, and attribute values |
| Q8. How do I read an operation attribute named by an operation word, such as `slice=`? | S7.5, S3.1 | `name : IDENTIFIER or OPERATION` (IDLV2.g4:28); read `name().getText()` | Add one sentence: attribute names are read by text, never by token type |
| Q9. How do I get the `parameter(k)` number and a `constant` literal from the tree? | S7.5 | `operands().getText()` as `int` (inst:296-297); a literal is any `value` that is not a `%` identifier or is `true`/`false` (ops:199-203) | Give the accessor path, or a worked tree for one line |
| Q10. In what form does a program reach the simulator: a sequence of (instruction, attribute values)? | S1.2, S5.1 | Unanswered; no driver exists in `taidl/`, README.md:6-8 defers to the external act-oracle repo | S4.2 or new S4.6: the abstract issue model (name plus a value per attribute, in any order, by name) |
| Q11. What is the initial content of each data model and of `d0`? | S4.3, S4.4 | Unanswered | State it: zero-filled, or undefined until written; that decides read-before-write |
| Q12. What size is `d0`, and may a region reach past it? | S4.3, S1.3 | Unanswered (size implementation-defined per S1.3); `d0` has no size field (accelerator.py:31) | Say whether a simulator grows `d0` on demand or takes a size; say a region past `d0` is undefined |
| Q13. How do sub-byte types (`s2`, `u2`, `s4`, `u4`) lie in a data model and in `d0` bytes? | S7.3, S8.4 | Unanswered; only the type names exist (data_model.py:4-24) | State packing per `u8` (element order, bit order) or reject them in `d0` casts |
| Q14. Which NumPy dtype backs `bf16`, `f8e4m3fn`, `f8e5m2`, and what are the NaN and infinity rules? | S7.3 | Unanswered; no `numpy` or `ml_dtypes` dependency (pyproject.toml has none) | Reference table: type -> bit layout and special values, or "as XLA-HLO"; name the minimum library |
| Q15. Byte order of multi-byte element types in `d0`? | S1.3, S8.4 | Unanswered (implementation-defined) | Fix little-endian for `d0` so two generators share binaries; keep other layouts optional |
| Q16. Is a state variable stored as Python `int` or `int64`; do updates wrap? | S4.4, S6.2 | Unanswered; overflow is implementation-defined (S6.2); declared initial value is `int` (accelerator.py:55-57) | Recommend exact `int` plus a loud error past 64 bits |
| Q17. Do I evaluate constraints at issue time, and what do I do when one fails? | S5.3, S1.3 | Unanswered; no evaluator exists in `taidl/` | Allow, and recommend, raising an error on a failed constraint (undefined case) |
| Q18. Are placement-mark constraints in `constraints` and `constraint_asts` already? | S5.5 | Yes, appended there (inst:228-231) and also kept in `placed` | Add "a generator reads only `constraints`/`constraint_asts`", with the names |
| Q19. What does ROOT do at run time? | S7.1 | Nothing beyond the front-end checks (inst:276-318) | Say ROOT is syntactic only; outputs come from `%Out<k>` names |
| Q20. Evaluation order across lines: file order? Is a line only used if reachable? | S7.1, S7.5 | Operands must be defined by an earlier line (ops:215-216, 185-188); so file order is valid | Say "execute lines in file order; every line runs" |
| Q21. What if an instruction has inputs or outputs but no semantics? | S5.2, S7.1 | Not rejected: `check_values` runs only when semantics exist (inst:99, 254-257) | S1.3 row or S9 row: regions without semantics are undefined, or rejected |
| Q22. Shape of a region value when `d0` and sizes in bytes; when `dimensions` is empty? | S5.2, S7.2 | Region value shape = sizes + unit shape; `d0` has one size (inst:123); both lists may be empty (accelerator.py:37-41 only checks elements) | One worked example of a scalar data model |
| Q23. Is a computed shape dimension of 0 or less in a line (not a region) defined? | S7.4, S1.3 | Unanswered | Extend the S1.3 row beyond regions |
| Q24. Do operation-attribute expressions (`dimensions={`@c.k`}`) use post-update state? | S5.4 step 3, S7.4 | Parsed like semantics expressions (inst:249-252); order unwritten in code | S5.4 step 3: say "operation attributes" too |
| Q25. What does a negative integer in `dimensions`, `iota_dimension`, `feature_index` mean? | S8.2 ("either sign") | Accepted by form check only (ops:261-265); meaning unanswered | Say negative axes are undefined (StableHLO has none) or count from the end |
| Q26. Which StableHLO and XLA-HLO revision is the reference? | S8.1, S8.4 | Unanswered | Pin a StableHLO spec version and an XLA commit or date |
| Q27. When StableHLO and XLA-HLO disagree (integer divide by zero, shift counts past width, `convert` float to int overflow), which wins? | S8.1, S8.4 | Unanswered; S8.1 only says StableHLO where both exist | Add a list of known disagreements with the chosen answer |
| Q28. Default `type` of `compare` when omitted? | S8.2 | Optional in the table (ops:97); default unanswered | Define it from the operand element type |
| Q29. `dot` accumulation and intermediate precision (bf16 `exponential`, `reduce_add` in bf16)? | S8.3, S8.4 | Unanswered; line type is the result type only | Make it an S1.3 implementation-defined row, with a recommended f32 accumulate |
| Q30. Initial value of each `reduce_<reducer>`? | S8.4 | Unanswered (S8.4 says implementation-defined) | Fix the reducer identity: 0, 1, lowest, highest, all-ones, 0, 0 |
| Q31. `constant` with `true` or `false` for a non-`pred` type; a literal that does not fit the type? | S7.5, S8.3 | Form-only check (ops:199-203) | Define the conversion or mark undefined |
| Q32. Operand order for `clamp`, `select`, `dynamic_slice` indices, `dynamic_update_slice`, `batch_norm_inference`, `triangular_solve`? | S8.2 gives counts only | Counts only (ops:98-99, 120-121, 156); order unanswered | Add an operand-order column, or "as the reference operation orders them" |
| Q33. Does a `pred` operand of `select` or a `compare` result need a scalar form? | S8.3, S8.4 | Unanswered; shapes unchecked | Cite the reference rule once, or list the broadcast cases allowed |
| Q34. Must the simulator trust a line's declared type and shape, or infer them? | S8.4 | Spec: a mismatch is undefined; no inference code exists | Say the simulator may trust the line; a checking simulator may raise |
| Q35. Errors for an unsupported feature: which Python exception? | S1.2 | Unanswered | Name one exception type, such as `NotImplementedError`, and the message form |
| Q36. Do the examples cover state, constraints, updates, marks, multiple outputs, in-place? | examples | Only `softmax` writes its own input (in place); no state, constraints, or marks appear in either example | Ship one example that uses each S5 feature |
| Q37. Is the `mini_trn_double` `copy` after `transpose` an identity that a simulator may skip? | examples | Yes by S8.4 (`copy` is identity); not stated | Add "`copy` returns its operand" to S8.4 |
| Q38. Numeric tolerance a conforming simulator is held to for floats? | S9, S1 | Unanswered; tests dir has front-end tests only (tests/*.py) | Add a conformance note: bit-exact only for integer ops; floats within reference rules |

## What the spec already does well for an implementer

- S5.4 gives a numbered execution order, so state updates, shapes, and semantics have one evaluation point.
- S5.2 states "reads every input region before it writes any output region", which fixes the snapshot rule softmax needs.
- S6.2 gives floor division and divisor-signed remainder with the identity `a == (a / b) * b + a % b`, and Python's `//` and `%` match.
- S1.3 lists every undefined and implementation-defined case in one table with section pointers.
- S7.2 gives the exact index map between a region value and its data model, including the `d0` byte case and the presumed reshape.
- S8.2 gives operand counts, attribute names, forms, and required flags per operation word, matching `OPERATIONS` (operations.py:41-165).
- S8.3 gives element types per word, and S8.5 bounds the language by listing what is out.
- S9 lists the exact rejection cases, so a generator may skip re-validating names, operand counts, and attribute forms.
- S6 "Parsed trees" names the five tree attributes (`constraint_asts`, `update_asts`, `instr_input_asts`, `instr_output_asts`, `instr_expression_asts`) a generator walks.
- S3 copies the ANTLR rules, so the tree shape is predictable.
- S10 is a strict glossary, which keeps generator docs aligned.
- Both examples parse, and every operation they use (`reshape`, `bitcast_convert`, `transpose`, `copy`, `dot`, `exponential`, `reduce_add`, `broadcast`, `divide`) has a row in S8.2.
