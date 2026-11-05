# Registry seed

First codes from the current catalog.
`Now` is the file and line that raises today; `Spec` is the TAIDL language specification section id (S9 rows by their page headings); the id ends line 1 of the message.
`new` marks a code whose rule is not rejected by the reference front end today (S9 "Not rejected").

## R codes (front end)

| Code | Rule | Spec | Now |
| --- | --- | --- | --- |
| RF001 | A name is an ASCII identifier | S4.1, S9 Declarations | `taidl/taidl/names.py:10` |
| RF002 | A name is not a Python keyword | S4.1 | `names.py:12` |
| RF003 | A name is given once | S4.1 | `names.py:14` |
| RF004 | An instruction attribute is named once across both lists | S5.1 | `names.py:14` via `instruction.py:67` |
| RF005 | Addressed dimensions and unit shape are positive integers | S4.3 | `accelerator.py:41` |
| RF006 | An element type is one of S7.3 | S4.3, S7.3 | `accelerator.py:43` |
| RF007 | State is declared before instructions | S4.5 | `accelerator.py:50` |
| RF008 | An initial value is an `int` | S4.4 | `accelerator.py:56` |
| RF009 | A list of strings, not one string | S5.1 | `instruction.py:455` |
| RF010 | A region is `[data model, starts, sizes]` | S5.2 | `instruction.py:119` |
| RF011 | A region names a declared data model | S5.2 | `instruction.py:94` |
| RF012 | Starts and sizes count equals the addressed dimensions | S5.2 | `instruction.py:126` |
| RF013 | A marker list has one flag per input; a placement mark needs one output | S5.5 | `instruction.py:163-172, :190-208` |
| RF014 | A tie pairs an input and an output of one buffer (new) | S5.5 | none today |
| RF015 | A tie has equal size text on input and output (new) | S5.5 | none today |
| RF016 | `set_apart` is not given on identical input and output text (new) | S5.5 | none today |
| RF017 | Identical input and output text needs `set_inplace` (new) | S5.5 | none today |
| RF018 | A text parses by the grammar | S3 | `instruction.py:265, :338` |
| RF019 | A reference names a declared name | S6.1 | `instruction.py:358` |
| RF020 | A literal is inside the signed 64-bit range | S2, S6.2 | `instruction.py:363, :436` |
| RF021 | A literal divisor is not 0 | S6.2 | `instruction.py:363` |
| RF022 | A value name starts with `%` | S7.1 | `instruction.py:272` |
| RF023 | Exactly one `ROOT` | S7.1 | `instruction.py:279` |
| RF024 | `ROOT` is on the last output | S7.1 | `instruction.py:315` |
| RF025 | An operand count or kind matches the row | S8.2 | `operations.py:196-214` |
| RF026 | An operation attribute exists | S8.2 | `operations.py:227` |
| RF027 | An operation attribute is given once | S8.2 | `operations.py:229` |
| RF028 | An operation attribute has its form | S8.2 | `operations.py:235` |
| RF029 | A required operation attribute is given | S8.2 | `operations.py:238` |
| RF030 | An operand is defined by an earlier line | S7.1 | `operations.py:216` |
| RF031 | A value is defined once | S7.1 | `operations.py:186` |
| RF032 | `parameter(k)` defines `%In<k>` | S7.2 | `instruction.py:299` |
| RF033 | `%In<k>` is defined only by `parameter(k)` | S7.2 | `instruction.py:304` |
| RF034 | A region has its value and a value its region | S7.2 | `instruction.py:153, :158` |
| RF040 | An instruction with output regions has semantics (new) | S5.2, S7.2 | `semantic_generator.py:16`; `generator.py:119-121` |
| RF041 | A description is complete: regions given before semantics is checked (new) | S7.2 | `instruction.py:90, :210-215`; `spec_checks.py:195` |
| RF042 | A region value has its data model's element type (new) | S7.2 | `semantic_generator.py:45` |
| RF043 | A region value has the element count of its region, for literal sizes (new) | S7.2 | `spec_checks.py:415`, `semantic_generator.py:53` |
| RF044 | A `constant` literal fits its element type (new) | S8.4 | `spec_checks.py:75`, `oracle_visitor.py:387` |
| RF045 | A folded divisor is not 0 (new) | S6.2 | `spec_checks.py:210, :292` |
| RF046 | A data model dimension and an initial value are inside the signed 64-bit range (new) | S4.3, S4.4 | `spec_checks.py:302` |
| RF047 | A literal size is at least 1 (new) | S5.2 | `generic/utils.py:662` |
| RF048 | An addressing attribute appears only in a start or a constraint (new) | S6.1 | `spec_checks.py:203, :287` |
| RF049 | Shapes and element types agree per operation (new) | S7.4, S8.4 | `generic/shapes.py` |
| RF050 | An operation is used at an element type of its S8.3 row (new) | S8.3 | `oracle_visitor.py:255` |
| RF051 | A v2 construct is removed; the message names its replacement (new) | the section of the replacement | the removed-names table (D15) |

## U codes (generators)

| Code | Feature | Now |
| --- | --- | --- |
| UB001 | An instruction with other than one output region | `spec_checks.py:26` |
| UB002 | An element type outside the backend's 13 | `spec_checks.py:48`, `ir/dtype.rs:145` |
| UB003 | An operation word outside the backend's 18 | `spec_checks.py:58`, `expression.rs:463` |
| UB004 | An operand count the selector's rewrite does not take | `spec_checks.py:66`, `nodes.rs:398`, `expression.rs:432` |
| UB005 | A `slice` stride other than 1 | `spec_checks.py:85`, `nodes.rs:345` |
| UB006 | A value no later line reads, or `ROOT` above the last line | `spec_checks.py:188, :192` |
| UB007 | An addressing attribute in a semantics or a size | `spec_checks.py:203, :287` |
| UB008 | A computational attribute the selector cannot read from a shape | `spec_checks.py:230, :237` |
| UB009 | State variables in a start, a constraint, a semantics or an update | `spec_checks.py:250, :364, :375, :382` |
| UB010 | A constraint the allocator cannot hold | `spec_checks.py:364` |
| UB011 | A two-level tile in a layout | `line.rs:60` |
| UB012 | A `reduce` other than a sum | `expression.rs:118` |
| UB013 | A kernel root that is a parameter or a constant | `lib.rs:154, :186` |
| UB014 | A tile that does not divide its shape | `shapes.rs:86, :115, :147` |
| UB015 | A tied operand that is HBM-pinned or a shared view | `inplace_fix.rs:6-17, :59-80` |
| UB016 | A kernel constant the allocator cannot write | `emitter.cc:30, :76, :104, :130, :284` |
| UB017 | A tensor larger than its data model | `parser.cc:152, :162, :226-232, :295` |
| UB018 | No placement within the solver's time limit | `act_malloc.cc:179-182` |
| UO001 | A data model, instruction, attribute or value name the oracle reserves | `generator.py:41`, `api_generator.py:59`, `instruction_codegen.py:202`, `semantic_generator.py:29` |
| UO002 | A backtick expression in a scalar operation attribute | `oracle_visitor.py:416` |
| UO003 | A dtype outside the oracle's map | `decorator.py:92, :96` |

## V codes (faults, scope F)

| Code | Rule | Spec | Now |
| --- | --- | --- | --- |
| VF001 | A constraint holds at the call: implicit, marker, or written, including output-output apart and each pair's relation | S5.3, S1.3.5 | `generic/utils.py:796-804` |
| VF002 | A region stays inside its data model | S5.2 | `generic/utils.py:709` |
| VF003 | A size is at least 1 | S5.2 | `generic/utils.py:662` |
| VF004 | A divisor is not 0 | S6.2 | `generic/utils.py:80`, `templates/templates.txt:319-320` |
| VF005 | An expression operator yields a signed 64-bit integer; also raised at generation for a value the backend cannot hold | S6.2 | `generic/utils.py:268-271` (a warning today); `spec_checks.py:218, :252, :294` |
| VF006 | A region value has its region's shape | S7.2 | `generic/utils.py:743, :783` |
| VF007 | Unused: overlap is a constraint, reported as `VF001` | S1.3.5 | none |
| VF008 | An on-chip element is read only after an earlier issue wrote it (new) | S1.3 | none today |
| VF009 | A backtick constant is inside its element type (new) | S8.4 | none today |
| VF010 | A declared and a computed dim agree at selection (new) | S7.4 | `egraph.rs:403-409` (a panic today) |

- Every V row obeys the no-write rule: a fault writes no region and no state.
- `VF005` also covers the backend's generation-time check of a computed integer.
- Input-output overlap and marker violations are `VF001`; the rejections at declaration are `RF014` to `RF017`.
- No V code is raised by tensor data: divide, remainder, and convert take fixed values.

## K, I, X ranges

| Range | Contents | Now |
| --- | --- | --- |
| KB001 to KB040 | Kernel text: header, line split, types, shapes, operands | the 24 kernel-reader rows of `catalog/act-backend.md` |
| KO001 to KO020 | Kernel declaration and call arguments | `generic/decorator.py:71-185`, `generic/utils.py:278-281, :675-692` |
| KB041 | A valid kernel no candidate implements; lists the drop reasons | `pipeline.rs:59-60` |
| KO021 to KO040 | Assembly code and binary input of an oracle, including a word matching no instruction | none today |
| IB001 to IB099, IO001 to IO019, IF001 to IF019 | Internal rows | the 34 backend, 3 oracle internal rows |
| XB001 to XB040, XO001 to XO009, XA001 to XA059 | Flags, files, builds, scripts | `pipeline.rs:295-427`, `generator.py:104-193`, `act/docker/*.sh` |

## Layer rows

| Code | Rule | Spec |
| --- | --- | --- |
| RF052 | A type is `s<w>` for w in 1..64 or `u<w>` for w in 1..63; `u64` is rejected | S5.1 |
| RF053 | A type key names a declared attribute | S5.1 |
| RF054 | Every word bit is covered by exactly one slice or fixed bit | S5.1 |
| RF055 | No word matches the fixed bits of two instructions | S5.1 |
| RF056 | An encoded attribute has a declared type whose width equals its slice total | S5.1 |
| RF057 | A derived attribute has exactly one affine link to one encoded attribute; the message names the attribute and the defect | S5.1 |

- A value outside a type is `VF001` (S1.3.10), not a new code.
