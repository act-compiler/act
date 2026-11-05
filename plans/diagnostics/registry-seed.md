# Registry seed

First codes from the current catalog.
`Now` is the file and line that raises today; `Spec` is the spec-draft section id (S9 rows by their page headings).
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
| RF013 | A mark has one flag per input | S5.5 | `instruction.py:166, :197` |
| RF014 | An input is marked once | S5.5 | `instruction.py:169, :206` |
| RF015 | A placement mark needs exactly one output | S5.5 | `instruction.py:195` |
| RF016 | A placement mark stays inside the output's data model | S5.5 | `instruction.py:208` |
| RF017 | `set_apart` input starts differ from the output's | S5.5 | `instruction.py:226` |
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
| UB015 | An overwritten operand with another reader | `inplace_fix.rs:38, :71` |
| UB016 | A kernel constant the allocator cannot write | `emitter.cc:30, :76, :104, :130, :284` |
| UB017 | A tensor larger than its data model | `parser.cc:152, :162, :226-232, :295` |
| UB018 | No placement within the solver's time limit | `act_malloc.cc:179-182` |
| UO001 | A data model, instruction, attribute or value name the oracle reserves | `generator.py:41`, `api_generator.py:59`, `instruction_codegen.py:202`, `semantic_generator.py:29` |
| UO002 | A backtick expression in a scalar operation attribute | `oracle_visitor.py:416` |
| UO003 | A dtype outside the oracle's map | `decorator.py:92, :96` |

## V codes (run-time violations)

| Code | Rule | Spec | Now |
| --- | --- | --- | --- |
| VO001 | A constraint holds at the call | S5.3 | `generic/utils.py:796-804` |
| VO002 | A region stays inside its data model | S5.2 | `generic/utils.py:709` |
| VO003 | A size is at least 1 | S5.2 | `generic/utils.py:662` |
| VO004 | A divisor is not 0 | S6.2 | `generic/utils.py:80`, `templates/templates.txt:319-320` |
| VO005 | An integer stays inside the signed 64-bit range | S6.2 | `generic/utils.py:268-271` (a warning today) |
| VO006 | A region value has its region's shape | S7.2 | `generic/utils.py:743, :783` |
| VO007 | Output regions do not overlap | S5.2 | none today |
| VB001 | A value computed from attributes is decidable at generation | S6.2 | `spec_checks.py:218, :252, :294` |

## K, I, X ranges

| Range | Contents | Now |
| --- | --- | --- |
| KB001 to KB040 | Kernel text: header, line split, types, shapes, operands | the 24 kernel-reader rows of `catalog/act-backend.md` |
| KO001 to KO020 | Kernel declaration and call arguments | `generic/decorator.py:71-185`, `generic/utils.py:278-281, :675-692` |
| IB001 to IB099, IO001 to IO019, IF001 to IF019 | Internal rows | the 34 backend, 3 oracle internal rows |
| XB001 to XB040, XO001 to XO009, XA001 to XA059 | Flags, files, builds, scripts | `pipeline.rs:295-427`, `generator.py:104-193`, `act/docker/*.sh` |
