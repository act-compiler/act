# Catalog: taidl and act

Scope: `taidl/taidl/*.py` (the front end) and `act/` (umbrella: `act/act`, `act/tests`, `act/docker`, tutorials).
Generated ANTLR code under `taidl/taidl/antlr4/` is excluded.
Class values: `invalid spec` (front end should reject), `unsupported` (unsupported by this generator), `internal bug` (should never fire), `environment/IO`.
Messages are verbatim and shortened; `{x}` marks an interpolated value.

## Front end: names and declarations

| Where | Kind | Message | Trigger | Class |
| --- | --- | --- | --- | --- |
| `taidl/taidl/names.py:10` | ValueError | `{kind} name {name!r} is not an identifier` | A name that is not an ASCII identifier | invalid spec |
| `taidl/taidl/names.py:12` | ValueError | `{kind} name '{name}' is a Python keyword` | A name equal to a Python keyword | invalid spec |
| `taidl/taidl/names.py:14` | ValueError | `{kind} name '{name}' is given twice` | A repeated accelerator, data model, state, instruction or attribute name | invalid spec |
| `taidl/taidl/accelerator.py:41` | ValueError | `{name}: {what} {dims!r} is not a list of positive integers` | `dimensions` or `unit_dim` not a list of integers of at least 1 | invalid spec |
| `taidl/taidl/accelerator.py:43` | ValueError | `{name}: type '{var_type}' is not one of pred, s2, ...` | An element type outside the 19 in `data_model.py:4-24` | invalid spec |
| `taidl/taidl/accelerator.py:50` | ValueError | `state '{name}' follows an instruction: state is declared with the data models, before the instructions` | `add_initial_state` after `add_instruction` | invalid spec |
| `taidl/taidl/accelerator.py:56` | ValueError | `state '{name}': value {value!r} is not an integer` | A non-`int` initial value | invalid spec |

## Front end: instruction regions, marks, attributes

| Where | Kind | Message | Trigger | Class |
| --- | --- | --- | --- | --- |
| `taidl/taidl/instruction.py:94` | ValueError | `{instr}: no data model is named '{name}'` | A region names an undeclared data model | invalid spec |
| `taidl/taidl/instruction.py:119` | ValueError | `{instr}: a region is [buffer, starts, sizes]` | A region that is not a 3-item list | invalid spec |
| `taidl/taidl/instruction.py:126` | ValueError | `{instr}: a region of {buffer} takes starts and sizes of length {count}` | Starts or sizes of the wrong length | invalid spec |
| `taidl/taidl/instruction.py:153` | SyntaxError | `{instr}: semantics has no '%In0' for input region 0` | An input or output region without its value in the semantics | invalid spec |
| `taidl/taidl/instruction.py:158` | SyntaxError | `{instr}: semantics line L, column C: '%In1' has no input region, set_inputs gave N` | A semantics value without its region | invalid spec |
| `taidl/taidl/instruction.py:166` | ValueError | `{instr}: set_inplace needs one flag per input` | Wrong flag count | invalid spec |
| `taidl/taidl/instruction.py:169` | ValueError | `{instr}: set_inplace marks an input also marked by {method}` | One input marked by two marks | invalid spec |
| `taidl/taidl/instruction.py:195` | ValueError | `{name}: {method} needs exactly one output, found {n}` | A placement mark with other than one output | invalid spec |
| `taidl/taidl/instruction.py:197` | ValueError | `{name}: {method} needs one flag per input` | Wrong flag count | invalid spec |
| `taidl/taidl/instruction.py:206` | ValueError | `{name}: {method} marks an input also in place` | A placement mark on an in-place input | invalid spec |
| `taidl/taidl/instruction.py:208` | ValueError | `{name}: {method} marks an input outside {out_buffer}` | A placement mark on an input of another data model | invalid spec |
| `taidl/taidl/instruction.py:226` | ValueError | `{name}: {method} marks an input at the output's own starts` | A mark whose starts are spelled alike in every dimension | invalid spec |
| `taidl/taidl/instruction.py:455` | TypeError | `{instr}: {what} takes a list of strings, not one string` | A bare string for attributes, constraints or updates | invalid spec |

## Front end: semantics and expression texts

| Where | Kind | Message | Trigger | Class |
| --- | --- | --- | --- | --- |
| `taidl/taidl/instruction.py:265` | SyntaxError | `{instr}: semantics line L, column C: {ANTLR message}` | A semantics the grammar cannot parse | invalid spec |
| `taidl/taidl/instruction.py:272` | SyntaxError | `... value name '{x}' has no leading '%'` | A line name without `%` | invalid spec |
| `taidl/taidl/instruction.py:279` | SyntaxError | `{instr}: semantics needs exactly one ROOT, found {n}` | Zero or several ROOT marks | invalid spec |
| `taidl/taidl/instruction.py:286` | SyntaxError | `{instr}: semantics line L, column C: {fault}` | Any fault returned by `check_operations` or `semantics_literal_fault` (rows below) | invalid spec |
| `taidl/taidl/instruction.py:299` | SyntaxError | `... parameter(N) declares '%InN', not '{x}'` | `parameter(N)` on a line not named `%InN` | invalid spec |
| `taidl/taidl/instruction.py:304` | SyntaxError | `... '%InK' is declared by parameter(K), not by {op}` | `%In<k>` defined by another operation | invalid spec |
| `taidl/taidl/instruction.py:315` | SyntaxError | `... ROOT is on '{x}', not on the last output '{y}'` | ROOT not on the highest `%Out<k>` | invalid spec |
| `taidl/taidl/instruction.py:338` | SyntaxError | `{instr}: {place} '{text}' line L, column C: {ANTLR message}` | A syntax error in a start, size, constraint, update or backtick expression | invalid spec |
| `taidl/taidl/instruction.py:358` | SyntaxError | `'@a.x' names no addressing attribute` (also computational attribute, state variable) | A reference to an undeclared name | invalid spec |
| `taidl/taidl/instruction.py:363` | SyntaxError | `'{literal}' is past 64 bits` or `'/' divides by zero` | A literal outside signed 64 bits, or a division whose next sibling is the text `0` (`:406`, `:415`) | invalid spec |
| `taidl/taidl/instruction.py:436` | Fault tuple, raised at `:286` | `'{text}' is past 64 bits` | An integer literal in a dimension or attribute of a semantics line | invalid spec |
| `taidl/taidl/operations.py:186` | Fault tuple, raised at `taidl/taidl/instruction.py:286` | `'{name}' is defined twice` | A value defined by two lines | invalid spec |
| `taidl/taidl/operations.py:196` | Fault tuple | `{word} takes 1 operand, found {n}` | `parameter` or `constant` with other than one operand | invalid spec |
| `taidl/taidl/operations.py:203` | Fault tuple | `{word} takes {form}, found '{x}'` | `parameter` without an input number, `constant` without a literal | invalid spec |
| `taidl/taidl/operations.py:207` | Fault tuple | `{word} takes {n} or more operands, found {m}` | Too few operands for a variadic word | invalid spec |
| `taidl/taidl/operations.py:210` | Fault tuple | `{word} takes {n} operand{s}, found {m}` | Wrong operand count | invalid spec |
| `taidl/taidl/operations.py:214` | Fault tuple | `{word} takes values as operands, found '{x}'` | A literal where a value is required | invalid spec |
| `taidl/taidl/operations.py:216` | Fault tuple | `'{name}' is defined by no earlier line` | A use before definition | invalid spec |
| `taidl/taidl/operations.py:227` | Fault tuple | `{word} has no attribute '{name}': it takes {known}` | An operation attribute the word lacks | invalid spec |
| `taidl/taidl/operations.py:229` | Fault tuple | `{word} is given '{name}' twice` | A repeated operation attribute | invalid spec |
| `taidl/taidl/operations.py:235` | Fault tuple | `'{name}' of {word} takes {form}, found '{x}'` | An attribute value of the wrong form | invalid spec |
| `taidl/taidl/operations.py:238` | Fault tuple | `{word} needs the attribute '{name}'` | A required attribute missing | invalid spec |

## Front end: silent paths (no diagnostic where one may belong)

| Where | Kind | Message | Trigger | Class |
| --- | --- | --- | --- | --- |
| `taidl/taidl/accelerator.py:40-43` | silent fallback | none | `base_dir` falls back to `os.getcwd()` when the caller's file is missing | environment/IO |
| `taidl/taidl/instruction.py:112-129` | no check | none | Region starts and sizes are checked for length only; size 0 or a negative literal passes (F5 in `work/spec/plan/bugs.md`) | invalid spec |
| `taidl/taidl/instruction.py:103-110` | silent replace | none | A second `set_inputs` replaces the regions and drops the placement marks | invalid spec |
| `taidl/taidl/instruction.py:89-90, :210-215` | no check | none | `check_values` runs only after both `set_inputs` and `add_semantics` ran; omitting `set_inputs` skips it (`inputs_given` at `:90`) | invalid spec |
| `taidl/taidl/accelerator.py:41-43` | no check | none | A data model dimension above 2**63-1 passes; the backend rejects it at `act-backend/act_backend/spec_checks.py:302` | invalid spec |
| `taidl/taidl/operations.py:3-4` | no check | none | Shapes and element types of a semantics line are not checked | invalid spec |

## act: package and test scripts

| Where | Kind | Message | Trigger | Class |
| --- | --- | --- | --- | --- |
| `act/act/generators/__init__.py:6` | ImportError (propagates) | none | `import taidl_to` when the oracle is not installed | environment/IO |
| `act/act/generators/__init__.py:13` | ImportError (propagates) | none | `import act_backend` when the backend is not installed | environment/IO |
| `act/tests/check_versions.py:96-99` | TypeError (`re.search(...)[1]` on None) | none | A version pattern missing from `requirements.yml` or `docker/act-env.env` | environment/IO |
| `act/tests/check_versions.py:119` | sys.exit | `{name} is {first} in {place} and {value} in {place}`; `pip check fails` | Version drift across files | environment/IO |

## act: docker and tutorial scripts

| Where | Kind | Message | Trigger | Class |
| --- | --- | --- | --- | --- |
| `act/docker/build.sh:72-75` | exit 1 | `Unknown flag: -{ch}` | An unknown short flag | environment/IO |
| `act/docker/build.sh:82-85` | exit 1 | `Unknown argument: {x}` | An unknown argument | environment/IO |
| `act/docker/build.sh:130` | abort | `--force not confirmed: Aborting.` | No confirmation | environment/IO |
| `act/docker/build.sh:140` | abort | `Rebuild requested but containers reference this image. Re-run with --rebuild --force ...` | Containers use the image | environment/IO |
| `act/docker/run.sh:50` | exit | `Unknown arg: {x}` | An unknown argument | environment/IO |
| `act/docker/run.sh:104-105` | exit 1 | `Image {name} not found locally. Build it first with ./docker/build.sh` | Image absent | environment/IO |
| `act/docker/run.sh:138-143` | abort | `No name provided.`, `Invalid name: {x}. Aborting.` | Empty or malformed container name | environment/IO |
| `act/docker/run.sh:176-182` | abort | `No name provided.`, `Invalid name: {x}. Aborting.`, `Supported Format: ...` | Same, second copy of the prompt | environment/IO |
| `act/docker/run.sh:187` | abort | `Container {name} already exists. Aborting.` | Name taken | environment/IO |
| `act/docker/run.sh:194` | abort | `Invalid choice. Aborting.` | A bad menu answer | environment/IO |
| `act/docker/cleanup.sh:20` | exit | `Unknown arg: {x}` | An unknown argument | environment/IO |
| `act/docker/cleanup.sh:52` | message | `Failed to remove {name}` | Removal failed | environment/IO |
| `act/docker/image.sh:26-27` | exit 1 | `Error: Unsupported architecture: {uname}` | Not x86_64 or arm64 | environment/IO |
| `act/docker/exec.sh:20` | warning | `warning: could not restore the ownership of the files this run wrote` | `chown.sh` failed | environment/IO |
| `act/docker/chown.sh:18` | exit 1 | `Failed to change ownership.` | `chown -R` failed | environment/IO |
| `act/tutorials/asplos26/docker.sh:43` | message on stdout | `Error: On arm64, you must specify either --sim or --compile` | Missing flag on arm64 | environment/IO |
| `act/tutorials/asplos26/solve.sh:8,19` | exit | `Usage: ...`, `Invalid exercise: {x}` | A bad exercise name; the same lines in `micro25` and `pldi26` | environment/IO |
| `taidl/docker/cleanup.sh:35` | exit | `Error: Unsupported architecture: {arch}` | Not x86_64 or arm64 | environment/IO |
| `taidl/docker/run.sh:110` | exit | `Image {name} not found locally. Pull it or build from act/docker/.` | Image absent | environment/IO |
