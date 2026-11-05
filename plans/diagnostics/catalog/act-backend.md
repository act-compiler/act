# Catalog: act-backend

Scope: Python `act-backend/act_backend/*.py`, Rust `act-backend/act_backend/generic/src`, C++ `act-backend/act_backend/generic/cpp`.
Class values: `invalid spec` (front end should reject), `unsupported` (unsupported by this generator), `internal bug` (should never fire), `environment/IO`.
A rule on the input kernel (HLO text) is classed `unsupported` when the kernel is valid XLA-HLO the backend cannot compile, and `environment/IO` when the kernel text is malformed.
Paths are relative to `act-backend/act_backend/` unless they start with `generic/`; Rust paths are under `generic/src/`, C++ under `generic/cpp/malloc/`.

## Python: generation entry (`generator.py`, `template_loader.py`)

| Where | Kind | Message | Trigger | Class |
| --- | --- | --- | --- | --- |
| `generator.py:104` | ValueError | `Cost library {cost} is not an object file (.o), a static library (.a) or a shared library (.so)` | `cost=` with another suffix | environment/IO |
| `generator.py:110` | ValueError | `Cost library {cost!r} has a line break in its path` | Newline in the path | environment/IO |
| `generator.py:113` | ValueError | `Cost library {cost} is a shared library under a directory with a colon` | Colon in a `.so` directory | environment/IO |
| `generator.py:117` | FileNotFoundError | `Cost library {cost} does not exist` | Missing file | environment/IO |
| `generator.py:193` | RuntimeError | `Backend build failed in {dir}:\n{cargo output}` | `cargo build --release` fails | environment/IO |
| `template_loader.py:77` | ValueError | `{template_file}: no value for the key {keys}` | A template key without a value | internal bug |
| `template_loader.py:80` | ValueError | `{template_file}: no key for the value {keys}` | A value without a template key | internal bug |
| `template_loader.py:135` | ValueError | `{source}: no value for the placeholder {names}` | Same, for inline placeholders | internal bug |
| `template_loader.py:138` | ValueError | `{source}: no placeholder for the value {names}` | Same | internal bug |

## Python: `spec_checks.py` (run in order at `generator.py:125-131`)

| Where | Kind | Message | Trigger | Class |
| --- | --- | --- | --- | --- |
| `spec_checks.py:26` | ValueError | `{instr}: the instruction has {n} outputs, and the allocator places one` | Other than one output region | unsupported |
| `spec_checks.py:48` | ValueError | `{name}: its semantics uses the element type '{t}', which the selector does not support` | A line typed outside the 13 of `isel_ir2isa_rewrites_rs_generator.py:24-38` (no `s2 s4 u2 u4 f8e4m3fn f8e5m2`) | unsupported |
| `spec_checks.py:58` | ValueError | `{name}: its semantics uses the operation '{word}', which the selector does not support` | A word outside the 18 of `isel_ir2isa_rewrites_txt_generator.py:11-30` | unsupported |
| `spec_checks.py:66` | ValueError | `{name}: its semantics gives the operation '{word}' {n} values, where the selector takes {k}` | For example `concatenate` of three values | unsupported |
| `spec_checks.py:75` | ValueError | `{name}: its semantics gives the constant '{v}' the literal '{x}', which is not one {type} value` | A literal outside the element type's range | invalid spec |
| `spec_checks.py:85` | ValueError | `{name}: its semantics slices '[a:b:s]' with the stride {s}, where the selector takes a stride of 1` | A slice stride other than 1 | unsupported |
| `spec_checks.py:188` | ValueError | `{name}: its semantics defines '{v}', which no line below reads` | A dead value before the last line | unsupported |
| `spec_checks.py:192` | ValueError | `{name}: its semantics marks '{v}' ROOT above its last line` | ROOT not on the last line | unsupported |
| `spec_checks.py:195` | ValueError | `{name}: its semantics reads {n} inputs by parameter, and the instruction has {m}` | Reachable when `set_inputs` was never called (`taidl/taidl/instruction.py:90`) | invalid spec |
| `spec_checks.py:203` | ValueError | `{name}: its semantics reads the address attribute '{x}', which is not known when an instruction is selected` | `@a.` inside a backtick expression | unsupported |
| `spec_checks.py:210` | ValueError | `{name}: its semantics divides by zero in '{expr}'` | A constant-zero divisor after folding integers | invalid spec |
| `spec_checks.py:218` | ValueError | `{name}: its semantics has the dimension '{expr}', which 64 bits cannot hold` | A computed dimension outside signed 64 bits | invalid spec |
| `spec_checks.py:230` | ValueError | `{name}: no shape of its semantics names the computational attribute '{a}', and no operation's attribute holds it alone or times a positive integer` | A `@c.` the selector cannot read back | unsupported |
| `spec_checks.py:237` | ValueError | `{name}: no dimension of its semantics holds the computational attribute '{a}' alone or times a positive integer` | Same | unsupported |
| `spec_checks.py:287` | ValueError | `{what} has the size '{text}', which is not integers and computational attributes joined by '*', '/', '%', '+' and '-'` | A size with `@a.` or `@s.` | unsupported |
| `spec_checks.py:292` | ValueError | `{what} has the size '{text}', which divides by zero` | A zero divisor in a size | invalid spec |
| `spec_checks.py:294` | ValueError | `{what} has the size '{text}', which 64 bits cannot hold` | A size outside signed 64 bits | invalid spec |
| `spec_checks.py:302` | ValueError | `{model}: the data model has the dimension '{d}', which 64 bits cannot hold` | A dimension above 2**63-1 | invalid spec |
| `spec_checks.py:321` (reason `:250`) | ValueError | `{what} has the address '{text}', which reads the state variable '{s}', which the backend does not model` | `@s.` in a start | unsupported |
| `spec_checks.py:321` (reason `:252`) | ValueError | `... which 64 bits cannot hold` | A start outside signed 64 bits | invalid spec |
| `spec_checks.py:321` (reason `:255`) | ValueError | `... which divides by zero` or `takes a remainder by zero` | A zero divisor in a start | invalid spec |
| `spec_checks.py:364` | ValueError | `{instr}: {mark gives it / it declares} the constraint '{text}', which the backend does not support` | A constraint with `@s.`, a divisor over an address, or a 64-bit overflow (`honoured`, `:331`) | unsupported |
| `spec_checks.py:375` | ValueError | `{name}: it updates machine state by '{u}', which the backend does not model` | Any state update | unsupported |
| `spec_checks.py:382` | ValueError | `{name}: its semantics reads the state variable '{s}', which the backend does not model` | `@s.` in a semantics | unsupported |
| `spec_checks.py:415` | ValueError | `{name}: '{v}' is {type}, more elements than the {room} of its {where} region in {buffer}` | A value larger than its region (literals only) | invalid spec |

## Python: silent paths

| Where | Kind | Message | Trigger | Class |
| --- | --- | --- | --- | --- |
| `generator.py:119-121` | silent skip | none | An instruction with no semantics is left out of the backend | invalid spec |
| `spec_checks.py:394-415` | no check | none | A value's element type is never compared with its data model's `var_type` | invalid spec |
| `spec_checks.py:46-52` | no check | none | A data model of an unsupported element type passes when no semantics line uses it | unsupported |
| `generator.py:141-173` | print | `Copied generic backend structure to ...`, `Generated *.rs` ... | Progress on stdout, no warnings channel | environment/IO |

## Rust: command line and search (`pipeline.rs`, `build.rs`)

| Where | Kind | Message | Trigger | Class |
| --- | --- | --- | --- | --- |
| `generic/build.rs:55-74` | expect, panic | `failed to remove cpp/malloc/build`, `cmake failed`, `make failed` | The C++ allocator does not build | environment/IO |
| `pipeline.rs:295-296` | eprintln, exit 1 | `Error: missing the required flag {flag}` | No `--input` or `--output` | environment/IO |
| `pipeline.rs:303-304` | eprintln, exit 1 | `Error: missing file name after --input` | Flag last on the line | environment/IO |
| `pipeline.rs:308` | expect | `Failed to get current directory` | Unreadable cwd | environment/IO |
| `pipeline.rs:312-316` | eprintln, exit 1 | `Error: input file '{p}' is not an .hlo file` | Wrong suffix | environment/IO |
| `pipeline.rs:320-321` | eprintln, exit 1 | `Error: input file '{p}' does not exist` | Missing file | environment/IO |
| `pipeline.rs:329-330` | eprintln, exit 1 | `Error: missing file name after --output` | Flag last on the line | environment/IO |
| `pipeline.rs:336-340` | eprintln, exit 1 | `Error: output file '{p}' does not have a .py extension` | Wrong suffix | environment/IO |
| `pipeline.rs:360-361` | eprintln, exit 1 | `Error: missing directory after --log` | Flag last on the line | environment/IO |
| `pipeline.rs:366-370` | eprintln, exit 1 | `Error: the log directory '{p}' is not a directory` | A file at `--log` | environment/IO |
| `pipeline.rs:373-378` | eprintln, exit 1 | `Error: cannot create the log directory '{p}': {e}` | `create_dir_all` fails | environment/IO |
| `pipeline.rs:385-391` | eprintln, exit 1 | `Error: cannot create the output's directory '{p}': {e}` | `create_dir_all` fails | environment/IO |
| `pipeline.rs:405-414` | eprintln, exit 1 | `Error: cannot write the output '{p}': {e}` | The output cannot be opened | environment/IO |
| `pipeline.rs:421-427` | eprintln, exit 1 | `Error: cannot create the log directory '{p}': {e}` | `mkdtemp` fails | environment/IO |
| `pipeline.rs:56, :100, :198` | expect | `Failed to remove existing output file`, `Failed to copy best ASM to output path` | Filesystem failure | environment/IO |
| `pipeline.rs:59-60` | eprintln, exit 1 | `Error: no assembly found: {reason}` | Every candidate dropped or the time limit ran out (reason from `:208-218`) | unsupported |
| `pipeline.rs:96-102` | eprintln, exit 1 | `Error: the cost model {reason} for {asm}` (`returned {n}`, `gave the non-finite cost {c}`) | The linked `act_cost` fails (`cost/c.rs:17,21`) | environment/IO |
| `pipeline.rs:265-266` | println | `PII #{n} dropped: {reason}` | A candidate fails in the allocator; printed on stdout, not stderr | unsupported |
| `pipeline.rs:551-552` | eprintln, exit 1 | `Error: stopped for an unexpected reason` | `StopReason::Other` from the e-graph runner | internal bug |
| `pipeline.rs:558-563` | eprintln, exit 1 | `Error: timeout() called at {t} but overall time limit of {l} not reached` | Search ended early | internal bug |
| `pipeline.rs:344-349` | println | `Warning: Output file '{p}' already exists and will be overwritten.` | Output exists; printed on stdout, not stderr | environment/IO |

## Rust: kernel reader (input HLO text)

| Where | Kind | Message | Trigger | Class |
| --- | --- | --- | --- | --- |
| `isel/initializer/lib.rs:46` | Err(String) | `cannot read the kernel {path}: {e}` | Unreadable file | environment/IO |
| `isel/initializer/parser.rs:246` | Err | `the kernel is empty` | No lines | environment/IO |
| `isel/initializer/parser.rs:22` | Err | `the kernel's first line '{x}' is not an HloModule line with an entry_computation_layout` | Bad header | environment/IO |
| `isel/initializer/parser.rs:316` | Err | `the kernel's ENTRY marks {n} lines ROOT, where the reader takes one` | Several ROOT lines | environment/IO |
| `isel/initializer/line.rs:32` | Err | `the type '{x}' is not an element type with a shape` | Bad type text | environment/IO |
| `isel/initializer/line.rs:60` | Err | `the layout {..} of the shape [..] gives a second tile, which the backend does not support` | Two-level tiling | unsupported |
| `isel/initializer/line.rs:96` | Err | `the reader cannot split the kernel's line '{x}' into a name, a type and an operation` | Malformed line | environment/IO |
| `isel/initializer/lib.rs:85` | Err | `the kernel's ENTRY computes no value` | Empty ENTRY | environment/IO |
| `isel/initializer/lib.rs:119` | Err | `the kernel's inputs hold more bytes than 64 bits can count` | Huge shapes | unsupported |
| `isel/initializer/lib.rs:154` | Err | `the kernel's root {r} is a parameter, which no instruction computes` | Root is a parameter | unsupported |
| `isel/initializer/lib.rs:162` | Err | `{x} reads parameter {n}, which the kernel's header does not give` | Parameter number out of range | environment/IO |
| `isel/initializer/lib.rs:168` | Err | `{x} reads parameter {n}, which an earlier line reads` | A parameter read twice | unsupported |
| `isel/initializer/lib.rs:186` | Err | `the kernel's root {r} is a constant, which no instruction computes` | Root is a constant | unsupported |
| `isel/initializer/nodes.rs:39` | Err | `the kernel reads {s}, which no earlier line of it defines` | Use before definition | environment/IO |
| `isel/initializer/nodes.rs:56` | Err | `{x} gives the element type {t} to {y}, which keeps the element type {u} of {z}` | Type changes across a type-keeping word | environment/IO |
| `isel/initializer/nodes.rs:319` | Err | `{x} slices a value of rank {r} with bounds for rank {k}` | Slice rank mismatch | environment/IO |
| `isel/initializer/nodes.rs:330` | Err | `{x} slices with '{t}', which is not a [start:limit:stride] range` | Bad range text | environment/IO |
| `isel/initializer/nodes.rs:345` | Err | `{x} slices with the stride {s}, where the backend supports only a stride of one` | Stride not 1 | unsupported |
| `isel/initializer/nodes.rs:356` | Err | `{x} slices {a} of a dimension of {n}, where 0 <= start < limit <= {n}` | Range out of bounds | environment/IO |
| `isel/initializer/nodes.rs:366` | Err | `{x} gives the shape [..] to a slice of the shape [..]` | Declared shape differs | environment/IO |
| `isel/initializer/nodes.rs:398` | Err | `{x} concatenates {n} {values}, where a concatenate takes 2` | Not two operands | unsupported |
| `isel/initializer/nodes.rs:413` | Err | `{x} concatenates values of rank {r} along dimension {d}` | Dimension out of rank | environment/IO |
| `isel/initializer/nodes.rs:431` | Err | `{x} concatenates [..] and [..] along dimension {d}, which does not give its shape [..]` | Shape mismatch | environment/IO |
| `isel/initializer/shapes.rs:24` | Err | `the shape [{s}] holds '{x}', which is not a dimension` | Bad shape text | environment/IO |
| `isel/initializer/shapes.rs:68` | Err | `the shape [{s}] holds more bytes than 64 bits can count` | Huge shape | unsupported |
| `isel/initializer/shapes.rs:86, :147` | Err | `the tile ({t}) of the shape [{s}] does not give one size per dimension` | Tile rank | unsupported |
| `isel/initializer/shapes.rs:115` | Err | `the tile ({t}) does not divide the shape [{s}]` | Non-dividing tile | unsupported |
| `isel/initializer/shapes.rs:127` | Err | `the tile ({t}) of the shape [{s}] holds '{x}', which is not a size` | Bad tile text | environment/IO |
| `isel/initializer/expression.rs:106` | Err | `a reduce applies the region {r}, which the kernel does not define` | Undefined region | environment/IO |
| `isel/initializer/expression.rs:118` | Err | `the region {r} computes something other than a sum, the only reduce the backend supports` | Reducer not add | unsupported |
| `isel/initializer/expression.rs:169` | Err | `{x} gives {op} no {name} attribute` | Attribute missing | environment/IO |
| `isel/initializer/expression.rs:191` | Err | `{x} gives '{t}' in {name}, which is not a dimension` | Bad dimension list | environment/IO |
| `isel/initializer/expression.rs:220` | Err | `{x} reads parameter {t}, which is not a parameter number` | Bad parameter text | environment/IO |
| `isel/initializer/expression.rs:297` | Err | `{x} applies {op} to {n} {values}, where {op} takes 2` | Operand count | environment/IO |
| `isel/initializer/expression.rs:364` | Err | `{x} concatenates along {..}, where a concatenate takes one dimension` | Dimension list | environment/IO |
| `isel/initializer/expression.rs:432` | Err | `{x} reduces {n} operands, where a reduce takes 2, a value and its initial value` | Variadic reduce | unsupported |
| `isel/initializer/expression.rs:463` | Err | `{x} applies {op}, an operation the backend does not support` | Operation outside the 18 | unsupported |
| `ir/dtype.rs:145` | Err | `the element type '{s}' is not one the backend supports` | Kernel type outside the 13 | unsupported |
| `isel/extractor/inplace_fix.rs:38` | Err | `an overwritten operand views storage that has another reader` | In-place input with a second reader | unsupported |
| `isel/extractor/inplace_fix.rs:71` | Err | `an overwritten operand has another reader and is pinned in HBM` | Same, pinned | unsupported |

## Rust: internal (panics, asserts, expects)

| Where | Kind | Message | Trigger | Class |
| --- | --- | --- | --- | --- |
| `ir/metadata.rs:24-26`, `ir/pii.rs:70-71` | expect | `Unable to create file`, `Unable to serialize data`, `Unable to write data` | Log or metadata write fails | environment/IO |
| `ir/pii.rs:36` | panic! | `Not a valid pii node operation: {op}` | A node without buffers | internal bug |
| `ir/pii.rs:42` | panic! | `ANY buffer requires at least one child to infer buffer type` | Leaf with buffer ANY | internal bug |
| `ir/pii.rs:48` | panic! | `Cannot infer buffer: children have different buffers` | Mixed child buffers | internal bug |
| `ir/buffer.rs:15` | panic! | `Buffer::ANY does not have a string representation` | Printing ANY | internal bug |
| `ir/egraph.rs:140-149` (10 sites) | expect | `OpBroadcast needs metadata!` (and nine more words) | A rewrite result without metadata | internal bug |
| `ir/egraph.rs:335` | Err(FromOpError) | egg `FromOpError` for the op text | Unparsable e-graph term | internal bug |
| `ir/egraph.rs:407` | panic! | `internal error: a rewrite joins the values {t}{s} and {t}{s}` | Two shapes in one e-class | internal bug |
| `isel/initializer/nodes.rs:30` | unreachable! | `binary operation {op}` | An operator the reader let through | internal bug |
| `isel/rewrites/applier.rs:27, :41, :55` | panic! | `No precondition function for rewrite {name}!` (also metadata, set_shapes) | A rewrite name outside the generated match | internal bug |
| `isel/rewrites/applier.rs:125` | assert_eq! | none (length equality) | RHS e-classes and metadata differ | internal bug |
| `isel/rewrites/ir2ir_rewrites.rs` (40 `assert_eq!`; `:73` unreachable!; `:197` panic!) | assert_eq!, unreachable!, panic! | `rewrite 0 matches a dot`; `OpSlice should have metadata!` | LHS e-class counts differ from the rule | internal bug |
| `isel/extractor/utils.rs:54` | panic! | `annotate_constants: no operator-only representative found for constant eclass {id}` | A constant e-class without a term | internal bug |
| `isel/extractor/fast.rs:191` | debug_assert! | `select2pii_graph: possible infinite loop` | Compiled out by `cargo build --release` (`generator.py:182`); the loop then breaks silently | internal bug |

## C++ allocator (`generic/cpp/malloc/`)

| Where | Kind | Message | Trigger | Class |
| --- | --- | --- | --- | --- |
| `src/main.cc:9` | cerr, return 1 | `Usage: {argv0} input.pii metadata.txt` | Fewer than two arguments | environment/IO |
| `src/main.cc:20` | cerr, return 1 | `Input file must have .pii extension` | Wrong suffix | environment/IO |
| `src/act_malloc.cc:29` | return 1 + reason | `the allocator cannot read {pii}` | Missing candidate file | environment/IO |
| `src/act_malloc.cc:62-63` | return 1 + reason | `the allocator cannot read the line \`{line}\`: {why}` | Parser or builder failure (rows below) | internal bug |
| `src/act_malloc.cc:71` | return 1 + reason | `the candidate holds no line` | Empty candidate | internal bug |
| `src/act_malloc.cc:78` | return 1 + reason | constant `why` (rows `emitter.cc`) | A constant the allocator cannot write | unsupported |
| `src/act_malloc.cc:90` | return 1 + reason | `the allocator orders only {n} of its {m} instructions` | A cyclic or unreachable instruction | internal bug |
| `src/act_malloc.cc:150-156` | return 1 + reason | `the allocator cannot read {metadata}`, `{metadata.why}` | Missing or bad metadata | internal bug |
| `src/act_malloc.cc:168` | return 1 + reason | emitter `why` | Output cannot be written | environment/IO |
| `src/act_malloc.cc:179-182` | return 1 + reason | `the solver found no placement within its time limit of {n}ms`, `the solver found no placement of its tensors` | Capacities too small or time out | unsupported |
| `src/parser.cc:13-14` | cerr + assert(false) | `parse error: not an integer: \`{token}\`` | A token that is not an integer; with NDEBUG, returns 0 | internal bug |
| `src/parser.cc:34,46,57,71,82,104` | return false + why | `missing ':' separator`, `missing '=' separator`, `missing '[' or ']' for storage offset`, ... | A candidate line the writer in `pii.rs` got wrong | internal bug |
| `src/parser.cc:152,162` | return false + why | `the tensor {n} has a shape of rank {r}, where its storage {s} has rank {k}`; `... has {x} along dimension {i}, where its storage {s} holds {cap}` | A kernel tensor larger than the data model | unsupported |
| `src/parser.cc:205,209,257,269,277,307` | refuse (return nullptr + why) | `unknown storage: {s}`, `the tensor {t} is defined twice`, `unknown operand: {o}`, `missing slices attribute`, `missing ':' in slice specification`, `missing dimension attribute` | A malformed candidate | internal bug |
| `src/parser.cc:226,228,232` | refuse | `only HBM storage supports fixed offset specification`, `storage offset must be non-negative`, `the tensor ... does not fit its capacity of {cap}` | A fixed offset the data model cannot hold | unsupported |
| `src/parser.cc:295` | refuse | `the slice {t} takes {a}:{b} of the {n} of {o} along dimension {d}, which its storage {s} does not address` | A slice over a unit dimension | unsupported |
| `src/parser.cc:187-188, :316` | cerr + assert(false) | `create_instruction error: unknown instruction: {op}` | An op word without a case; with NDEBUG, returns nullptr | internal bug |
| `src/parser.cc:341-363` | return false + why | `the tensor {t} has {n} elements, where its region {r} in {s} has {m}`; `... lies in {s} as {a}, where {y} reads it as {b}` | Region and value disagree after selection | internal bug |
| `src/constraint.cc:112-115` and callers `:130,141,152,156,163,175,187` | cerr + assert(false) | `live_range error: tensor has no uses` (and six more) | Inconsistent tensor graph; with NDEBUG, returns `{0,0}` | internal bug |
| `src/constraint.cc:55, :86` | assert | none | Offsets or dimensions of mismatched size | internal bug |
| `src/emitter.cc:30,76,104,130,284` | return false + why | `a constant of element type {t}`, `a constant of value {v}`, `a {type} constant given as an array`, `a constant built by bitcast-convert from u8`, `a constant built by {op}` (wrapped at `:317` as `the allocator cannot write ...`) | A kernel constant outside the writer's cases | unsupported |
| `src/emitter.cc:329,337,340,345,379,387,391,397` | throw runtime_error | `addr is required in metadata.json` (and seven more); caught at `:399` as `the allocator cannot parse metadata.json: ...` | Bad metadata JSON from `ir/metadata.rs` | internal bug |
| `src/emitter.cc:408` | why | `the constant tensor {t} has no known value` | A constant without a value | internal bug |
| `src/emitter.cc:439,450,458` | why | `the output path {p} has no filename component`, `... does not contain a valid filename stem`, `the allocator cannot write {p}` | Bad output path | environment/IO |
| `include/tensor.h:38,87`, `include/solver.h:40` | assert | `a tensor is defined by one instruction` (`:87`); two with no message | Violated tensor invariants; `divisor != 0` | internal bug |
| `include/instructions.h:129,130,140,156,261-263,274,292` | assert | `Too many indices for get_h` and null/bound checks | Generated per-instruction code | internal bug |
