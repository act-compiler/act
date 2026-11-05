# Catalog: act-oracle

Scope: `act-oracle/taidl_to/*.py`, `taidl_to/generic/*.py` (copied into each generated oracle), and `taidl_to/templates/templates.txt`.
Class values: `invalid spec` (front end should reject), `unsupported` (unsupported by this generator), `internal bug` (should never fire), `environment/IO`.
Rows tagged "call" fire when a kernel author runs the generated oracle, not at generation; they are classed `environment/IO` (input of a run) and counted apart.
Paths are relative to `act-oracle/taidl_to/`.

## Generation: names

| Where | Kind | Message | Trigger | Class |
| --- | --- | --- | --- | --- |
| `generator.py:41` | NotImplementedError | `{model}: data model name '{model}' is reserved by the oracle` | A data model named like a trace counter, a debug value, or `In<k>`/`Out<k>` | unsupported |
| `api_generator.py:59` | NotImplementedError | `{name}: instruction name '{name}' is reserved by the oracle` | An instruction named as a Python or `api.py` name | unsupported |
| `instruction_codegen.py:202` | NotImplementedError | `{func}: attribute name '{a}' is reserved by the oracle` | An attribute named as a generated local | unsupported |
| `semantic_generator.py:29` | NotImplementedError | `{instr}: value name '{v}' is reserved by the oracle` | A semantics value named like a data model or trace name | unsupported |
| `generator.py:74-82` | CalledProcessError | cmake output | The debug-handler build fails | environment/IO |

## Generation: semantics

| Where | Kind | Message | Trigger | Class |
| --- | --- | --- | --- | --- |
| `semantic_generator.py:16` | ValueError | `{instr}: no semantics gives '%Out0' for output region 0` | Output regions and no semantics | invalid spec |
| `semantic_generator.py:40` | ValueError | `{instr}: '{v}' has no {where} region` | `%In<k>` with no input region at all (the front end checks it only after `set_inputs`) | invalid spec |
| `semantic_generator.py:45` | ValueError | `{instr}: '{v}' is {t}, its data model {buffer} holds {t2}` | A region value typed unlike its data model | invalid spec |
| `semantic_generator.py:53` | ValueError | `{instr}: '{v}' is {type}[..], more elements than the {room} of its {where} region in {buffer}` | A value larger than its region (literals only) | invalid spec |
| `oracle_visitor.py:239` | NotImplementedError | `{instr}: operation '{w}' is not supported by the oracle` | A word outside `TYPES`; dead today (O5 in `work/oracle/plan/checks.md`) | internal bug |
| `oracle_visitor.py:255` | NotImplementedError | `{instr}: operation '{w}' at type '{t}' is not supported by the oracle` | A word at a type outside its row | invalid spec |
| `oracle_visitor.py:261` | NotImplementedError | `{instr}: operation '{w}' at an index of type '{t}' is not supported by the oracle` | A non-integer start index | invalid spec |
| `oracle_visitor.py:280` | ValueError | `{instr}: compare of '%{v}' takes type {a} or {b} at {t}, found {k}` | A `compare` type that does not fit its operands | invalid spec |
| `oracle_visitor.py:289` | ValueError | `{instr}: {w} of '%{v}' {reason}` | Shapes or attributes disagree (`generic/shapes.py:disagreement`) | invalid spec |
| `oracle_visitor.py:168` | ValueError | `{instr}: {mlir_literal error}` | An `epsilon` its f32 cannot hold | invalid spec |
| `oracle_visitor.py:387` | TypeError, ValueError | `{instr}: constant {literal} at {type} is out of its range` (also `is a pred literal`, `is not an integer`) | A constant outside its element type | invalid spec |
| `oracle_visitor.py:416` | NotImplementedError | `{instr}: an expression in '{name}' is not supported by the oracle` | A backtick expression in a scalar operation attribute | unsupported |
| `generic/utils.py:555` | ValueError | `copy of {x} : {t} does not have the result type {t2}` | A `copy` whose result type differs; no instruction name | invalid spec |
| `generic/utils.py:397, :406, :412, :422, :425` | TypeError, ValueError | `{shown} at {typ} is a pred literal`, `... is out of its range`, `... is not an integer` | A literal rejected by `mlir_literal`; wrapped at `oracle_visitor.py:387` and `generic/utils.py:471-472` | invalid spec |

## Generation: loop expressions and checks emitted into the oracle

| Where | Kind | Message | Trigger | Class |
| --- | --- | --- | --- | --- |
| `generic/utils.py:161` (raised `:193, :206, :210`) | ValueError | `{instr}: {name} {expr!r} is not a loop expression` | A start or size text the loop reader cannot parse | invalid spec |
| `generic/utils.py:203` | ValueError | `{instr}: {name} {expr!r} reads {token}, which no enclosing loop holds` | A loop variable out of scope | invalid spec |
| `generic/utils.py:80` | ZeroDivisionError(env) | the iteration environment only; rewrapped at `templates/templates.txt:319-320` and `:359-360` | A divisor that is 0 at some iteration | invalid spec |
| `generic/utils.py:662` | ValueError | `{instr}: {kind} {text} is {n} [at {i = v}], not positive` | A size or dimension below 1 at a call (`check_sizes`) | invalid spec |
| `generic/utils.py:709` | ValueError | `{instr}: window [{a}:{b}] on dimension {k} of {buffer} runs outside its extent {n}` | A region outside its data model at a call | invalid spec |
| `generic/utils.py:743` | ValueError | `{instr}: '%{v}' is {t}[..], not the shape [..] of its {where} region in {buffer}` | A value shape differing from its region at a call | invalid spec |
| `generic/utils.py:783` | ValueError | `{instr}: {word} of '{lhs}' {reason} [at {i = v}]` | A looped line whose shapes disagree at a call | invalid spec |
| `generic/utils.py:760` | Unroll (control flow) | none | A held value changes over a loop; caught by the caller | internal bug |
| `generic/utils.py:796-804` | ValueError | `{instr}: constraint '{text}' does not hold [with {ref=v}] [at {i = v}]` | A call that breaks a constraint | invalid spec |

## Call time: kernel declaration and arguments (class environment/IO, tag "call")

| Where | Kind | Message | Trigger | Class |
| --- | --- | --- | --- | --- |
| `generic/decorator.py:21` | ValueError | `unknown simulation platform {p!r}` | Platform other than CPU or GPU | environment/IO |
| `generic/decorator.py:32` | ValueError | `simulation platform 'GPU' cannot be set: jax has started only its CPU platform` | GPU after jax started | environment/IO |
| `generic/decorator.py:71` | TypeError | `kernel {n!r}: hbm {h!r} is not an integer` | Non-integer `hbm` | environment/IO |
| `generic/decorator.py:77` | TypeError | `{what}: addr {a!r} is not an integer` | Non-integer address | environment/IO |
| `generic/decorator.py:81` | TypeError | `{what}: shape {s} has dimension {d!r}, not an integer` | Non-integer dimension | environment/IO |
| `generic/decorator.py:85` | ValueError | `{what}: shape {s} has dimension {d}, below zero` | Negative dimension | environment/IO |
| `generic/decorator.py:92, :96` | TypeError | `{what}: dtype {t!r} is not supported by the oracle` | A dtype outside `XLA_ELEM_TYPE` | unsupported |
| `generic/decorator.py:100` | ValueError | `{what}: {n} elements of {dtype} do not fill whole bytes` | Sub-byte count not a multiple | environment/IO |
| `generic/decorator.py:107, :109` | ValueError | `{what} has shape {s}, declared {d}`; `{what} has dtype {t}, declared {u}` | A constant differing from its declaration | environment/IO |
| `generic/decorator.py:120` | ValueError | `{kind} {i} of kernel {n!r}, {b} bytes at byte {a}, runs outside HBM of {h} bytes` | A buffer past HBM | environment/IO |
| `generic/decorator.py:136` | ValueError | `{kind} {i} of kernel {n!r}, ... overlaps {kind} {j}, ...` | Two buffers overlap | environment/IO |
| `generic/decorator.py:150` | ValueError | `unknown oracle mode {m!r}` | Mode not `fsim-compile`, `fsim`, ... | environment/IO |
| `generic/decorator.py:156, :163` | ValueError | `kernel {n!r} takes no arguments in 'fsim-compile'`; `kernel {n!r} is not compiled; run it in 'fsim-compile' first` | Wrong call order | environment/IO |
| `generic/decorator.py:169` | ValueError | `kernel {n!r} declares {k} inputs, got {m} arguments` | Wrong argument count | environment/IO |
| `generic/decorator.py:180, :183, :185` | TypeError, ValueError | `{what} is a {type}, not an array`; `... has shape ..., declared ...`; `... has dtype ..., declared ...` | A bad argument | environment/IO |
| `generic/utils.py:278, :281` | TypeError | `{instr}: {name} {v!r} is not an integer`; `{instr}: computational attribute {name} takes an integer, not the loop expression {v!r}` | A bad attribute value in an API call | environment/IO |
| `generic/utils.py:675, :677, :689, :692` | ValueError, TypeError | `debug: {d!r} names no buffer`; `... gives no index; ...`; `... has index {e!r}, not an integer`; `... gives {n} indices for the {m} dimensions of {name}` | A bad `debug` request | environment/IO |

## Warnings and silent paths

| Where | Kind | Message | Trigger | Class |
| --- | --- | --- | --- | --- |
| `generic/utils.py:268-271` | warnings.warn | `{call}: {what} is {v}, past the signed 64-bit range` (called at `decorator.py:72,78`, `utils.py:287,659,706`) | An integer outside signed 64 bits (implementation-defined in S1.3, so a warning stands for a defined case) | invalid spec |
| `generic/shapes.py:77-146` | return None | none | Shape helpers return None for "unknown"; the check is skipped | internal bug |
| `generator.py:30-33` | rmtree | none | The previous oracle is removed before any check; by design, pinned by `tests/test_refusals.py:467` | environment/IO |
| `instruction_codegen.py` (instruction without semantics) | silent | none | An instruction with no semantics and no output regions generates an empty body (`semantic_generator.py:19`) | invalid spec |
