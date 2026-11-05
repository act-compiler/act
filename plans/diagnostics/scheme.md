# Diagnostic scheme for TAIDL v3

One format, one registry, six classes.
Every ISA specification is accepted with a defined meaning or rejected with a coded error that cites a TAIDL language specification section.
v3 has no undefined behavior (owner decision 16): a run-time case that breaks a rule is a fault, a defined error.

## Classes and codes

A code is a class letter, a scope letter, and three digits: `RF012`, `UB004`, `VF002`.
A code is never reused or renumbered; a retired code stays in the registry marked retired.

| Class | Letter | Meaning | Raised by | Spec reference | Today's catalog class |
| --- | --- | --- | --- | --- | --- |
| Reject | `R` | The ISA specification breaks a rule of the language | front end only (scope `F`) | S9 row and the defining section | invalid spec |
| Unsupported | `U` | The description is valid; this generator does not implement a feature | a generator (scope `B` backend, `O` oracle) | the generator's unsupported-feature row, which cites a section (S1.2) | unsupported || Fault | `V` | A run-time case breaks a rule; the norm below says it is an error | oracle at a call; a generator at generation or selection where it can decide (scope `F`) | the S1.3 row, which becomes a defined-error row | invalid spec (late) |
| Input | `K` | The input is invalid or outside the input specification: kernel text, `kernel(...)` declaration, call arguments, assembly code or binary given to an oracle | backend reader, oracle API | the generator's input rules | environment/IO (kernel rows) |
| Internal | `I` | Should never fire; a bug in a tool | any layer | none | internal bug |
| Environment | `X` | Files, processes, flags, builds, versions | any layer, scripts | none | environment/IO |

A `U` code has two outcomes (S1.2): exclusion and rejection.
An exclusion leaves one instruction out of the generated tool and is reported at severity `warning`, printed at generation and stored in the generated tool.
A rejection stops generation.
A generator excludes an instruction its target cannot model, and rejects a problem the architect can correct (Q11.3).
The oracle excludes nothing.

Scope letters: `F` front end and the language, `B` act-backend, `O` act-oracle, `A` act umbrella.
The letter `V` names the violated rule; the class is Fault.
An `R` code has scope `F` only, so two generators reject the same description with the same code.
A `V` code has scope `F` too (`VF001`): a fault is a language rule, every generator reuses the code, and the tool prefix names the raiser.
What a generator cannot decide goes in its unsupported table (`U`), not in a coded fault.
A candidate that fails inside backend search is a recorded drop with the failure's own code, never a stop (`failure-policy.md`).
A generator that finds an `R` condition raises `I` (the front end missed it) after the front end check ships (D5).

## Run-time norm

- Every fault is a defined error, never a silent wrap or clamp.
- A fault writes no region and no state; the program stops at the first fault, and every earlier issue has committed.
- A fault depends only on attributes, state, and region text; tensor data never faults.
- The faults are those S1.3 lists today, restated: a region outside its data model, a size of 0 or less, a divisor of 0, an expression operator that leaves signed 64 bits, a failed constraint or other REQUIRE, a region value of another shape, a read of an on-chip element no earlier issue wrote, and a backtick constant outside its element type.
- Overlap faults (outputs with each other, inputs with outputs) are failed constraints, one fault (S1.3.5), code `VF001`; an undeclared same-buffer pair must be equal or apart.
- Integer divide or remainder by zero, `min / -1`, and float-to-integer `convert` out of range or NaN in tensor data give fixed values and no code (joint panels 1 and 5).
- The report names the lowest issue index; within one issue any applicable clause id may be named.
- The oracle raises a `V` error at the call that reaches it, naming the instruction and the iteration (`at i = 2`); a test per code checks that no region and no state changed.
- A backend never emits an issue that faults, and lists each precondition it cannot see in its coverage doc.
- A backend never assumes an unsupported input away: it rejects at generation what it can decide (`U` or `R`), and at selection a disagreeing declared and computed dim is the shared fault `VF`, not a panic.
- A marker (`set_inplace`, `set_apart`, `set_no_partial_overlap`) is shorthand that expands to the constraints it stands for (owner decision 30); a violated marker is `VF001` like any constraint.
- A permitted overlap reads the state before the issue.
- An in-place mark that pairs an input and an output in different buffers is rejected (owner decision 24).
- The single norm for docs: a defined error, or a defined behavior stated in one sentence of S5 to S8; nothing in between.

## Message format

```
error[RF025]: instruction 'mac': 'add' takes 2 operands, found 3 [S8.2]
  --> isa/mini.py:41 (ops.add)
warning[UB001]: instruction 'sem': two output regions are unsupported by act-backend [S5.2]
  --> isa/mini.py:58 (instruction)
```

- Line 1: `<severity>[<code>]: <object>: <position>: <sentence> [S<id>[, S<id>]]`; severity is `error` or `warning`.
- The position is the line and column inside a text, which only the 2.x string front end has; in v3 it is omitted and line 2 is the author's line.
- The suffix is a TAIDL language specification section id (`S9.4` for a numbered S9 row); it ends line 1 so CI logs, `grep`, and the first line of `str(error)` carry it; omitted for `X` and `I`.
- A generator's `coverage.md` id never appears in a message; `TaidlError.spec` carries the section id as a field.
- Line 2: the ISA specification file and line of the call (`file:line (call)`); for a kernel, the kernel file and line; omitted for `X` and `I`.
- An exclusion line has the same two lines at severity `warning`; its object is the instruction and its line 2 is the first line of the instruction's function.
- A tool prefixes line 1 with its name when it is a program (`act-backend: error[UB002]: ...`); a Python exception carries the same two lines in `str(error)`.
- Machine form (`--diagnostics=json`, D13): one JSON object per diagnostic with `code`, `severity`, `object`, `message`, `file`, `line`, `column`, `spec`.

## Wording rules

- One sentence, no final period, at most 160 characters including the suffix, ASCII only.
- Order: object first (`instruction 'mac'`, `data model 'sbuf'`, `state variable 'pc'`, `value '%t'`), then position, then the fact.
- Quote the offending text verbatim in single quotes; never a Python `repr`.
- Use glossary terms (S10): ISA specification, data model, region, region value, addressing attribute, computational attribute, state variable, tensor operator, kernel function; never the terms `glossary/words-to-avoid.md` lists.
- The verb "reject" appears in docs and code names; a message states the fact: `'add' takes 2 operands, found 3`.
- An `R` message states the rule it breaks in the glossary's words and the found value: `<subject> <verb> <expected>, found <found>`.
- A `U` message is `<feature> is unsupported by <generator>`; it names the feature in the words of the spec section and gives no reason about internals.
- A `V` message is `<subject> violates <rule>`, then the values and the iteration; `V` names the violated rule.
- Fixed phrases: `divides by zero`, `is outside the signed 64-bit range`, `is defined by no earlier line`, `is given twice`, `takes N operands, found M`.
- A removed v2 construct is one `R` message naming the construct, its replacement, and the section id (see Removed names).
- A rejected spelling is one `R` message naming the spelling found and the spelling to use: `instruction 'mac': '/' applies to an integer expression only as '//', found '/' [S6.2]`.
- Never: refuse, honor, respect, "cannot", "does not model", "is reserved by the oracle" (use `U`: `name 'x' is unsupported by the oracle`).

## Location

- Front end: every node of the model records its call site, the first frame outside `taidl`; an `R` error reads it, so it names the line the architect wrote.
- A whole-instruction problem names the first line of the instruction's function.
- A trace-time error, such as `if` on a symbolic value, is raised in the architect's own frame, so the Python traceback and line 2 agree.
- `check` collects the problems that are not trace-time errors and raises them as one `Rejection`; the 2.x string front end stores the call site on each `add_*` and `set_*` call instead.
- Texts keep line and column counted from 1 in the text (S9 preamble); a backtick expression adds the semantics offset (`instruction.py:345-368`).
- Kernel: the reader keeps the 1-based line number of each kernel line and reports `kernel.hlo:12`.
- Oracle: a call-time error names the instruction and, inside a loop, the iteration environment (model: `generic/utils.py:662-783`).
- C++ allocator: returns a reason string with a code, never asserts (B25).

## Which layer raises what

| Layer | Raises | Never raises | Exit status |
| --- | --- | --- | --- |
| Front end (`taidl`) | `R`, `X` (its own I/O), `I` | `U`, `V` | Python exception |
| Generator Python (`act_backend`, `taidl_to`) | `U` (a warning for an exclusion, an error for a rejection), `V` (decidable, scope `F`), `X`, `I` | `R` | Python exception |
| Backend Rust and C++ | `U`, `K`, `X`, `I`; `V` at selection (shared shape fault) | `R` | 1 for `K`, `V`, and `X`; 2 for `U`; 3 for `I` |
| Generated oracle at a call | `V`, `K`, `X` | `R`, `U` | Python exception |
| Scripts in `act` | `X` | all others | 1 |

## Exceptions (Python)

S9 fixes the built-in bases (`ValueError`, `TypeError`, `SyntaxError`); each class keeps its base so S9 stays true.
The front-end alias `Fault = tuple[CommonToken, str]` (taidl/taidl/operations.py:167-168) is renamed `Rejection`, so `Fault` means only the run-time class.
`Rejection` is the exception `check` raises: it holds one entry per problem, each with code, section id, instruction, and the author's file and line, and `str` prints each entry in the two-line form.

| Class | Bases | Used for |
| --- | --- | --- |
| `TaidlError` | `Exception` | Carries `code`, `object`, `position`, `call_site`, `spec` |
| `DeclarationError` | `TaidlError`, `ValueError` | `R` on declarations, regions and marks |
| `ArgumentError` | `TaidlError`, `TypeError` | `R` for a wrong argument type |
| `TextError` | `TaidlError`, `SyntaxError` | `R` on a spelling in an instruction body (raised at trace time) or on a text (2.x) |
| `Rejection` | `TaidlError`, `ValueError` | The collected `R` entries `check` raises |
| `UnsupportedError` | `TaidlError`, `NotImplementedError` | `U` |
| `FaultError` | `TaidlError`, `ValueError` | `V` |
| `InputError` | `TaidlError`, `ValueError` | `K` |
| `InternalError` | `TaidlError`, `RuntimeError` | `I` |
| `ToolError` | `TaidlError`, `RuntimeError` | `X` |

`NotImplementedError` then means only `U` (O6 aligns the oracle).
Rust and C++ build a `Diagnostic { code, object, line, message }` and print it in the same two-line form; a panic hook prints `error[IB000]` and exits 3.

## Registry

- One file `taidl/diagnostics.json` lists every `R` code and every `V` code (scope `F`): `code`, `class`, `object kind`, `template`, `spec`, `status` (active or retired).
- The route of the registry into a generated oracle is open; the shape module's copy stamped with the taidl version is the model.
- Each generator keeps `U`, `K`, `I`, `X` codes in its own `diagnostics.json`; the Python generator emits the Rust constants with the generated crate, and a small C++ header of the `I` codes is checked in and tested against the registry.
- S9 and each generator's unsupported-feature table are generated from the registries (D11); a test fails when a code lacks a row, a spec reference, or a triggering test (D10).
- `registry-seed.md` proposes the first codes from the current catalog.

## Warnings

- A warning is a diagnostic that lets generation or the run continue; v3 emits none for a spec condition (owner decision 16).
- `warn_width` (`generic/utils.py:268-271`) becomes a `VF005` error.
- A generator warns about its own limits (`W` is a severity, not a class): an exclusion is `warning[UB001]: ...`.
- A warning of this kind is also stored in the generated tool, so it is not lost with the terminal.

## Removed names

- A removed v2 construct raises one coded `RF` rejection (`RF051`) naming the construct, its replacement, and the section id.
- One removed-names table feeds one lookup: a class-level `__getattr__` for members and the `OPERATIONS` lookup for words.
- There is no stub per member and no converter in `taidl` 3.
- The row for `add_semantics` names the converter script on the 2.x line (U13).
- The table is a tested feature, not a placeholder; it is seeded after the v3 API is fixed (D15).
- Markers stay (owner decision 30), so `set_inplace` does not enter the table.
