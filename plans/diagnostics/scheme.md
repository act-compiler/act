# Diagnostic scheme for TAIDL v3

One format, one registry, six classes.
Every ISA description is accepted with a defined meaning or rejected with a coded error that cites a spec section.
v3 has no undefined behavior (owner ruling 16): a run-time violation is a defined error.

## Classes and codes

A code is a class letter, a scope letter, and three digits: `RF012`, `UB004`, `VO002`.
A code is never reused or renumbered; a retired code stays in the registry marked retired.

| Class | Letter | Meaning | Raised by | Spec reference | Today's catalog class |
| --- | --- | --- | --- | --- | --- |
| Reject | `R` | The ISA description breaks a rule of the language | front end only (scope `F`) | S9 row and the defining section | invalid spec |
| Unsupported | `U` | The description is valid; this generator does not implement a feature | a generator (scope `B` backend, `O` oracle) | the generator's unsupported-feature row, which cites a section (S1.2) | unsupported |
| Violation | `V` | A run-time case breaks a rule; the norm below says it is an error | oracle at a call; a generator at generation where it can decide | the S1.3 row, which becomes a defined-error row | invalid spec (late) |
| Input | `K` | The kernel or the call is malformed: kernel HLO text, `kernel(...)` declaration, call arguments | backend reader, oracle API | the generator's input rules | environment/IO (kernel rows) |
| Internal | `I` | Should never fire; a bug in a tool | any layer | none | internal bug |
| Environment | `X` | Files, processes, flags, builds, versions | any layer, scripts | none | environment/IO |

Scope letters: `F` front end, `B` act-backend, `O` act-oracle, `A` act umbrella.
An `R` code has scope `F` only, so two generators reject the same description with the same code.
A generator that finds an `R` condition raises `I` (the front end missed it) after the front end check ships (D5).

## Run-time norm

- Every run-time violation is an error, never a silent wrap or clamp.
- The violations are those S1.3 lists today: a region outside its data model, a size of 0 or less, a divisor of 0, an integer outside signed 64 bits, overlapping outputs, a failed constraint.
- The oracle raises a `V` error at the call that reaches it, naming the instruction and the iteration (`at i = 2`).
- The compiler backend rejects at generation what it can decide (`U` or `R`) and states in its unsupported table what it assumes for the rest.
- An in-place mark (S5.5, ruling 17) means the output reuses the marked input's storage; the output region must equal the marked input region, and a difference is `R` when the starts and sizes are literals, `V` otherwise.
- The single norm for docs: a defined error, or a defined behavior stated in one sentence of S5 to S8; nothing in between.

## Message format

```
error[RF025]: instruction 'mac': semantics line 3, column 7: 'add' takes 2 operands, found 3
  --> isa/mini.py:41 (add_semantics)
  = S8.2, S9
```

- Line 1: `<severity>[<code>]: <object>: <position>: <sentence>`; severity is `error` or `warning`.
- Line 2: the ISA description file and line of the call (`file:line (call)`); for a kernel, the kernel file and line; omitted for `X` and `I`.
- Line 3: `= S<id>[, S<id>]`; omitted for `X` and `I`.
- A tool prefixes line 1 with its name when it is a program (`act-backend: error[UB002]: ...`); a Python exception carries the same three lines in `str(error)`.
- Machine form (`--diagnostics=json`, D13): one JSON object per diagnostic with `code`, `severity`, `object`, `message`, `file`, `line`, `column`, `spec`.

## Wording rules

- One sentence, no final period, at most 160 characters, ASCII only.
- Order: object first (`instruction 'mac'`, `data model 'sbuf'`, `state variable 'pc'`, `value '%t'`), then position, then the fact.
- Quote the offending text verbatim in single quotes; never a Python `repr`.
- Use glossary terms (S10): ISA description, data model, region, region value, addressing attribute, computational attribute, state variable, operation word, kernel; never the terms `work/terminology/words-to-avoid.md` lists.
- The verb "reject" appears in docs and code names; a message states the fact: `'add' takes 2 operands, found 3`.
- An `R` message states the rule it breaks in the glossary's words and the found value: `<subject> <verb> <expected>, found <found>`.
- A `U` message is `<feature> is unsupported by <generator>`; it names the feature in the words of the spec section and gives no reason about internals.
- A `V` message is `<subject> violates <rule>`, then the values and the iteration.
- Fixed phrases: `divides by zero`, `is outside the signed 64-bit range`, `is defined by no earlier line`, `is given twice`, `takes N operands, found M`.
- Never: refuse, honor, respect, "cannot", "does not model", "is reserved by the oracle" (use `U`: `name 'x' is unsupported by the oracle`).

## Location

- Front end: each `add_*` and `set_*` call stores its call site (file, line) on the object it builds; an `R` error reads it.
- Texts keep line and column counted from 1 in the text (S9 preamble); a backtick expression adds the semantics offset (`instruction.py:345-368`).
- Kernel: the reader keeps the 1-based line number of each kernel line and reports `kernel.hlo:12`.
- Oracle: a call-time error names the instruction and, inside a loop, the iteration environment (model: `generic/utils.py:662-783`).
- C++ allocator: returns a reason string with a code, never asserts (B25).

## Which layer raises what

| Layer | Raises | Never raises | Exit status |
| --- | --- | --- | --- |
| Front end (`taidl`) | `R`, `X` (its own I/O), `I` | `U`, `V` | Python exception |
| Generator Python (`act_backend`, `taidl_to`) | `U`, `V` (decidable), `X`, `I` | `R` | Python exception |
| Backend Rust and C++ | `U`, `K`, `X`, `I` | `R`, `V` | 1 for `K` and `X`; 2 for `U`; 3 for `I` |
| Generated oracle at a call | `V`, `K`, `X` | `R`, `U` | Python exception |
| Scripts in `act` | `X` | all others | 1 |

## Exceptions (Python)

S9 fixes the built-in bases (`ValueError`, `TypeError`, `SyntaxError`); each class keeps its base so S9 stays true.

| Class | Bases | Used for |
| --- | --- | --- |
| `TaidlError` | `Exception` | Carries `code`, `object`, `position`, `call_site`, `spec` |
| `DeclarationError` | `TaidlError`, `ValueError` | `R` on declarations, regions and marks |
| `ArgumentError` | `TaidlError`, `TypeError` | `R` for a wrong argument type |
| `TextError` | `TaidlError`, `SyntaxError` | `R` on a text or a semantics |
| `UnsupportedError` | `TaidlError`, `NotImplementedError` | `U` |
| `ViolationError` | `TaidlError`, `ValueError` | `V` |
| `InputError` | `TaidlError`, `ValueError` | `K` |
| `InternalError` | `TaidlError`, `RuntimeError` | `I` |
| `ToolError` | `TaidlError`, `RuntimeError` | `X` |

`NotImplementedError` then means only `U` (O6 aligns the oracle).
Rust and C++ build a `Diagnostic { code, object, line, message }` and print it in the same three-line form; a panic hook prints `error[IB000]` and exits 3.

## Registry

- One file `taidl/diagnostics.json` lists every `R` and shared `V` code: `code`, `class`, `object kind`, `template`, `spec`, `status` (active or retired).
- Each generator keeps `U`, `K`, `I`, `X` codes in its own `diagnostics.json`; the build script generates `diagnostics.rs` and `diagnostics.h` for the Rust and C++ code.
- S9 and each generator's unsupported-feature table are generated from the registries (D11); a test fails when a code lacks a row, a spec reference, or a triggering test (D10).
- `registry-seed.md` proposes the first codes from the current catalog.

## Warnings

- A warning is a diagnostic that lets generation or the run continue; v3 emits none for a spec condition (ruling 16).
- `warn_width` (`generic/utils.py:268-271`) becomes a `V` error.
- A generator may warn about its own limits (`W` is a severity, not a class): `warning[UB030]: ...`.
