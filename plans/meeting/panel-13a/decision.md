# Panel 13a decision: surface, contract, and layers

Plan paths are relative to /home/user/act/plans; spec pages sit under language/spec-draft/; code paths are relative to /home/user.
Every citation marked below was re-read for this decision; an uncited claim in a round file carried no weight.
Binding: all owner decisions (OD), panels 3, 10, 11, and 12.

| Question | Decision | Confidence |
| --- | --- | --- |
| Q13a.1 (SF4) | Any non-`bool` object `operator.index` accepts is taken wherever a plain `int` is, stored as exact `int`; `bool` and `float` reject, except `True` and `False` as an `ops.constant` value at `pred` | high |
| Q13a.2 (SF8) | `@acc.instruction(...)` returns an instruction handle; `acc.encode` matches it by identity; calling it raises `TraceError` naming S3.1; bare `@acc.instruction` means `@acc.instruction()` | high |
| Q13a.3 (SF11) | A region whose starts and sizes are all plain belongs to no trace; every other symbolic object used in another trace rejects (S7.2); a recording call outside an instruction function raises at once (S3.1) | high |
| Q13a.4 (SF12) | A signature problem is reported once and the body not run; any other collected problem is recorded and its call returns a placeholder of its result kind; a call with a placeholder argument checks nothing, records nothing, and returns a placeholder; `bool()` on one raises at once | medium-high |
| Q13a.5 (SF16) | OD21 holds as the spelling of printed text: the S4.6 printer and every message write `@s.<name>`, `@a.<name>`, `@c.<name>`; Python objects carry bare names; not escalated | high |
| Q13a.6 (SF21) | `check` rejects a plain size below 1, a plain start below 0, a plain start plus plain size past a declared access dimension (never `d0`), and two output regions of one buffer whose start and size expressions all equal (S5.2) | high |
| Q13a.7 (SF23) | Plain `bool` arguments are removed before recording: `all_of` drops `True`, `any_of` drops `False`; a `require` argument or `all_of` left `False` rejects (S6.3) naming Python `if`; `any_of()` or `all_of()` left with no condition rejects | medium-high |
| Q13a.8 (SF26) | Entries by declaration, then call order; a lint is a `taidl.TaidlWarning` through `warnings.warn_explicit` at the author's file and line, emitted only when `check` finds no problem; the namespace note is the command's print of `Model.metadata_namespaces`; no `Model.notes` | medium-high |
| Q13a.9 (SF28) | `runpy.run_path` with its default `__name__`; a collector active for the run checks every `Accelerator` constructed, in construction order; exit 0, 1 `Rejection`, 2 the file raised, 3 internal error, 4 usage, 5 no `Accelerator` | medium |
| Q13a.10 (SF32) | The backend rejects at kernel compile time when extraction finds no covering, class `K`, object the kernel; the message names the lowest uncovered operation and the stop cause, then lists excluded instructions whose tensor operators include it, as a hint outside the match key | medium-high |
| Q13a.11 (SF38) | Each metadata entry is deep-copied at its declaration call; every read of a node's metadata returns a fresh deep copy in S12.1 types (dicts and lists); nothing mutable is reachable in the `Model` | medium-high |
| Q13a.12 (LY-03) | A link is an `==` directly under `require` whose two sides fold, over `+`, `-`, unary `-`, and `*` by a plain `int`, to `a*E + b*D + c0 == 0` with `a` or `b` in {1, -1}; the `Model` stores (E, k, c, orientation); `//`, `%`, a second attribute, or a link inside `any_of` is not a link | medium-high |
| Q13a.13 (LY-05) | Bare `Comp` or `Addr` has type `s64` and no declared type; the attribute node records whether the type was declared; an encoded attribute with no declared type rejects | high |
| Q13a.14 (LY-09) | All blocks of one ISA specification share one `word`; a different one rejects (S11.3); an instruction with no block has no binary form, and the assembler generator excludes it with a printed reason | high |
| Q13a.15 (LY-10) | Decode matches fixed bits, reads fields (two's complement when signed), applies each link with exact division (S1.3.5), checks each derived value against its type (S1.3.10), evaluates no other written constraint, and stops at the lowest failing word index | high |
| Q13a.16 (LY-11) | A binary program is a sequence of `word`-bit unsigned words indexed from 0, shown in hexadecimal in messages; the language fixes the byte form: `ceil(word/8)` bytes per word, little-endian by default, the top `8*ceil(word/8) - word` bits of the word zero; `byte_order='big'` is a generator option | medium-high |
| Q13a.17 (LY-16) | `check` runs all three block checks and no generator repeats one; two instructions conflict when their fixed bits agree on every bit position both fix | high |
| Q13a.18 (LY-21) | Every collection the front end builds in a node is a tuple; metadata follows Q13a.11 | high |
| Q13a.19 (LY-26) | `meta` is a dict from namespace to entry; an entry is a `str`-keyed dict, possibly empty; the front end rejects another entry type naming node and namespace | high |
| Q13a.20 (LY-28) | `d0` carries no metadata; a datum about hbm goes in the accelerator's entry; no `d0_meta` keyword | high |
| Q13a.21 (LY-31) | Absent metadata means defaults; a datum a generator cannot default excludes that instruction or buffer naming node, namespace, and key; an accelerator-level datum it cannot default rejects (`U`); an empty usable set rejects; neutrality compares the files under the generator's directory and the printed exclusion list in deterministic mode | medium-high |
| Q13a.22 (LY-32) | `generate_<name>(source, /, base_dir=None, **options)` returns a frozen `taidl` report (`output_dir`, `excluded`, `namespaces_read`); it writes only under `<base_dir>/accelerators/<accelerator>/<name>/ (the Q7.11 directory)` and paths the report lists; exclusions print in the two-line warning form; no registry | high |
| Q13a.23 (LY-34) | An act-shipped generator registers one scope letter before release; an outside generator uses scope `G` with codes keyed by its namespace in its own `diagnostics.json`; an unknown key or bad value in its namespace is class `U`, rejection outcome | medium-high |
| Q13a.24 (LY-35) | `CONTRACT_VERSION` is a string `major.minor`, also `Model.contract_version`; an addition bumps the minor; a removal is deprecated one minor, then bumps the major; a generator states its major and lowest minor and rejects outside that window with an `X` error naming both | high |

## Rules

- The edge converts and the core sees one type: `operator.index` at every integer position, exact `int` inside; plain data belongs to no trace (Q13a.1, Q13a.3, Q13a.7).
- `check` decides from plain integers and structural equality only, rejects a case that faults at every issue, and reports a problem once, at the call that caused it (Q13a.4, Q13a.6, Q13a.7).
- Reference prefixes are a spelling of text, never of Python objects (Q13a.5).
- The `Model` is immutable at every depth: tuples for what the front end builds, fresh plain copies for what the author wrote (Q13a.11, Q13a.18).
- Whatever is decidable from the ISA specification alone runs in `check`; a generator checks only its own namespace and its own model limits, and reports as data what a test asserts (Q13a.17, Q13a.21, Q13a.22, Q13a.23).
- A mismatch of versions or widths rejects loudly; the language fixes a default two generators' files must share, and a generator option overrides it (Q13a.14, Q13a.16, Q13a.24).

Where no dissent is listed, the panel converged by round 2.

## Per question

### Q13a.1 (SF4)
- `__index__` is Python's lossless integer protocol, and conformance.md:33 already carves out `pred`.

### Q13a.2 (SF8)
- grammar.md:44 already states the bare form; layers.md:41 has `acc.encode` take what `@acc.instruction` gives, so a handle matched by identity serves a family loop that rebinds one name (grammar.md:50-55).
- `TraceError` subclasses `TypeError` (grammar.md:87), so a `__call__` that raises it keeps Python's error shape and adds the `R` code; Fable's handle with no `__call__` loses the code.

### Q13a.3 (SF11)
- An all-plain region holds a buffer and integers, so it has no trace to leak from; S5.5 compares expressions (conformance.md:56-57), not region identity, and each instruction records its own node at the module-level call site (object_model.md:16).

### Q13a.4 (SF12)
- conformance.md:17 binds collect-all, and panel-3/decision.md:135 has tests assert `code` and `object`, so the entry set must follow from a rule two front ends share.
- Sonnet's first-problem stop breaks collect-all inside a body; Fable's "only cause is a placeholder" needs a cause table per S8 row.
- Opus's per-call rule needs neither: each recording call has one result kind, so a call with a placeholder argument returns a placeholder and reports nothing, in every front end; `bool()` on one stays an at-once raise (grammar.md:70-76).
- Dissent: Sonnet's first-problem rule, conceded in round 2.

### Q13a.5 (SF16)
- OD21 keeps the prefix `@s.` (minutes/owner-decisions.md:20); panel-11/decision.md:5 binds a Python surface with no lexer, and `@` cannot sit in a Python name.
- The one place a reference is spelled as text is the printer and the messages, which already order `state variable 'pc'` (diagnostics/scheme.md:72); the 2.x `KINDS` table (taidl/taidl/instruction.py:397) is the same convention.
- OD21's text settles the question: the prefix stays in every text form, and no reading drops it, so nothing changes direction or scope; Sonnet's request for an owner confirmation is not forwarded.

### Q13a.6 (SF21)
- instructions.md:55 puts the implicit constraints on evaluated starts and sizes; a plain start or size is already evaluated, so a violation is decided from plain integers (conformance.md:10) and faults at every issue.
- conformance.md:75 already rejects a plain shape dimension below 1 by the same logic; the equal-output row mirrors the `apart` row at conformance.md:56.
- Dissent: none after round 2; Opus and Fable added the negative-start row.

### Q13a.7 (SF23)
- A family loop variable is plain (grammar.md:47), so `require(n <= 4)` reaches `require` as a `bool`; rejecting the passing members has no genuine cause (conformance.md:8).
- A condition is a comparison (expressions.md:35) and conditions have no boolean leaf, so recording `True` would add a node kind no generator reads and break `Model` equality between members; dropping a plain `bool` is structural, not algebra (panel-3/decision.md:92).
- instructions.md:63 ("a check on plain values only is a Python `assert`") takes the matching edit.
- Dissent: Opus's round-1 recorded `True`, conceded.

### Q13a.8 (SF26)
- `Model.notes` would make two conforming front ends return unequal `Model` objects over a lint that is not part of conformance (instructions.md:128, object_model.md:23); a runner gets structured records from `warnings.catch_warnings(record=True)`.
- The warning is emitted only after `check` finds no problem, so `-W error::taidl.TaidlWarning` never hides a `Rejection`; conformance.md:25 has the command print the namespace note from `Model.metadata_namespaces` (generator_metadata.md:24), so the note needs no warning.
- scheme.md:137 ("v3 emits none for a spec condition") needs one clause for a lint outside conformance; diagnostics track.
- Dissent: Sonnet's `Model.notes`, conceded.

### Q13a.9 (SF28)
- `runpy.run_path` keeps a `__main__` block inert with its default name; a `run_name` equal to a module name would shadow that module in `sys.modules` for the run, so the stem rule is dropped.
- A globals scan misses `variants = [build(n) for n in (4, 8)]` and exits "no accelerator" on a file that built two, a failure with no genuine cause (OD40); a collector that `Accelerator.__init__` appends to, installed for the command's run only, catches every construction and leaves `taidl.check` with no registry.
- The exit codes name phases that already exist: a `Rejection`, author code raising, `check` itself raising, usage, nothing built; they follow pytest's 1, 3, 4, 5; v3 examples guard generator calls under `if __name__ == '__main__':`.
- Dissent: Sonnet and Fable's three-code set, and their globals scan; Opus's collector is the one that meets OD40 on a list of accelerators.

### Q13a.10 (SF32)
- act-backend/act_backend/generic/src/pipeline.rs:204-219 shows the backend knows only "no candidate was extracted" or the first drop reason, plus the stop cause; "needs an excluded instruction" is unobservable without a second search.
- act-backend/act_backend/generic/src/isel/extractor/slow.rs:116 keeps a subtree only when every child has an extraction, so one bottom-up pass finds the lowest uncovered operation.
- Sonnet's filter "all operators occur in the kernel" admits most excluded instructions on a large kernel; filtering by the uncovered operation is exact.
- The class is `K`: OD40 lets the generated backend stop only on a kernel "outside the input specification", scheme.md:16 names kernel input as `K`, and scheme.md:20-23 defines the `U` outcomes at generation time, which is over.
- Dissent: Fable and Sonnet named `U`; neither cited a generation-time act.

### Q13a.11 (SF38) and Q13a.18 (LY-21)
- Verified this round: `json.dumps` and `copy.deepcopy` of a `MappingProxyType` raise `TypeError`; `json.dumps(m, default=dict)` works but only when every emitter remembers it.
- S12.1 defines plain data as exactly list and dict and rejects a tuple (generator_metadata.md:12-13), so a generator must read back the same types the architect wrote; a proxy or a tuple is a third type with `(1,) != [1]` as a silent wrong branch.
- The copy is taken at the declaration call (generator_metadata.md:23), so a family loop that mutates one dict publishes each iteration's value; a fresh copy per read keeps `n.meta == n.meta` and leaves nothing mutable reachable (object_model.md:9).
- Collections the front end builds (attributes, regions, constraints, slices, fixed bits, encoded and derived lists) are tuples, which hash and compare by value; U2.md:12 reads "a fresh plain-data copy per read" in place of "read-only plain-data mapping".
- Dissent: Fable's proxies with `default=dict`; a mutation of a read copy is a write to the generator's own data, not a silent loss.

### Q13a.12 (LY-03)
- panel-10/decision.md:20 refuses "a nonaffine link", so the binding test is affinity, a property of the function, which a fold defines and an enumerated list of trees only approximates; LY-03's own list leaves `4 + 8*D` and `-(8*D)` undecided.
- The fold reads one condition once and compares no two trees, so panel-3/decision.md:92 (no normalizer for S6.1 equality) is untouched; the recorded constraint is still evaluated as written (expressions.md:26).
- The `Model` stores (E, k, c, orientation) (object_model.md:12, layers.md:67-70), so a generator never sees the spelling; `E + 1 == 8*D` is a link with a = 1; a fold that leaves signed 64 bits, or gives k = 0, rejects naming the attribute.
- Dissent: Sonnet's one-bare-attribute side, which rejects `E + 1 == 8*D` for no gain once both sides fold.

### Q13a.13 (LY-05)
- panel-10/decision.md:17 binds "a declared type whose width equals its slice total" and :61 gives the reason; object_model.md:19 already records the flag.
- Dissent: Fable's round-1 "bare equals `Comp[s64]`", conceded.

### Q13a.14 (LY-09)
- Mixed widths are the variable length panel-10/decision.md:63 defers; "no word matches the fixed bits of two instructions" (layers.md:53) is defined only for one width; a block-less instruction is the S1.2 exclusion (overview.md:27).

### Q13a.15 (LY-10)
- A written constraint may read state (expressions.md:14), which a binary lacks, so decode cannot evaluate it; the oracle checks it at issue.
- A derived attribute "may carry its own type" (layers.md:71), and `D = (E - c) / k` can leave it for an in-range E, so decode checks the derived type; an encoded field of width w under a type of width w has no out-of-range value, so no field-range check is listed.
- Dissent: Fable's round-1 omission of the derived type check, conceded.

### Q13a.16 (LY-11)
- Two assemblers for one ISA specification must emit one file, which only a language default gives; a required keyword with one value in practice is noise at every call (OD41).
- RISC-V fixed parcel order little-endian whatever the data endianness, and every target in OD15's scope is little-endian; the padding is stated per word so a big-endian override keeps its meaning.
- Dissent: Opus's no-default, conceded in round 2.

### Q13a.17 (LY-16)
- OD42 gives `check` what is decidable from the ISA specification alone; panel-10/decision.md:17 wrote "at generation time" before panel-11/decision.md:28 made `check` the step that "runs every front-end rule", and `check` precedes every generator, so the two texts agree.
- layers.md:50 and conformance.md:110 move the second block check to the front end's Layers table.

### Q13a.19 (LY-26), Q13a.20 (LY-28)
- S12.3 rejects "an unknown key in its own namespace" (generator_metadata.md:26), which presupposes a dict; `d0` is predeclared (declarations.md:19) and the accelerator entry reaches every generator (generator_metadata.md:8).

### Q13a.21 (LY-31)
- OD39 makes metadata optional and ignored by other generators; the S1.2 outcome for "valid, not modeled here" is exclusion (overview.md:27), so a missing per-node datum excludes that node and the rest generates.
- An accelerator-level datum has nothing to exclude, so a generator that cannot default it rejects with `U` naming node, namespace, and key (OD40); conformance.md:111 rejects an empty usable set.
- OD27 makes wall-clock output non-deterministic, so neutrality is compared in deterministic mode.
- Dissent: Opus's round-1 blanket rejection, conceded; Fable's node-level split is adopted.

### Q13a.22 (LY-32)
- act-oracle/taidl_to/generator.py:21-33 and act-backend/act_backend/generator.py:79-96 already write under `<base_dir>/targets/<accelerator>/<name>`; the backend binary at `<base_dir>/backends/<accelerator>` is a listed path for track B to state or move.
- Both return `None` today, so a T12 runner asserting `code` and `object` (panel-3/decision.md:135) would parse printed text or a crate; a frozen report in `taidl` is part of the U2 contract, not a generator touching `taidl`.
- Exclusions through `warnings` would collide with Q13a.8's `-W error`, so they print in the two-line warning form (scheme.md:21); positional-only `source` ends the `model` against `accelerator` split (generator_metadata.md:34, shared/plan/U11.md:4).
- Dissent: Opus's `None`, conceded.

### Q13a.23 (LY-34)
- A code is class, scope, and three digits (scheme.md:9), so an outside generator needs a scope letter, and a letter in `taidl` per generator breaks generator_metadata.md:39; scope `G` with the namespace qualifying the code and the tool prefix on line 1 (scheme.md:66) resolves both.
- Each generator already keeps its own `diagnostics.json` (scheme.md:131); `K` is run input (scheme.md:16) while a bad key sits in the ISA specification, the `U` rejection outcome (scheme.md:20-23).
- Dissent: Sonnet's `K`, conceded.

### Q13a.24 (LY-35)
- A generator that reads a member added in 3.2 and meets a 3.1 `Model` fails later with `AttributeError`, so the lower minor bound is needed; U2.md:6 ("an added member keeps the version") is edited so an addition bumps the minor.
- OD33 pins released versions, so the check fires only in development installs; versions are class `X` (scheme.md:18).
- Dissent: Sonnet's major-only check, conceded.

## Escalation to the owner

- None; Q13a.5 is settled by OD21's text together with panel 11's binding surface, and no decision above changes direction, scope, or an external commitment.
