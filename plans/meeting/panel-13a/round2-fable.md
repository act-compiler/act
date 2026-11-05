# Panel 13a, round 2: Fable, compiler infrastructure and assembler architect

Code paths are relative to /home/user; plan paths to /home/user/act/plans; spec pages are under language/spec-draft/.
Sources re-read this round: riscv-opcodes `src/riscv_opcodes/shared_utils.py`, LLVM `llvm/utils/TableGen/DecoderEmitter.cpp` and `llvm/lib/Target/RISCV/RISCVInstrFormats.td` at main, and the Python `json` page; the RISC-V unprivileged manual itself was unreachable (three mirrors 404 or 503), so its length-encoding rule rests on round-1 quotations only.
Where all three round-1 files agree the entry is one line; the rebuttals and concessions carry the evidence.

### Q13a.1 (SF4) NumPy integers
- No opposition: all three take any non-`bool` `__index__` object and store an exact `int`; Sonnet's `ops.constant` at `pred` exception is already conformance.md's own text (surface.md:15), so nothing divides.
- Final: round-1 position, confidence high.

### Q13a.2 (SF8) What `@acc.instruction` returns
- No opposition: handle, call raises naming S3.1, bare form equals `()`; grammar.md:52 already states the bare form, and layers.md:46 takes the handle.
- Final: round-1 position, confidence high.

### Q13a.3 (SF11) All-plain regions across traces
- No opposition on the rule; Opus adds that each instruction still gets its own region node with the module-level call site (object_model.md:16), which I adopt.
- Final: round-1 position, confidence medium-high.

### Q13a.4 (SF12) Tracing after a collected problem
- Opposing (Sonnet): "a problem whose only cause is a placeholder" needs a definition of cause per S8 row, so two front ends differ on `ops.add(bad, x)`; end the body at its first collected problem instead.
- Concession in part: my cause wording was the weak point; Opus's rule, "every check with a placeholder among its inputs is skipped", is decidable per call and needs no cause.
- Rebuttal: conformance.md:17 promises that one run reports every problem, and a body with two bad regions reports one under Sonnet's rule; the placeholder is clang's `RecoveryExpr` with `containsErrors()`, and assembler parsers recover per statement, where the unit here is one recording call, not the whole body.
- Final: signature problem reported once and the body not run; every other collected problem returns a placeholder of the expected kind; a check with a placeholder input is skipped; a placeholder reaching `bool()` still raises at once (grammar.md:70).
- Confidence medium-high; divides: whether one placeholder class can keep S4 shape inference total, which no file yet shows.

### Q13a.5 (SF16) OD21 on the Python surface
- No opposition: `@s.`, `@a.`, `@c.` are the spelling of printed text and messages (scheme.md:72), the objects carry bare names; Sonnet's escalation is the same confirmation I asked for.
- Final: round-1 position, confidence high; the moderator forwards it as a reading, not a change.

### Q13a.6 (SF21) Always-faulting plain cases
- Opposing (Sonnet): add a third row, a plain start below 0.
- Concession: a negative plain start faults at every issue whatever the bound, `d0` included (OD30's "no address is negative", owner-decisions.md:29), and it is decided from plain integers alone (conformance.md:10); I missed it.
- Final: reject a plain size below 1, a plain start below 0, a plain start plus plain size past a declared access dimension, and two output regions of one buffer whose start and size expressions all equal; `d0` is exempt from the bound row only.
- Confidence high; nothing divides.

### Q13a.7 (SF23) A plain `bool` in a constraint
- Opposing (Opus): a plain `True` is "recorded as written"; (Sonnet) it is accepted and dropped.
- Position change toward Sonnet: a condition is a comparison (expressions.md:35) and object_model.md:21 stores a condition per constraint, so recording `True` needs a new node kind that no generator can use, while dropping is the structural step panel 3 allows (panel-3/decision.md:92-93).
- Final: plain `bool` arguments are removed before recording; a `False` that leaves `require` or `all_of` unsatisfiable is rejected under S6.3 with a message naming Python `if`; `any_of` or `all_of` left with no condition is rejected with the empty-argument message.
- Confidence medium-high; divides: Opus's recorded `True` leaf, which `Model` equality (object_model.md:23) would then distinguish from the member written without the line.

### Q13a.8 (SF26) Entry order and the warning channel
- Opposing (Sonnet): both channels, `warnings.warn_explicit` and a `Model.notes` tuple, since a runner under `-W ignore` loses the warning.
- Concession: `warn_explicit` with the author's file and line is the right call, so the two-line form carries the same `file:line` as an entry.
- Rebuttal: `Model.notes` is a member a generator can branch on, which the Q11.3 barrier exists to stop (panel-11/decision.md:7), and a runner that passes `-W ignore` asked for silence; under `-W error::taidl.TaidlWarning` the warning is raised inside `check`, which is the architect's explicit request and not a second failure path.
- Final: entries by declaration, then call order; lints and notes through `warn_explicit` as `taidl.TaidlWarning`; no `Model.notes`.
- Confidence medium-high; divides: Sonnet's second consumer, which no generator plan names.

### Q13a.9 (SF28) `python -m taidl check spec.py`
- Opposing (Opus): default `__name__` from `run_path`, collect `Accelerator` objects from the returned globals, three exit codes.
- Concession: the stem buys nothing once a `__main__` block is inert, scanning globals needs no hook in `Accelerator.__init__`, and a fifth exit code is a knob before a use; pytest's 5 was my only reason.
- Rebuttal to Sonnet: a trace-time error is the author's Python crashing (conformance.md:21), not a checked rejection, so it should not share code 1 with `Rejection`.
- Final: `runpy.run_path` with its default name; check every `Accelerator` bound in the returned globals, deduplicated by identity, in binding order; exit 0, 1 on any `Rejection`, 2 when nothing was checked (trace-time error, no accelerator, usage).
- Confidence medium-high; divides: an accelerator built inside a list or a function, which the globals scan misses and a message tells the author to bind.

### Q13a.10 (SF32) A kernel that needs an excluded instruction
- No opposition on the rule: reject at kernel compile time naming the uncovered operation; the hint lists excluded instructions whose tensor operators include it; Opus's point that "all operators occur in the kernel" holds for every instruction once a kernel is large settles the filter.
- Final: round-1 position, confidence medium-high.

### Q13a.11 (SF38) and Q13a.18 (LY-21) Frozen collections and metadata
- Opposing (Sonnet): return a fresh plain copy on every read, since `json.dumps` fails on a `MappingProxyType` and an Allo emitter serializes first.
- Rebuttal: the Python `json` page defines `default` as "a function that is called for objects that can't otherwise be serialized" and serializes a tuple as an array, so `json.dumps(entry, default=dict)` handles nested proxies in one keyword; copy-on-read makes `entry['x'] = 1` succeed on a throwaway, a silent no-op against engineering.md's fail-loudly rule, and breaks `is` between two reads of one node.
- Concession to Opus: the proxy must sit over a private deep copy made at the declaration call (generator_metadata.md:23), since a proxy alone is a live view.
- Final: deep copy at declaration; sequences as tuples, mappings as `MappingProxyType` over the copy, at every depth; the README states `default=dict` once.
- Confidence high; divides: Sonnet's serialization cost, which is one keyword.

### Q13a.12 (LY-03) The accepted link shapes
- Opposing (Opus and Sonnet): fold any tree linear in the attributes over `+`, `-`, unary `-`, and `*` by a plain `int` to coefficients, instead of an enumerated list of trees.
- Concession: panel 10 refuses a "nonaffine link" (panel-10/decision.md:20), and affine is a property of the function, not of the tree, so the fold is its definition; the normalizer panel 3 refused compared two trees for equality (panel-3/decision.md:92-93), while the fold reads one tree once and never feeds S6.1; the `Model` already stores the link as `(E, k, c, orientation)` (object_model.md:18, layers.md:37 of the gap report), so a generator never sees the spelling, which was my custom-code worry.
- Final: Opus's form, both sides folded to `a*E + b*D + c0 == 0` with `a` or `b` in `{1, -1}`, so `E + 1 == 8*D` and `(D + 1)*8 == E` are links and `2*E == 8*D`, `//`, `%`, a second attribute, and a link inside `any_of` are not; a fold that leaves signed 64 bits is rejected, and the printer shows the folded `E == k*D + c`.
- Confidence medium-high; divides: Sonnet's one-bare-attribute side, which rejects `E + 1 == 8*D` for no gain once both sides are folded.

### Q13a.13 (LY-05) Bare `Comp` and the declared type
- Opposing (Opus and Sonnet): panel 10 binds "a declared type whose width equals its slice total" (panel-10/decision.md:17) and reasons that a defaulted 64-bit attribute "names no rule for which bits enter a field" (:61), so an encoded attribute needs a written type and the `Model` records the distinction.
- Concession: :17 and :61 are binding text, the draft already carries "whether the type was declared" on the attribute node (object_model.md:19), and TableGen offers no precedent either way since `bits<n>` has no default to compare with; my reading dropped a panel decision for a saving of one member.
- Final: a bare `Comp` or `Addr` has type `s64` and no declared type; an encoded attribute with no declared type is rejected naming the attribute; the flag stays on the node as the draft has it.
- Confidence medium-high; divides: nothing, once the flag is read as a spelling fact whose one consumer is this row.

### Q13a.14 (LY-09) One word width per ISA specification
- No opposition; the TableGen point verified this round: `RVInst`, `RVInst48`, and `RVInst64` set `Size = 4`, `6`, `8` with `bits<32>`, `bits<48>`, `bits<64>` `Inst` fields, so the per-call `word` is the spelling a later variable-length lift keeps, and the equality check is the one line it deletes.
- Final: round-1 position, with Opus's note that `word` stays on `acc.encode` (panel-11/decision.md:25); confidence high.

### Q13a.15 (LY-10) What decode checks
- Opposing (Opus): decode also checks each link's exact division (S1.3.5) and each derived value against its type (S1.3.10).
- Concession: my "a field of width w holds exactly the type's range" is true of the encoded field and false of the derived value, since `D = (E - c) / k` can leave `D`'s declared type for an in-range `E`; decode is the only place that sees it, and a static range check at `check` is not forced because the encode direction can fault for some values without faulting always (Q13a.6's line).
- Rebuttal to Sonnet's "field ranges": an encoded field of width w under a type of width w has no out-of-range value, so that check is vacuous and should not be listed.
- Final: decode matches fixed bits, reads fields (two's complement for a signed type), applies each link with exact division, checks each derived value against its type, evaluates no other written constraint, and stops at the lowest failing word index.
- Confidence high; nothing divides beyond wording.

### Q13a.16 (LY-11) Words to bytes
- Opposing (Opus): the primary form is one hexadecimal word per line, and a byte form takes `byte_order` with no default, since a wrong order is a silent wrong result and no case study carries an encoding.
- Rebuttal: a required keyword with one value in practice is written `byte_order='little'` at every call forever; RISC-V fixed parcel order to little-endian regardless of data endianness (the lkml quotation both Opus and I cite), and every target in OD15's scope is little-endian, so the default is not a guess but the whole population; two assemblers for one ISA specification must emit one file, which only a normative default gives (Sonnet's point).
- Concession: the word sequence with a 0-based index is the primary form and the hexadecimal text per line is a free second output, since the message index is defined on words either way.
- Final: binary program as a word sequence indexed from 0, shown in hexadecimal in messages; serialization `byte_order` default `'little'`, `ceil(w/8)` bytes per word, high bits of the last byte zero; the default is language text, the option a generator keyword.
- Confidence medium-high; divides: Opus's no-default, which the first big-endian case study would settle.

### Q13a.17 (LY-16) Who checks fixed-bit disjointness
- No opposition on the rule; two evidence corrections: riscv-opcodes `process_standard_instructions` does check across instructions, exiting on `'Instruction {name} in extension {ext_name} overlaps with {key} in {item["extension"]}'` unless `warn_overlap` is set, so Sonnet's "only inside one instruction" is wrong and the source-of-truth tool checks disjointness itself; LLVM's `FilterChooser::doFilter()` prints "Decoding Conflict:" and `DecoderEmitter::run()` ends with `PrintFatalError("Decoding conflict encountered")`, so the static step stops the build before any disassembler exists.
- Final: `check` runs all three block checks; conflict is agreement on every bit both fix; layers.md:60 drops its second-check clause; confidence high.

### Q13a.19 (LY-26) The shape of a metadata entry
- No opposition: a `str`-keyed dict, possibly empty, rejected otherwise naming node and namespace; confidence high.

### Q13a.20 (LY-28) Metadata for `d0`
- No opposition: no entry on `d0`, hbm data in the accelerator's entry, no keyword; confidence high.

### Q13a.21 (LY-31) May a generator require metadata
- Opposing (Opus): a datum a generator cannot default is a `U` rejection naming node, namespace, and key, since adding a key is an architect correction under OD40.
- Rebuttal in part: for an instruction or buffer the S1.2 outcome for "valid, not modeled here" is exclusion (overview.md:27), the rest of the ISA specification still generates, and the expectations table records `excluded`; a whole-specification stop for one unannotated instruction fails more than the cause.
- Concession in part: a datum on the accelerator node has nothing to exclude, so there the generator rejects with `U` naming node, namespace, and key, as Opus says.
- Final: missing per-node metadata excludes that node with a reason naming node and namespace; missing accelerator-level metadata the generator cannot default rejects; neutrality compares files under the generator's directory and the printed exclusion list in the deterministic mode (OD27).
- Confidence medium-high; divides: Opus's blanket rejection, resolved by node level.

### Q13a.22 (LY-32) The entry point
- Opposing (Opus): return `None`, warn each exclusion, store the list in the output; positional-only `source`.
- Concession: positional-only (`source, /`) removes the parameter name from the contract and ends the `model` against `accelerator` split (generator_metadata.md:39, U11).
- Rebuttal: exclusions through `warnings.warn` collide with Q13a.8, since a CI runner under `-W error::taidl.TaidlWarning` then turns every expected exclusion into an exception while the expectations table lists them as `excluded`; the T12 runner asserts `code` and `object` per entry (panel-3/decision.md:136) and should not parse stderr or a generated crate to do it; the oracle has none to report, so its report is empty, not absent.
- Final: `generate_<name>(source, /, base_dir=None, **options)` writes under `<base_dir>/targets/<acc>/<name>/` and the paths its report lists, and returns a frozen report with `output_dir`, `excluded` (instruction, code, section id, file, line), and `namespaces_read`; exclusions also print at severity `warning` (scheme.md:21), as today.
- Confidence high on directory and report, medium on the printed line; divides: Opus's `None`, with the backend binary path for track B to state.

### Q13a.23 (LY-34) Scope letters for a new generator
- Opposing (Sonnet): a letter in `taidl/diagnostics.json` per generator touches `taidl`, against generator_metadata.md's rule for outside authors; use a shared scope `G` keyed by namespace with class `K`; (Opus) an outside generator needs no letter since the tool prefix names the raiser.
- Concession: an in-tree generator registers a letter, an outside one must not touch `taidl`, and a code still needs a scope letter in its five characters (scheme.md:9), so Sonnet's reserved `G` with the namespace qualifying the code is the hook that satisfies both.
- Rebuttal on class: `K` is "the input is invalid" for kernel text, calls, assembly, or binary (scheme.md:16), and an unknown metadata key sits in the ISA specification the architect corrects, which is the `U` rejection outcome (scheme.md:20-23) that the oracle already uses for a correctable problem.
- Final: in-tree generators register one letter before release; outside generators use scope `G`, codes keyed by namespace in their own registry file; an unknown key is class `U` with the rejection outcome.
- Confidence medium-high; divides: Sonnet's `K`.

### Q13a.24 (LY-35) `CONTRACT_VERSION` and a mismatch
- No opposition on the string, the field, the one-minor deprecation, or rejection over warning; Opus's class `X` for the mismatch (scheme.md:18, versions) is right and I adopt it; Sonnet's major-only check loses the lower minor bound, which keeps a generator from reading a member added after the `Model` it was given.
- Final: `major.minor`, `Model.contract_version`, generator states its major and lowest minor, rejects outside that window with an `X` error naming both versions; confidence high.

## Escalation to the owner

- None; Q13a.5 is a confirmation of a reading, and no position changes direction, scope, or an external commitment.
