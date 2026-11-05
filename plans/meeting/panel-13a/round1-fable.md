# Panel 13a, round 1: Fable, compiler infrastructure and assembler architect

Code paths are relative to /home/user; plan paths to /home/user/act/plans; spec pages are under language/spec-draft/.
Every owner decision and panel decisions 3, 10, 11, and 12 bind; questions follow the debate rows of reviews/spec-gaps/surface.md, then layers.md.

### Q13a.1 (SF4) NumPy integers at the surface
Position: the surface takes any object whose `operator.index` succeeds, except `bool`; the edge converts it to an exact `int`, and every later rule sees only `int`.
1. `operator.index(a)` is Python's own test for an integer, and since 3.10 its result has exact type `int` (docs.python.org/3/library/operator.html), so one edge call gives the "exactly `int`" the rows want without naming NumPy.
2. `numpy.int_` does not inherit from `int` (numpy.org/doc/stable/reference/arrays.scalars.html), so `type(x) is int` rejects every `np.arange` loop variable, the ordinary way to write a family of 64 members.
3. `float` has no `__index__`, so `n * 0.5` rejects as the finding wants; `bool` has one, so the rule names it as the one exclusion, matching conformance.md:33; today's front end tests only text against `INT64` (taidl/taidl/instruction.py:400-408), so no code constrains the choice.
Confidence medium-high; a corpus family where an `__index__` object other than a NumPy integer gave a wrong message would change it.

### Q13a.2 (SF8) What `@acc.instruction` returns
Position: the decorator returns an instruction handle, not the function; calling the handle raises at once naming S3.1; bare `@acc.instruction` means `@acc.instruction()`.
1. The body ran once at decoration (grammar.md:44), so a second call has no meaning the record can hold; a handle without `__call__` makes the call a `TypeError` in the author's frame, the S3.2 pattern (grammar.md:59).
2. `acc.encode(instr, ...)` takes "the instruction function" (layers.md:41) only because nothing else existed; a handle that is the instruction node lets `encode` match by identity, so a family loop that rebinds `copy` (grammar.md:51-55) still hands `encode` the right member.
3. `@dataclass`, `@dataclass()`, and the full call are documented as equivalent (docs.python.org/3/library/dataclasses.html); the same rule costs one `callable` test and closes the open case at semantics.md:18.
Confidence medium-high on the handle, medium on the bare form; a corpus file that calls one body from another to share a tensor expression would change it toward a helper function, still not a callable handle.

### Q13a.3 (SF11) Reuse of all-plain regions across traces
Position: a region whose starts and sizes are all plain belongs to no trace and may serve any instruction; every other symbolic object is bound to its trace, and its use elsewhere is rejected under S7.2; `read`, `write`, `require`, `update`, and `ops` outside a function raise at once.
1. JAX raises `UnexpectedTracerError` when a tracer leaks and is used in another trace, and finds it only at the later use (docs.jax.dev/en/latest/errors.html); a symbolic start carries an attribute of one function, so the same rule is forced here.
2. A region of plain values carries only a buffer and integers, so a module-level `TILE = d1.region([0], size=[64])` shared by a family is a constant, and rejecting it forces the loop body to repeat the call; its call site is the module line (object_model.md:16), which is the line the architect wrote.
3. Nothing in S5.2 (instructions.md:44-49) depends on where a region was made, and the tracer knows whether a trace is active, so the outside-a-function check is one flag.
Confidence medium-high; evidence that any S5.5 rule uses region identity would change it, since one object read in two instructions would then alias.

### Q13a.4 (SF12) Tracing after a collected problem
Position: a signature problem is reported once and the function is not run; any other collected problem returns a placeholder of the expected kind, and a problem whose only cause is a placeholder is not reported.
1. rustc's `TyKind::Error` exists for this: it is produced only after an error was reported and propagates so cascading errors are suppressed (rustc-dev-guide.rust-lang.org/ty.html).
2. Without a placeholder, `read` of a non-region must raise or return `None`; raising turns a collected problem into a trace-time stop, against conformance.md:17, and `None` makes the next `ops` call report a problem the architect did not make.
3. Tests assert `code` and `object` per entry (panel-3/decision.md:135-136), so the entry set must be defined, and "first problem per cause" is the only definition two front ends can meet.
Confidence high; a placeholder that cannot keep S4 shape inference total would change it to stopping the body at the first collected problem, still inside one `Rejection`.

### Q13a.5 (SF16) How OD21 holds on the Python surface
Position: a state variable is the object `acc.state` returns; the `Model` printer and every message spell its reference `@s.<name>`, and an attribute's `@a.<name>` or `@c.<name>`; the moderator forwards this as a confirmation of the reading, since no decision changes.
1. The prefixes are a 2.x text convention: `KINDS` maps `@a.`, `@c.`, `@s.` to kind names (taidl/taidl/instruction.py:397) and the docstring calls a state variable `@s.<name>` (taidl/taidl/accelerator.py:47).
2. The owner's paper writes control registers by bare name inside semantics, `assert(occupancy < 4)` and `occupancy = occupancy + 1` (MICRO 2025 paper, Fig. 6, p. 1321, act-compiler.github.io/assets/pdf/2025-micro-taidl.pdf), so bare names in the authored file are the owner's prior form.
3. OD41 moves semantics off strings and OD21 keeps `@s.`; both hold when `@s.` is the reference spelling of printed text and messages, which diagnostics/scheme.md:72 already orders as `state variable 'pc'`, and the printer is "what the front end understood" (object_model.md:22).
Confidence high; an owner statement that `@s.` must appear in the authored file would change it.

### Q13a.6 (SF21) Always-faulting plain cases
Position: reject them: a plain size below 1, a plain start plus plain size past a declared access dimension, and two output regions of one buffer whose start and size expressions all equal.
1. conformance.md:10 withholds rejection only for a case the front end "cannot decide from the recorded expressions"; plain integers against a plain dimension are decided, and the page already rejects a plain shape dimension of 0 (:66) by the same logic.
2. rustc's `unconditional_panic` lint denies, as an error by default, an operation that will always panic such as a constant division by zero (doc.rust-lang.org/rustc/lints/listing/deny-by-default.html); an instruction that faults at every issue is the same dead code, and OD40's genuine cause is a number the architect fixes.
3. The equal-expression output pair mirrors the `apart` row (conformance.md:47): the structural test that rejects a contradictory marker rejects a contradictory pair; `d0` has no declared bound (declarations.md:41), so the row never touches it.
Confidence high; nothing short of a corpus instruction that relies on a plain out-of-bounds region would change it.

### Q13a.7 (SF23) A plain `bool` in a constraint
Position: a plain `True` is accepted as a constraint that always holds; a plain `False` is rejected under S6.3 as a constraint no issue can satisfy; `any_of()` and `all_of()` with no condition are rejected.
1. `require(n <= 4)` in a family loop over plain `n` is the natural spelling (grammar.md:50-55); for the passing members a rejection has no genuine cause (conformance.md:8), so `True` must pass.
2. For `False` the member can never issue; the Q13a.6 rule and the `unconditional_panic` precedent (doc.rust-lang.org/rustc/lints/listing/deny-by-default.html) make that a rejection, and the fix is `if n <= 4:` around the member.
3. Dropping `True` from `all_of` and `False` from `any_of` is structural, not algebra, so it stays inside panel 3's no-normalizer ruling (panel-3/decision.md:92-95); `any_of()` and `all_of()` break expressions.md:34 and share one empty-argument message.
Confidence medium-high; a generator needing the always-true constraint for round-trip equality would change it to recording a `written` constraint with a plain `True` leaf.

### Q13a.8 (SF26) Entry order and the warning channel
Position: `Rejection` entries are ordered by declaration, then by call order within an instruction; a lint or note on success goes through `warnings.warn` with category `taidl.TaidlWarning` in the two-line format; the `Model` holds no warnings, only data such as the namespace list.
1. Python's filter grammar makes `-W error::taidl.TaidlWarning` turn one category into an exception and leave every other warning alone (docs.python.org/3/library/warnings.html), so CI gets a hard gate with no flag in `taidl`.
2. A generator reads only the `Model` (object_model.md:17) and S12.3 compares generator output; a warnings list on the `Model` is something a generator could branch on, which the Q11.3 type barrier exists to stop.
3. diagnostics/scheme.md:137 says v3 emits no warning for a spec condition while instructions.md:124 keeps the S5.5 lint; the category resolves it, a lint is advice without a code, and `python -m taidl check` prints it through the same filter.
Confidence medium-high on the channel, high on the order; a second consumer of lint output besides a person would change it toward a `notes` tuple on the `Model`.

### Q13a.9 (SF28) `python -m taidl check spec.py`
Position: run the file with `runpy.run_path` and `__name__` set to the file's stem; check every `Accelerator` built, in construction order; print entries and warnings on stderr; exit 0 on success, 1 on a `Rejection`, 2 on a trace-time error, 5 when no `Accelerator` was built.
1. `runpy.run_path(path, run_name=...)` sets `__name__` to `run_name` and `__file__` to the script (docs.python.org/3/library/runpy.html), so an `if __name__ == '__main__': generate_oracle(acc)` block stays inert and `base_dir` (declarations.md:19) still resolves from `__file__`.
2. Two accelerators in one file are two checks and one report; rejecting the file invents a rule S4 does not state, and a test file that shares buffers between two is a plausible author of two.
3. pytest separates "ran and failed" (1), "internal error" (3), "usage" (4), and "nothing collected" (5) (docs.pytest.org/en/stable/reference/exit-codes.html); a distinct code for "no accelerator" keeps a misnamed file from reading as a pass, and a trace-time traceback prints as is (diagnostics/scheme.md:88).
Confidence medium-high; a CI runner needing one non-zero code would change it to 1 for every failure with the kind in the text.

### Q13a.10 (SF32) A kernel that needs an excluded instruction
Position: the backend rejects at kernel compile time when extraction finds no covering; the error names the kernel operations left uncovered, then lists each excluded instruction whose tensor operators include one of them, as a hint.
1. The backend knows only "no candidate was extracted" or "every candidate was dropped, the first because ..." (act-backend/act_backend/generic/src/pipeline.rs:206-211); "needs an excluded instruction" is unobservable without a second search over the excluded set, which doubles compile time for a message.
2. LLVM's `Cannot select` fatal error names the DAG node with no pattern and, since a later change, the function (discourse.llvm.org/t/llvm-error-cannot-yet-select-0x1fcc5f0-f64-constantfp-0-000000e-00-id-4/20687); the uncovered node is the fact, the candidate pattern is the hint.
3. The exclusion list is stored in the generated crate (conformance.md:94), so the hint needs no new data; OD40 lets the backend stop on an input outside its specification, so the class is `U` with the rejection outcome (diagnostics/scheme.md:20).
Confidence medium-high; a measured search over the excluded set under a tenth of compile time would change it toward naming the exact instruction.

### Q13a.11 (SF38) Metadata after `check`
Position: the front end copies each entry at its declaration call and `check` exposes it read-only, dicts as `types.MappingProxyType` and lists as tuples; a generator sees tuples.
1. `MappingProxyType` is a read-only view with lookup, iteration, `get`, and `items` and no mutation (docs.python.org/3/library/types.html), so dict-shaped reads in a generator need no change, and a generator wanting a list calls `list()`.
2. Shallow freezing (object_model.md:9) is defeated by one `entry['x'].append(1)`; equality (object_model.md:23) and the U2 contract test then depend on call order.
3. The copy at the declaration call, not at `check`, keeps a family loop that mutates one dict (grammar.md:50-55) from publishing the last iteration's dict for every member.
Confidence high; nothing within the plain-data definition of S12.1 would change it.

### Q13a.12 (LY-03) The accepted link shapes
Position: a link is one top-level `require` condition with `==`, one side exactly one attribute, the other `k*X`, `X*k`, `X`, or `-X`, optionally with `+ c` or `- c` at the top, sides swappable, k and c plain `int`; nothing else, so neither `(X+1)*8` nor `X // 8`.
1. S6.1 compares trees structurally (expressions.md:18) and panel 3 ruled out a normalizer (panel-3/decision.md:92-95); the listed shapes are what a structural matcher handles by enumeration, and commutativity of `*` and `+` is the one symmetry it must enumerate.
2. RISC-V's scaled immediates are not in the bit placement at all: `bare_simm13_lsb0` uses `EncoderMethod getImmOpValueAsrN<1>` and `DecoderMethod decodeSImmOperandAndLslN<13, 1>`, custom C++ for a shift by one (llvm/lib/Target/RISCV/RISCVInstrInfo.td); the affine link is the declarative replacement, and widening it recreates custom code.
3. `X // 8` is no bijection, so decode would need the tie-break Q10.3 rejected (panel-10/decision.md:78); `(X+1)*8` equals `8*X + 8` only after algebra, and the message with the accepted list is cheaper than a normalizer no plan item holds.
Confidence high; a case study whose field needs a chain of two links would change it, and panel 10 refuses chains until one appears (panel-10/decision.md:87).

### Q13a.13 (LY-05) Bare `Comp` and the declared type
Position: a bare `Comp` or `Addr` is `Comp[s64]` or `Addr[s64]` in every respect; the `Model` records the type only, never whether it was spelled; the S11.3 row reduces to "an encoded attribute whose type width differs from its slice total", and conformance.md:84 drops its first clause.
1. Panel 10 makes `s64` the type of an untyped attribute (panel-10/decision.md:14), and TableGen's `bits<n>` is a width with no notion of declared versus defaulted (llvm.org/docs/TableGen/ProgRef.html); a width check is the whole check.
2. Panel 10's reason for a declared type was that a 64-bit default "names no rule for which bits enter a field" (panel-10/decision.md:61); with the width-equality check in force the rule is all 64 bits in two's complement, and a 64-bit field in a wider word is legal.
3. A `declared` flag is a spelling fact in the `Model`, which object_model.md keeps to what a generator needs, and U2.md:9 exposes the type and the default only; the practical mistake, a bare `Comp` under an 8-bit slice, is caught by the width check with the same message.
Confidence medium; a reading that panel 10's "declared type" wording binds the flag would change it.

### Q13a.14 (LY-09) One word width per ISA specification
Position: every encoding block of one ISA specification has the same `word`, or `check` rejects the second; an instruction with no block has no binary form, and the assembler and disassembler generator excludes it with a printed reason.
1. Mixed widths are variable length, which panel 10 deferred to a case study (panel-10/decision.md:63); TableGen carries per-instruction `Size = 4`, `6`, `8` on `RVInst`, `RVInst48`, `RVInst64` (llvm/lib/Target/RISCV/RISCVInstrFormats.td), so a later lift keeps the per-block spelling and deletes one check.
2. "No word matches the fixed bits of two instructions" (layers.md:53) has a meaning only for words of one width, so the equality check makes disjointness well-defined.
3. An instruction without a block in an ISA that has blocks is the S1.2 exclusion case (overview.md:27); rejecting a half-annotated ISA would make it unusable by the oracle too, against layers.md:4.
Confidence high; a case study in OD15's scope with two instruction lengths in one ISA would change it.

### Q13a.15 (LY-10) What decode checks
Position: encode takes an instruction and a value per attribute and returns one word; decode takes one word and returns the instruction and its attribute values by fixed bits, fields, and links only; it evaluates no other constraint; a failure stops at the lowest failing index.
1. LLVM's `MCDisassembler::getInstruction` returns `Success`, `SoftFail` for "disassemblable but invalid", or `Fail` (llvm.org/doxygen/classllvm_1_1MCDisassembler.html); decoding and architectural validity are two questions, and only the first belongs to a decoder.
2. A written constraint reads state before the issue (instructions.md:63), which a decoder of a static binary does not have; a decoded issue that breaks one faults when run (S1.3.5) at the same index with the same code, so nothing is lost.
3. Lowest failing index matches diagnostics/scheme.md:42, so the assembler, the disassembler, and the oracle report one position.
Confidence high; a type constraint failing on a decoded field would change it, and it cannot, since a field of width w holds exactly the type's range.

### Q13a.16 (LY-11) Words to bytes
Position: a binary program is a sequence of words indexed from 0, shown in hexadecimal in a message; serialization is a generator option `byte_order`, default `'little'`, with a word of w bits in `ceil(w/8)` bytes and the high bits of the last byte zero.
1. The ISA standard nearest this space stores instruction parcels little-endian regardless of the memory system's data endianness (RISC-V Unprivileged ISA, quoted at lkml.iu.edu/2202.3/03720.html), and every target in OD15's scope, x86 AMX, NVIDIA, and Trainium, is little-endian.
2. The spec fixes the word, the index, and the padding so two generators agree on S1.3.18 messages; byte order is a property of a file, so it is an option, not a layer (layers.md:17).
3. A w that is no multiple of 8 needs one padding rule, and zero high bits make the word recoverable by masking.
Confidence medium-high on the default, high on the index; a big-endian accelerator entering the corpus would change the default.

### Q13a.17 (LY-16) Who checks fixed-bit disjointness
Position: `check` runs all three block checks; two instructions conflict when their fixed bits agree on every bit position both fix; no generator repeats the check.
1. The check is decidable from the ISA specification alone, which is OD42's definition of what `check` does, and conformance.md:4 says the front end rejects exactly what S9 lists, so a row the front end does not check cannot sit in that table.
2. TableGen's decoder emitter prints "Decoding Conflict" and stops with `PrintFatalError("Decoding conflict encountered")` at table-generation time, before any disassembler exists (llvm/utils/TableGen/DecoderEmitter.cpp); the static step there is the static step here, `check`.
3. Panel 10 wrote "generation time" (panel-10/decision.md:17) before panel 11 made `check` a step of its own; panel 11's "`check` runs every front-end rule" (panel-11/decision.md:28) is the later binding text, and layers.md:50 already gives `check` the other two block checks.
Confidence medium-high; a moderator reading that panel 10's placement is an owner-level commitment would change it, and no owner decision names it.

### Q13a.18 (LY-21) Immutable collections in the `Model`
Position: as Q13a.11, every collection a node holds is a tuple or a `MappingProxyType`, and plain data read from a node follows the same rule.
1. Equality by value (object_model.md:23) and the frozen rule (:9) both need deep immutability; a tuple of plain data hashes and compares by value, a list does not hash.
2. `MappingProxyType` supports the whole read protocol (docs.python.org/3/library/types.html), so no generator loses a read; NaN is closed by "finite `float`" in S12.1 (LY-25).
Confidence high; nothing within plain data would change it.

### Q13a.19 (LY-26) The shape of a metadata entry
Position: `meta` is a dict from namespace to entry, and an entry is a `str`-keyed dict, possibly empty; another entry type is rejected by the front end, naming the node and the namespace.
1. S12.3 defines a generator's one check as "an unknown key in its own namespace" (generator_metadata.md:26); a key exists only in a dict, so the shape is already implied and the front end should state it.
2. MLIR attaches dialect data to an operation as an attribute dictionary, discardable attributes keyed `dialect.name` and verified by the named dialect (mlir.llvm.org/docs/LangRef/); the container shape is fixed by the IR and the values by the dialect, which is the S12.2 and S12.3 split.
3. A generator rejecting `5` as "unknown key" would name no key; one front-end row gives one message, and a scalar datum writes `{'value': 5}` at no cost.
Confidence high; nothing short of a generator whose entry must be a list would change it.

### Q13a.20 (LY-28) Metadata for `d0`
Position: `d0` carries no metadata entry; a datum about hbm goes in the accelerator's entry; no `Accelerator(d0_meta=)` keyword.
1. `d0` is predeclared (declarations.md:40) and `hbm` stays `hbm` (OD19); the accelerator node is the one node every generator reads, and an hbm hint is accelerator-wide by nature.
2. A keyword for one buffer is a knob with no use site; MLIR puts module-wide data on the module operation's attributes, not on a pseudo-operand (mlir.llvm.org/docs/LangRef/).
3. A later per-hbm need is one key under the generator's accelerator entry, which no spec change touches.
Confidence medium-high; a generator schema that must treat hbm like every other buffer node would change it toward `is_hbm` (LY-20) with an empty entry, still no keyword.

### Q13a.21 (LY-31) May a generator require metadata
Position: no; absent metadata means defaults; a generator that cannot default a datum for an instruction excludes it with a reason naming the node and the namespace; "output" in the neutrality test is the files under the generator's directory and the printed exclusion list, compared in the deterministic mode.
1. OD39 says metadata is data "which other generators ignore"; a generator that fails without its namespace makes the ISA specification invalid for a reason outside S4 to S11, which OD40 excludes as a rejection, and exclusion is the S1.2 outcome for "valid, not modeled here" (overview.md:27).
2. OD27 makes wall-clock output non-deterministic, so the neutrality comparison names the deterministic mode or the T35 run flakes.
3. MLIR's discardable attributes have "semantics defined externally" and are verified by the dialect that names them (mlir.llvm.org/docs/LangRef/); absence is no error there either.
Confidence medium-high; act-allo being unable to emit anything without a partition per buffer would not change it, since its expectations table then reads `excluded` per unannotated case.

### Q13a.22 (LY-32) The entry point
Position: `generate_<name>(source, base_dir=None, **options)`, `source` a `Model` or an `Accelerator`; it writes only under `<base_dir>/targets/<acc>/<name>/` and the paths its report lists; it returns a frozen report with `output_dir`, `excluded` (instruction, code, section id, file, line), and `namespaces_read`; found by import, no registry.
1. Both shipped generators write under `<base_dir>/targets/<acc>/oracle` and `.../backend` (act-oracle/taidl_to/generator.py:21-30, act-backend/act_backend/generator.py:79-93), so the subdirectory rule names today's layout; the backend's binary at `<base_dir>/backends/<acc>` (:83) is the one listed path.
2. Cargo keeps `target/<profile>/` and `target/<triple>/<profile>/` apart so builds of different configurations never collide (doc.rust-lang.org/cargo/reference/build-cache.html); one directory per generator under one accelerator is the same rule.
3. Both entries return `None` today (generator.py:21, :79) and the exclusion list exists only printed and in the crate (conformance.md:94), so the T12 runner must re-parse output to test it; `source` is the name because S12.4 says `model` and U11 says `accelerator`, and each is wrong for half the calls.
Confidence high on the directory and the report, medium on the parameter name; a caller that must keep positional `accelerator` for 2.x compatibility would change the name, and OD10 says no such caller exists.

### Q13a.23 (LY-34) Scope letters for a new generator
Position: a new generator registers one scope letter and its codes in the diagnostics registry before its release; an unknown key in its namespace is a `U` code in that scope with the rejection outcome, not `K`.
1. clang-tidy identifies a check's module by a registered prefix such as `bugprone-` or `modernize-` (clang.llvm.org/extra/clang-tidy/index.html); one registered letter per generator is the same scheme at this project's size, and twenty-six letters outlast the roadmap.
2. `K` is "the input is invalid" for kernels, calls, and assembly (diagnostics/scheme.md:16); a bad metadata key is a problem in the ISA specification the architect corrects, the `U` rejection outcome (scheme.md:20-23).
3. `R` has scope `F` only (scheme.md:28), so a generator cannot raise it, and `U` in its own scope keeps "two generators reject the same description with the same code" true for front-end rules alone.
Confidence high on the class, medium-high on registration; a third-party generator outside the `act` repositories would change it toward a reserved letter range by request.

### Q13a.24 (LY-35) `CONTRACT_VERSION` and a mismatch
Position: `CONTRACT_VERSION` is a string `major.minor`, also on the `Model` as `contract_version`; a minor adds members and keeps a removed one deprecated for one minor; a generator states the major it supports and the minimum minor it needs, and rejects a `Model` outside that window naming both versions.
1. Semantic versioning puts incompatible changes in MAJOR and additions in MINOR, and asks for one minor release carrying a deprecation before removal (semver.org); panel 3 already chose the one-minor deprecation rule (panel-3/decision.md:67).
2. A warning on mismatch is a silent fallback: the generator then reads a member that is absent or renamed and fails deeper with an `I` or an `AttributeError`, which engineering.md's fail-loudly rule and OD40's "clear error" both exclude.
3. A newer minor only adds, so it is accepted; an older minor than the generator needs lacks a member, so the lower bound is the generator's; the check is one comparison at the entry before any file is written, like the reserved-name rule (resolutions.md:37).
Confidence high; only an owner decision to pin generators by taidl version alone would change it, and OD33 pins releases, which is compatible.

## Escalation to the owner

- None; Q13a.5 asks only that the owner see the reading of OD21, and no position changes direction, scope, or an external commitment.
