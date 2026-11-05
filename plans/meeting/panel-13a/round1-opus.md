# Panel 13a round 1: Opus (language designer and front-end maintainer)

Plan paths are relative to /home/user/act/plans, spec pages sit under language/spec-draft/, and code paths are relative to /home/user.
Binding: owner decisions (OD), panels 3, 10, 11, and 12.

### Q13a.1 (SF4): NumPy integers
- Position: accept any non-`bool` object that `operator.index` accepts, converted to `int` at the call; the `Model` holds only `int`; `bool`, `float`, and objects without `__index__` are rejected.
1. Python's lossless integer protocol is `__index__`, used "whenever Python needs to losslessly convert the numeric object to an integer object (such as in slicing...)" (https://docs.python.org/3/reference/datamodel.html, object.__index__).
2. Slicing and `range` accept a NumPy integer by it, so one more rule than Python has surprises the architect; taidl needs no NumPy for it (taidl/pyproject.toml:15).
3. `bool` also has `__index__`, so it is excluded by name, as S4.4 already does for state (declarations.md:52); SF5 makes `__index__` on a symbolic value raise, so nothing symbolic is concretized.
- Confidence: medium. Change: an `__index__` type in common use that truncates.

### Q13a.2 (SF8): handle or function; bare form
- Position: the decorator returns an instruction handle, whose call raises at once (S3.1) and which exposes only its declared name; bare `@acc.instruction` means `@acc.instruction()`.
1. The body runs once at definition (grammar.md:44); calling the function again runs `read` and `write` outside a trace, whose error would name the wrong mistake.
2. `acc.encode` takes the instruction (layers.md:41), and a handle lets it check "an instruction of this `acc`" and pick one family member when a loop rebinds the name (grammar.md:50-55).
3. The prose already writes the bare form (grammar.md:44), and Python ships both forms: lru_cache "can be applied directly to a user function" (https://docs.python.org/3/library/functools.html, 3.8).
4. It is the decorator with no arguments, not a second spelling of an operation (panel-11/decision.md:93); without it the function lands in a keyword-only call and fails with a `TypeError` that names no fix.
- Confidence: high on the handle, medium on the bare form. Change: a generator or test that must call the original function after tracing.

### Q13a.3 (SF11): reuse of all-plain regions
- Position: allow it: a region whose starts and sizes are all plain belongs to no trace; any other symbolic object used in a second trace is rejected (S7.2); a recording call outside an instruction raises at once (S3.1).
1. An all-plain region is a value (buffer, plain starts, plain sizes) with no attribute in it, so it has no trace to leak from.
2. JAX draws the same line: a leaked tracer is "a JAX value that has leaked out of a function" (https://docs.jax.dev/en/latest/errors.html, UnexpectedTracerError), while a concrete array closed over is legal.
3. Each instruction still gets its own region node with the module-level call site (object_model.md:16), so `Model` equality and fault lines are unchanged.
- Confidence: medium-high. Change: an S5.5 rule that keys on region object identity across instructions.

### Q13a.4 (SF12): tracing after a collected problem
- Position: adopt SF12: a signature problem is reported once and the body is not run; any other collected problem returns a placeholder of the expected kind, and every check with a placeholder among its inputs is skipped.
1. Collect-all binds (panel-11/decision.md:28) and tests assert `code` and `object` (panel-3/decision.md:135-136), so the entry set must follow from the specification, not from recovery choices.
2. Clang added RecoveryExpr to keep invalid expressions in the AST and suppress follow-on errors through `containsErrors()` (https://reviews.llvm.org/D69330); a placeholder is that design for a tracer.
3. "Skip a check with a placeholder input" is decidable per call, so two front ends report equal sets; with `*args` no placeholder fits, since the tracer makes one value per declared parameter (S5.1).
4. A placeholder reaching `bool()` still raises at once (grammar.md:72), since that error sits on the author's line.
- Confidence: medium-high. Change: a corpus case where the skip hides a second genuine problem.

### Q13a.5 (SF16): reading of OD21
- Position: OD21 holds in every text form: the `Model` printer and every message spell a state reference `@s.<name>` and attributes `@a.` and `@c.`; the Python surface has objects and no prefix, as panel 11 binds (panel-11/decision.md:24).
1. OD21 keeps a reference prefix (minutes/owner-decisions.md:20); a prefix is lexical, and the v3 surface has no lexer (panel-11/decision.md:5).
2. The printer and messages are where an architect reads references as text (object_model.md:21-22; diagnostics/scheme.md:73 quotes verbatim), so `@s.` still tells state from attribute at a glance.
3. MLIR uses sigils the same way, in its textual form only: `%` for values, `@` for symbols (https://mlir.llvm.org/docs/LangRef/), while its builders carry objects.
- Confidence: high; no escalation, since no reading drops `@s.`. Change: an owner statement that OD21 meant a prefix inside the authoring surface.

### Q13a.6 (SF21): an always-faulting case rejects
- Position: yes; add both rows: a plain size below 1 or a plain start plus plain size past a declared access dimension, and two output regions of one buffer whose start and size expressions all equal (S5.2).
1. No symbolic branch exists (grammar.md:65), so every recorded region is reached on every issue: such an instruction can never run, which is an invalid specification under OD40.
2. conformance.md:66 already rejects a plain shape dimension of 0 for that reason, and both rows use only plain integers and equal expressions, the tools panel 3 allows (panel-3/decision.md:93).
3. Rust rejects code that "will always panic" by the deny-by-default lint `unconditional_panic` (https://doc.rust-lang.org/rustc/lints/listing/deny-by-default.html).
4. `d0` has no declared size (declarations.md:41), so the bound row never applies to it.
- Confidence: high. Change: an ISA that issues such an instruction on purpose as a trap.

### Q13a.7 (SF23): plain `True` and `False` in `require`
- Position: a plain `True` is a condition that always holds, recorded as written; a plain `False` is rejected at `check` (S6.3), by Q13a.6; `any_of()` or `all_of()` with no argument is rejected.
1. A family loop variable is plain (grammar.md:47), so `require(n <= 4)` reaches `require` as a `bool`; rejecting it as "not a condition" (conformance.md:57) fails a valid specification, against OD40.
2. A plain `False` makes every issue fault (S1.3.5), the always-faulting case of Q13a.6.
3. C++ treats a condition known at compile time the same way: a false `static_assert` is a compile error (https://en.cppreference.com/w/cpp/language/static_assert).
- Confidence: high on `True`, medium-high on `False`. Change: a family idiom that marks a member unissuable on purpose.

### Q13a.8 (SF26): entry order and the warning channel
- Position: entries go by declaration, then call order; a lint warning and the namespace note go through `warnings.warn` as `taidl.TaidlWarning`, in the two-line scheme form at severity `warning`; no `Model.notes`.
1. Warnings serve a condition that "(normally) doesn't warrant raising an exception", and a category subclass can be filtered or made an error (https://docs.python.org/3/library/warnings.html), so CI uses `-W error::taidl.TaidlWarning` with no new flag, and `pytest.warns` tests it.
2. A `Model.notes` member falls under U2's deprecation rule (shared/plan/U2.md:5) with no generator reading it, while S4.6 holds what generators consume (object_model.md:18); the namespace list stays a `Model` member (LY-29).
3. diagnostics/scheme.md:12-18 has no front-end warning class, so the lint's code class is a diagnostics-track item.
- Confidence: medium-high. Change: a generator that must act on a lint.

### Q13a.9 (SF28): the check command and two accelerators
- Position: run the file with `runpy.run_path` and its default `__name__`, check every `Accelerator` bound in the returned globals in binding order (deduplicated by identity), print entries on stderr prefixed by the accelerator name, exit 0, 1 on any `Rejection`, 2 on a trace-time error or none found.
1. `run_path` sets `__name__` to `'<run_path>'` and returns "the resulting module's globals dictionary" (https://docs.python.org/3/library/runpy.html), so `__main__` blocks skip and taidl keeps no registry.
2. Tutorials call `generate_oracle(qkv)` unguarded at module level (act/tutorials/pldi26/exercise1/solution/QKV.py:109), so v3 tutorials put it under `if __name__ == '__main__':`, or `check` would generate.
3. Two accelerators in one file are legal Python, such as two variants; checking all keeps OD42 without a selection flag, a knob before a use; an accelerator built only inside a function gets exit 2 with a message to bind it at module level.
- Confidence: medium. Change: a corpus file that builds accelerators in a loop or a factory.

### Q13a.10 (SF32): a kernel that needs an excluded instruction
- Position: reject a kernel the usable instructions cannot cover, naming the kernel operation with no candidate, and list the excluded instructions whose tensor operators include that operation.
1. "Needs" is undecidable without selecting with instructions the backend cannot model, and the exclusion reasons (state, two outputs; conformance.md:95) are what its e-graph cannot represent, so a rerun is unavailable.
2. Today the failure prints "Could not find an ASM representation." and the first drop reason, with no operation (act-backend/act_backend/generic/src/pipeline.rs:204-209).
3. LLVM's selector fails by naming the node: "Cannot select: <node>" (llvm-project, llvm/lib/CodeGen/SelectionDAG/SelectionDAGISel.cpp, `CannotYetSelect`); SF32's "all operators occur in the kernel" holds for every instruction once a kernel is large.
- Confidence: medium; panel 4's owner sizes finding the uncovered operation. Change: a cheaper exact test the extractor already has.

### Q13a.11 (SF38): metadata frozen, tuples visible
- Position: yes: `check` deep-copies each entry; the `Model` gives lists as tuples and dicts as `types.MappingProxyType` over the private copy; generators see tuples.
1. A MappingProxyType is "a dynamic view on the mapping's entries" that reflects later changes (https://docs.python.org/3/library/types.html), so a proxy without a copy is not frozen.
2. object_model.md:9 promises nothing changes after `check`, yet a list shared by a family loop (grammar.md:50-55) would change two nodes at once.
3. Python has no frozen list, so a tuple is the one standard immutable sequence; the only visible change is `(1,) != [1]`, stated once in the README.
- Confidence: high. Change: a generator that needs in-place edits, which its own copy already serves.

### Q13a.12 (LY-03): which link spellings count
- Position: define a link by its linear form: an `==` directly under `require`, both sides built from attributes, plain `int`, `+`, `-`, unary `-`, and `*` with a plain factor, collecting to `a*E + b*D + c0 == 0` over one encoded E and one derived D; it is a link when `a` or `b` is 1 or -1.
1. A list of trees leaves `c + k*D`, `(D + 1)*8`, and `8*(D - 1) + 4` to each front end; a linear form gives one answer for every tree.
2. MLIR defines affine expressions by the operations allowed (sum, difference, multiplication by a constant), not by tree shape (https://mlir.llvm.org/docs/Dialects/Affine/).
3. Collecting coefficients over `+`, `-`, and constant `*` is a small fold over one condition, not the dimension normalizer panel 3 refused (panel-3/decision.md:92-99); the `Model` stores (E, k, c, orientation) (LY-18), so panel 10's two forms stay the only meanings (panel-10/decision.md:19).
4. When both are 1 or -1 the orientation is E = k*D + c; `//`, `%`, and a link inside `any_of` stay non-links.
- Confidence: medium. Change: a reading that panel 10 binds a syntactic form.

### Q13a.13 (LY-05): bare `Comp` against `Comp[s64]`
- Position: bare `Comp` or `Addr` has type `s64` and no declared type; `Comp[s64]` declares it; the `Model` records which; an encoded attribute needs a declared type.
1. Panel 10 binds "each encoded attribute has a declared type whose width equals its slice total" (panel-10/decision.md:17), which presumes the distinction.
2. OD38 has an author "add" the width layer; writing the type is that act, and a reader of the `Model` must tell layer 0 from layer 1, while every rule of S5 to S8 reads `s64` either way (layers.md:15).
3. Python keeps the same distinction: an unannotated parameter has `inspect.Parameter.empty`, not `Any` (https://docs.python.org/3/library/inspect.html).
- Confidence: medium-high. Change: proof that no consumer needs the flag beyond the width check.

### Q13a.14 (LY-09): one word width
- Position: one width per ISA specification: every block's `word` is equal, else rejected (S11.3); `word` stays on `acc.encode` as panel 11 spells it; the assembler generator excludes a block-less instruction with a printed reason.
1. Panel 10 rules fixed length and one word and waits on a case study for variable length (panel-10/decision.md:18, :63); mixed widths are variable length.
2. A stream of mixed words needs a length rule before matching, as RISC-V marks length in the low bits of the first parcel (https://cs107e.github.io/readings/riscv-spec-20191213.pdf, section 1.5); the block has none.
3. A per-call `word` checked equal lets variable length land by lifting one check, with no surface change; exclusion follows Q11.3 (panel-11/decision.md:7).
- Confidence: high. Change: a case-study ISA with mixed widths in v3 scope.

### Q13a.15 (LY-10): what decode checks
- Position: decode checks what it computes: fixed bits, each link's exact division (S1.3.5), and each derived value's type (S1.3.10); it evaluates no other written constraint and stops at the lowest failing word.
1. Other constraints may read state and evaluated addresses (S5.3), which a disassembler lacks; the oracle checks them at issue (overview.md:35).
2. LLVM separates "the instruction was invalid" (Fail) from "valid but architecturally incorrect" (SoftFail) (https://llvm.org/doxygen/classllvm_1_1MCDisassembler.html); a word that breaks a run-time constraint is the second kind.
3. An encoded value fits its type by construction, since width equals slice total (layers.md:54); the lowest index matches diagnostics/scheme.md:42.
- Confidence: medium-high. Change: a user who needs offline full validation, served by an option, not by decode.

### Q13a.16 (LY-11): byte order
- Position: the primary form is words, one hexadecimal word per line, which needs no byte order; a byte form takes `byte_order` as a required entry-point keyword with no default; position is the word index from 0.
1. A wrong byte order yields wrong words with no error, the silent wrong result engineering.md forbids; a missing keyword fails loudly at the call.
2. Practice moves even within one ISA: RISC-V parcel order once followed memory endianness and is now fixed little-endian (https://lkml.iu.edu/2202.3/04391.html).
3. No case study carries an encoding (panel-10/decision.md:111), so a default is a guess; it is a run option, never metadata (generator_metadata.md:13).
- Confidence: medium. Change: a case-study ISA with a documented byte order.

### Q13a.17 (LY-16): who checks fixed-bit disjointness
- Position: `check` runs all three block checks and no generator repeats one; two instructions conflict when their fixed bits agree on every position both fix.
1. Disjointness follows from the ISA specification alone, and OD42 has `check` validate it "on its own, without any generator".
2. Panel 10's "block checks at generation time" covers all three (panel-10/decision.md:17), yet the draft puts two in `check` (layers.md:50); the phrase sets them apart from declaration time, and `check` precedes every generator.
3. It is architect-correctable (OD40), so one `RF` code serves every reader (diagnostics/scheme.md:14, :28); a generator check leaves an invalid specification that passes `check`.
- Confidence: high. Change: panel 10 text that assigns the check to the assembler generator by name.

### Q13a.18 (LY-21): frozen collections
- Position: as Q13a.11, for every collection a node holds (attribute lists, slices, fixed bits, metadata): tuples for sequences, read-only mappings over a private copy.
1. "Frozen" (object_model.md:9) must hold at every depth, or equality (object_model.md:23) changes after `check`; LY-25's finite-float rule keeps equality reflexive.
- Confidence: high. Change: as Q13a.11.

### Q13a.19 (LY-26): an entry is a dict
- Position: the front end requires each entry to be a `str`-keyed dict, possibly empty, and rejects another type naming node and namespace (S12.2).
1. S12.3 rejects "an unknown key in its own namespace" (generator_metadata.md:26), which presumes keys; a type test per generator repeats one rule.
2. `CONTRACT_VERSION` covers the container (generator_metadata.md:21), and the top-level shape belongs to it.
3. pyproject's tool table is the same design: each tool owns `tool.$NAME`, always a table (https://peps.python.org/pep-0518/).
- Confidence: high. Change: a generator whose datum cannot sit under a key.

### Q13a.20 (LY-28): metadata for `d0`
- Position: `d0` carries none in v3; hbm data goes in the accelerator's entry; `Accelerator(d0_meta=)` waits for a use.
1. No plan names an hbm datum; the one example is a partition hint on an on-chip buffer (generator_metadata.md:14).
2. A keyword added later is a minor contract change (shared/plan/U2.md:5), so waiting breaks nothing, and engineering.md adds abstraction at the second use.
3. MLIR keeps whole-program data as discardable attributes on the module, owned by the dialect that prefixes them (https://mlir.llvm.org/docs/DefiningDialects/).
- Confidence: medium-high. Change: a generator plan with a per-hbm datum distinct from accelerator data.

### Q13a.21 (LY-31): may a generator require metadata
- Position: yes, for its own namespace: a datum it cannot default is a rejection by that generator (`U`, rejection outcome) naming node, namespace, and key; stripping and output as LY-31 defines them.
1. OD39 makes metadata optional to the language and ignored by others; OD40 rejects what the architect can correct, and adding a key is a correction (panel-11/decision.md:7).
2. MLIR leaves a dialect's attributes to that dialect's verifier, `verifyOperationAttribute`, which may fail (https://mlir.llvm.org/docs/DefiningDialects/).
3. Neutrality concerns the other generators (generator_metadata.md:28), so a requiring generator never breaks it, and its expectations table records the case (generator_metadata.md:37).
- Confidence: medium. Change: an owner reading of OD39 that every generator runs on every specification without metadata.

### Q13a.22 (LY-32): entry point, output place, return value
- Position: `generate_<name>(source, /, base_dir=None, **options)`, `source` a `Model` or an `Accelerator`; it writes only under `<base_dir>/targets/<accelerator>/<name>/`, returns `None`, warns each exclusion (Q13a.8), and stores the list in its output; no registry.
1. Positional-only removes the parameter name from the contract, ending the `model` against `accelerator` split (generator_metadata.md:34; shared/plan/U11.md:4).
2. Both generators already write there (act-oracle/taidl_to/generator.py:31-33; act-backend/act_backend/generator.py:95-96); the backend's binary at `<base_dir>/backends/<name>` (act-backend/act_backend/generator.py:91) is the one exception for track B to state or move.
3. Both return `None` today (act-oracle/taidl_to/generator.py:21; act-backend/act_backend/generator.py:79-81); a report type adds a member for one use, and the umbrella only wraps in-tree generators (act/act/generators/__init__.py:4-15).
- Confidence: medium. Change: a T11 runner that cannot read exclusions without a return value.

### Q13a.23 (LY-34): scope letters
- Position: a generator the act project ships registers one scope letter before its release, the assembler first; an outside generator needs no letter, since the tool prefix names the raiser; an unknown key is class `U`, rejection outcome.
1. Letters today are F, B, O, A in an in-tree registry (diagnostics/scheme.md:26), and an outside author must touch neither `taidl` nor another generator (generator_metadata.md:38).
2. PEP 518 avoids name collisions by tying `tool.$NAME` to a name the tool already owns (https://peps.python.org/pep-0518/); the tool prefix (scheme.md:66) plays that role.
3. `R` is front end only and `K` is run input (scheme.md:14, :16, :28); the oracle's correctable rejection already takes `U` (scheme.md:82), and shared faults stay `VF` (scheme.md:29).
- Confidence: medium. Change: a plan to list outside generators in the act umbrella.

### Q13a.24 (LY-35): contract version mismatch
- Position: `CONTRACT_VERSION` is a `major.minor` string, also `Model.contract_version`; a generator states its major and lowest minor and rejects any other `Model` with an `X` error naming both; no warning path.
1. Semantic versioning puts "incompatible API changes" in the major and backward compatible additions in the minor (https://semver.org/), which matches U2's per-member rule (shared/plan/U2.md:5).
2. A warning lets a generator read a renamed member and emit a wrong tool, the silent wrong result engineering.md forbids; versions are class `X` (diagnostics/scheme.md:18).
3. OD33 pins released versions, so the check fires only in development installs and costs one comparison.
- Confidence: high. Change: a deprecation window that spans a major.

## Escalation to the owner

- None.
