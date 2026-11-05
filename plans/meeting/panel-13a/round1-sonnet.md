# Panel 13a round 1: Sonnet

Lens: architect writing an ISA specification, and a third-party generator author who reads only the `Model`.
Abbreviations: spec = plans/language/spec-draft; od = plans/minutes/owner-decisions.md; p3, p10, p11, p12 = plans/meeting/panel-N/decision.md; sf = plans/reviews/spec-gaps/surface.md; ly = plans/reviews/spec-gaps/layers.md.
Not retrieved: the RISC-V unprivileged manual (HTTP 404); no claim below rests on it.
Numbering follows the agenda: surface rows Q13a.1 to Q13a.11, layer rows Q13a.12 to Q13a.24.

## Surface

### Q13a.1 (SF4): NumPy integers
- Position: accept any non-bool object with `__index__` wherever a plain `int` is taken, store it as exact `int`; reject `bool` (except `ops.constant` at `pred`) and `float`.
1. od:34 (OD40) allows rejection only for a genuine cause; an `np.int64` from `for n in np.arange(1, 9)` is an integer with one meaning.
2. `operator.index` returns exact `int` since 3.10 and floats lack `__index__` (docs.python.org/3/library/operator.html), so one call decides.
3. `bool` has `__index__`, so the front end tests `bool` first; this keeps conformance.md:29 and :33 true after normalization ("exactly `int`" describes the stored value).
- Confidence: medium. Change my mind: a case where `__index__` admits a type with a different integer meaning (a symbolic NumPy object).

### Q13a.2 (SF8): what the decorator returns
- Position: an instruction handle, a small frozen object `acc.encode` takes; calling it raises at once naming S3.1; bare `@acc.instruction` equals `@acc.instruction()`.
1. layers.md:37 and :41 pass the decorated name to `acc.encode`, so the name must bind an object other than a function that re-traces.
2. Triton's `@triton.jit` returns a `JITFunction`, a handle and not the raw function (triton-lang.org/main/python-api/generated/triton.jit.html).
3. Bare use passes the function as the first argument (`name`) of `acc.instruction`; the decorator must detect a callable anyway, so accepting costs the same as a coded rejection.
- Confidence: high on the handle, medium on the bare form. Change my mind: a v3 corpus file where bare use hides a missing `name=`.

### Q13a.3 (SF11): reuse across traces
- Position: the finding's text; a region whose starts and sizes are all plain belongs to no trace; a symbolic object from instruction I is rejected in instruction J.
1. A module-level `d1.region([0], size=[16])` named once and read by ten instructions is natural; forbidding it forces copy and paste with no correctness gain.
2. An all-plain region carries no attribute or state leaf, so it can mean nothing different in another trace.
3. JAX raises when a traced value is used in the wrong context and passes concrete Python values (docs.jax.dev/en/latest/errors.html), the same split.
- Confidence: medium. Change my mind: a marker or call-site rule (S4.6) that breaks when one region object serves two instructions.

### Q13a.4 (SF12): collected problems and placeholders
- Position: a signature problem is reported once and the body is not run; the first collected problem inside a body ends that body's trace; later instructions and declarations still run; no placeholder values.
1. conformance.md:4 says "rejects exactly"; a test asserts `code` and `object` (p3:135-136), so the entry set must be derivable by rule, not by tracer luck.
2. The placeholder rule "a problem whose only cause is a placeholder is not reported" needs a definition of cause for every S8 row; two front ends will differ on `ops.add(bad, x)`.
3. Clang keeps an invalid node in its AST for recovery, at the cost of a large internals contract (clang.llvm.org/docs/InternalsManual.html, Error Handling).
- Confidence: medium. Change my mind: evidence that real specifications hold several independent errors inside one body, or a table that fixes the placeholder rule per S8 row.

### Q13a.5 (SF16): the `@s.` prefix under OD21
- Position: the finding's text; the `Model` holds bare names; the printer and every message spell `@s.<name>`, `@a.<name>`, `@c.<name>` through one function; the owner confirms this reading of OD21.
1. od:20 says `@s.` stays, and od:30 says attribute names are plain variable names, so the prefix cannot sit in the name.
2. A Python object cannot carry a sigil; MLIR's textual form uses `%` and `@` while its objects are plain (mlir.llvm.org/docs/LangRef/, from memory, not quoted).
- Confidence: high on the text; the reading of OD21 is the owner's. Change my mind: the owner wants the prefix visible in Python.

### Q13a.6 (SF21): always-faulting regions
- Position: reject at `check` four always-faulting cases: a plain size below 1, a plain start below 0, a plain start plus size past a declared access dimension, and two outputs of one buffer whose start and size expressions all equal.
1. Each faults at every issue (overview.md:44-45), so no run of the instruction can succeed; the architect can correct it (od:34).
2. Rust denies `unconditional_panic` by default for the same kind of case (doc.rust-lang.org/rustc/lints/listing/deny-by-default.html).
3. It decides only from plain integers and structural equality, so it obeys SF18 and p3:92-95; `d0` is skipped because its bound is run-time (declarations.md:41).
- Confidence: medium-high. Change my mind: a case study where a never-issuable instruction is intentional.

### Q13a.7 (SF23): plain `bool` conditions
- Position: a plain `bool` is a constant condition; `True` is accepted and dropped, a plain `False` in a conjunction rejects (message names Python `if`), `any_of()` or `all_of()` with no condition rejects.
1. `require(n <= 4)` in a family evaluates in Python before the tracer sees it (grammar.md:47); rejecting `True` has no cause (od:34).
2. `False` never holds, the same always-faulting class as Q13a.6, with the same Rust precedent.
- Confidence: medium. Change my mind: a family idiom that uses `require(False)` to switch members off.

### Q13a.8 (SF26): lint channel
- Position: both: `warnings.warn_explicit` with `taidl.TaidlWarning` at the author's file and line, and a tuple `Model.notes` of structured entries.
1. `Model.notes` gives a generator author data without parsing stderr; warnings alone are lost when a runner filters them.
2. `warnings.warn` alone is invisible to a runner that sets `-W ignore`; the default action prints once per module and line (docs.python.org/3/library/warnings.html), which fits author call sites.
3. `-W error` turns a lint into an exception at the architect's request; `check` then raises it and stops, which is that request's meaning.
- Confidence: medium. Change my mind: a decision that `check` never raises anything but `Rejection`.

### Q13a.9 (SF28): the check command
- Position: run the file with `__name__` set to the file stem; check every `Accelerator` built, in creation order; exit 0 all valid, 1 any rejection, trace-time error, or exception from the file, 2 no file, no `Accelerator`, or bad arguments.
1. Today's examples call generators at module level with no guard (act-backend/examples/mini_trn/mini_trn.py:112-115); `__name__ == '__main__'` would run them, so the stem keeps guarded calls quiet (`runpy.run_path` sets `__name__` from `run_name`, docs.python.org/3/library/runpy.html).
2. Unguarded generator calls still run, so the v3 corpus guards them under `if __name__ == '__main__':`.
3. Several accelerators per file occur (act-oracle/tests/test_kernel.py holds 6 constructions), so each is checked and reported alone.
4. Exit codes follow pytest: 0 pass, 1 failure, 4 usage (docs.pytest.org/en/stable/reference/exit-codes.html); a spec problem is 1 whichever way it surfaces, so CI separates invalid from misused.
- Confidence: medium. Change my mind: CI users who need rejection and trace error apart.

### Q13a.10 (SF32): kernel not covered
- Position: the finding's text, advisory: the error carries its code and the kernel as object, then lists each excluded instruction whose tensor operators all occur in the kernel, each with its recorded reason; the list is a hint and not part of the match key.
1. Knowing a kernel is uncoverable needs the search; the hint needs only the exclusion list and the kernel's operators, both deterministic (od:26, OD27).
2. Tests assert `code` and `object` (p3:135-136), so a hint cannot make two backends differ on conformance.
3. Compilers usually name the failing item and not the alternatives: GlobalISel calls an unlegalizable instruction "unsupported" (llvm.org/docs/GlobalISel/Legalizer.html); the hint is extra help for an architect who excluded something.
- Confidence: medium-high. Change my mind: a case study where the hint misleads more than it helps.

### Q13a.11 (SF38): metadata mutability
- Position: same rule as Q13a.18; confidence medium.

## Layers

### Q13a.12 (LY-03): link shapes
- Position: accept any tree that is linear in the one attribute X, folded by a small recursive function to (k, c); sides may swap; the other side must be exactly one bare attribute.
1. Linear means built from X, plain `int`, `+`, `-`, unary `-`, and `*` with a plain factor; `//`, `%`, a second attribute, and a link inside `any_of` are not links.
2. MLIR affine expressions allow exactly this closed set (mlir.llvm.org/docs/Dialects/Affine/), so `(B + 1) * 8` is legal and `B // 8` is not.
3. The fold is not a normalizer (p3:99): it never compares two trees, it reads (k, c) once, and S6.1 structural equality stays.
4. A generator never parses the shape; the `Model` stores (encoded attribute, k, c, orientation) (ly LY-18), so generosity costs the front end only.
- Confidence: medium. Change my mind: a defect found in the fold on negative k or on overflow near signed 64 bits.

### Q13a.13 (LY-05): declared type
- Position: a declared type is one written `Comp[t]` or `Addr[t]`, `s64` included; a bare kind has type `s64` and no declared type; the node stores `declared_type` (or `None`), and `type` returns the effective type; an encoded attribute without a declared type rejects.
1. p10:17 and :61 require a declared type whose width equals the slice total, because a default carries no rule for which bits enter a field.
2. `Comp[s64]` in a 64-bit field is then a visible choice by the architect.
3. The `Model` must record the distinction; U2 exposes only the default (shared/plan/U2.md:9), so a generator cannot tell.
- Confidence: high. Change my mind: the owner wants bare to mean "64 bits, encodable".

### Q13a.14 (LY-09): word width across instructions
- Position: all blocks of one ISA specification share one `word`; a different one rejects; an instruction without a block has no binary form and the assembler generator excludes it with a printed reason.
1. p10:63 defers variable-length encodings until a case study needs them; mixed `word` smuggles them in.
2. "A word that matches two instructions" (layers.md:53) needs one width to be well defined.
3. A disassembler for fixed width is a total function on one word; for mixed width it needs a length rule the spec lacks.
- Confidence: high. The RISC-V length encoding could not be fetched; this is the only unverified point. Change my mind: a named case study with two word widths.

### Q13a.15 (LY-10): decode and constraints
- Position: decode matches fixed bits, reads fields (two's complement for signed), applies links, and checks field ranges; it evaluates no other written constraint; failure stops at the lowest failing index.
1. A written constraint may read state (expressions.md:14, S5.3), which a binary file does not hold, so decode cannot evaluate it.
2. The two outcomes stay: no matching instruction is S1.3.18; a link with no integer partner is S1.3.5 (layers.md:68, :70).
3. riscv-opcodes validates bit placement per instruction and nothing semantic (github.com/riscv/riscv-opcodes, README): the same division of work.
- Confidence: high on the state argument. Change my mind: a use case that decodes with state supplied.

### Q13a.16 (LY-11): word sequence and bytes
- Position: the language defines a binary program as a sequence of `word`-bit unsigned words with a 0-based word index and hex in messages; it also fixes the default serialization (little-endian, `ceil(w/8)` bytes, high bits of the last byte zero); an assembler option `byte_order` overrides it.
1. If the default lived only in each generator, two assemblers could emit incompatible files from one ISA specification; a normative default prevents that.
2. An option is still needed for ISAs that are big-endian in memory.
- Confidence: medium. External source not retrieved (manual 404); this rests on the interoperability argument only. Change my mind: a decision that byte order is an `Accelerator` keyword.

### Q13a.17 (LY-16): where disjointness runs
- Position: `check` runs all three block checks; no generator repeats one; two instructions conflict when their fixed bits agree on every position both fix.
1. The check decides from the ISA specification alone, and OD42 gives `check` exactly such checks (od:36).
2. A conflict makes the specification ambiguous for every binary reader; leaving it to the one reader (conformance.md:85) makes each later reader repeat it, the duplication U2 exists to remove (shared/plan/U2.md:4).
3. riscv-opcodes checks duplicate bits only inside one instruction (README), so a cross-instruction check is not free in practice; it must be in the contract.
- Confidence: medium-high. Change my mind: p10 meant "only by the assembler" on a record I did not see.

### Q13a.18 (LY-21): frozen `Model`
- Position: node collections are tuples; metadata entries are deep-copied at declaration and every read returns a fresh plain copy (dicts and lists), so the `Model` holds no reachable mutable object.
1. `MappingProxyType` is a view that reflects later changes to the dict under it (docs.python.org/3/library/types.html), so it still needs a private copy.
2. Proxies and tuples break `json.dumps` of a metadata entry, which is the first thing an Allo emitter does (from memory, not retrieved).
3. Copy-on-read keeps entries plain, so a generator author can pass them on or serialize them.
4. This resolves SF38 the same way: a generator sees lists and dicts, never the author's objects.
- Confidence: medium. Change my mind: profiling shows read copies matter for large tables.

### Q13a.19 (LY-26): entry shape
- Position: `meta` is a dict from namespace to entry; the front end rejects an entry that is not a `str`-keyed dict, naming node and namespace.
1. S12.3 rejects "an unknown key in its own namespace", which presupposes keys (generator_metadata.md:26).
2. Checking dict-ness is not reading contents (generator_metadata.md:19), so S12.2 stands.
3. MLIR keeps the top-level attribute dictionary as the one structured container and lets nested values be anything (mlir.llvm.org/docs/LangRef/); this is the same two-level shape.
- Confidence: medium-high. Change my mind: a generator that needs scalar entries.

### Q13a.20 (LY-28): hbm metadata
- Position: `d0` carries no metadata; data about hbm goes in the accelerator's entry under the generator's schema; no `d0_meta` keyword.
1. d0 is predeclared (declarations.md:40), so no call site exists for `meta=`.
2. The accelerator entry already reaches every generator (generator_metadata.md:8).
3. A keyword for one future use is a knob before a use site; p10 reasoning 8 promotes shared data when a second generator reads it.
- Confidence: high. Change my mind: two generators both needing per-hbm data.

### Q13a.21 (LY-31): requiring metadata
- Position: a generator may need its own metadata only per instruction or buffer and answers a missing datum with an exclusion naming node and namespace, never a whole-specification reject; neutrality is tested by stripping one namespace from every node and comparing the other generators' files and exclusion lists in deterministic mode.
1. od:33 (OD39) says metadata is optional and other generators ignore it.
2. A protoc plugin returns one `error` for a whole request (protobuf plugin.proto), too coarse for a per-instruction datum.
3. Deterministic mode follows OD27 (od:26): a wall-clock search cannot be neutral.
- Confidence: medium-high. Change my mind: a generator whose output is meaningless without a datum.

### Q13a.22 (LY-32): entry-point contract
- Position: `generate_<name>(source, base_dir=None, **options) -> Report`; it writes only under `<base_dir>/targets/<accelerator>/<name>/`; `Report` is a frozen type in the `taidl` contract with `excluded` and `namespaces_read`; no registry and no umbrella listing.
1. Today the oracle writes `<base_dir>/targets/<name>/oracle` (act-oracle/taidl_to/generator.py:21-30) and the backend `<base_dir>/backends/<name>` (act-backend/act_backend/generator.py:79-86); both return `None`.
2. The finding's `<base_dir>/<name>/` collides when one file builds two accelerators (Q13a.9); the accelerator name belongs in the path.
3. A returned report lets the T12 runner assert exclusions by `code` and `object` without parsing stdout (p3:135-136).
4. A report type in `taidl` is part of the contract (U2), not a generator touching `taidl`.
5. The umbrella wrapper (act/act/generators/__init__.py:4-15) is a convenience; U10 rules out a registry (shared/plan/U10.md:5); protoc plugins likewise return files relative to an output directory (plugin.proto).
- Confidence: medium. Change my mind: the owner wants the v2 `backends/` layout kept.

### Q13a.23 (LY-34): codes for outside generators
- Position: the four existing scope letters stay with their packages; any other generator uses scope `G`, keyed by (namespace, code), with its codes in its own registry file in the registry schema; an unknown key in its namespace is class `K` in that scope.
1. A letter registered in `taidl/diagnostics.json` (p3:163) means a `taidl` change per new generator, against "touches neither taidl nor another generator" (generator_metadata.md:38).
2. A code is never reused (diagnostics/scheme.md:11-12); the namespace qualifier keeps two outside generators from reusing one.
3. `R` is front end only (scheme.md:21), so an unknown key cannot be `R`; K means invalid input to a generator.
- Confidence: low-medium. Change my mind: the owner prefers a closed registry with one-line pull requests.

### Q13a.24 (LY-35): `CONTRACT_VERSION`
- Position: a string `major.minor`, also `Model.contract_version`; a minor release adds members and deprecates for one minor; a generator states the major it supports and rejects another, naming both; a newer minor proceeds.
1. semver.org: minor for backward-compatible additions and deprecations, major for removals; p3 Q3.4 and U2 already say "one minor release".
2. OD26 and OD33 pin released versions (od:25, :43), so a mismatch means a wrong install, which a generator cannot correct; reject.
3. p12 says no `Model` is serialized, so the check is in-process at the entry; the field on the `Model` costs one string.
- Confidence: medium-high. Change my mind: an independent generator release cadence that makes a warn mode useful.

## Escalation to the owner
- Q13a.5: confirm that OD21 holds as the printed spelling `@s.<name>` with bare names in the Python objects.
- Q13a.16: confirm that the byte-order default belongs in the language and not in each generator.
