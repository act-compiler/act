# Panel 3 round 2: Opus (front-end maintainer)

Paths are relative to /home/user for code and to /home/user/act/plans for plans.
Owner decisions 10, 11, 14, 16, 19, 20 bind.

### Q3.1

- Strongest opposing argument (Fable, with Sonnet): one item under S4, sized L, with the move and the R31 checks as two ordered PRs, gives the same sequencing without a new cross-item dependency the audit already flags.
- Concession: the shared rule I cited ("structural prep as its own commit, then the behavioral delta") is about commits, and two PRs inside one item meet it.
- My two-item split adds an index row and a dependency arrow and buys nothing a reviewer can use.
- What I keep from round 1: the move is a port, not a copy, because `declared()` reads parse-tree methods (`act-oracle/taidl_to/generic/shapes.py:31-35` calls `.shape().dim()` and `dim.INT()`).
- That port is what keeps the item at L: about 620 lines of rule code (`shapes.py` plus `act-oracle/taidl_to/oracle_visitor.py:19-97`) re-pointed at the typed node model of Q3.4, then the R31 checks.
- All three agree: B23 is dropped, D4 attaches codes only, L3 stays the spec text.
- Final position: option B, one S4 item sized L, What lines list the R31 checks, PR 1 the move with no verdict change, PR 2 the new checks.
- Confidence: medium-high.
- Changed: yes, from C to B.
- Still divides the panel: nothing material.

### Q3.2

- All three agree on option A: B21 to B23 become "call `Accelerator.validate()` first and delete `spec_checks.py:394-418`", kept as a scheduled, tested backend task, with the literal check untouched until S3 ships (R25).
- Verified: the existing check skips any non-literal size (`act-backend/act_backend/spec_checks.py:413-414`), so it needs no new code to cover R25.
- Final position: option A, one S-sized backend item that depends on S3.
- Confidence: high.
- Changed: no.

### Q3.3

- All three agree on option C: L15 moves to Q1, L16 is already Q1 (`language/plan/frontend.md:16`), D10 drops L28, S2 precedes B13 and drops it.
- Sonnet's low confidence came from unread sizes; Fable and I both cite L15 as S and L28 as after v3 (`language/plan/terminology.md:13`).
- Fable's refinement is compatible with mine: D10 owns the banned-word list file, and L28 consumes it after v3.
- Final position: option C as above, with D10 owning the word list.
- Confidence: medium-high.
- Changed: no.

### Q3.4

- All three agree on A now plus D's trigger: typed Python model, small typed node model, trees reachable but unstable, `CONTRACT_VERSION`, per-member deprecation, second non-Python consumer as the trigger for a serialized form.
- Fable's point that today's surface includes undocumented members (`comp_attr`, `inplace`, `placed`, the magic `d0`, per `shared/interface.md`) strengthens the case against B.
- Final position: option A with D's trigger.
- Confidence: high.
- Changed: no.

### Q3.5

- All three agree on option A; Fable adds that `fold` and `literal_fits` stay exported functions.
- I agree: the facts are the contract, the functions are importable but are not the way a generator is expected to get facts.
- One addition from Q3.6: each line's computed dims are published as folded integer expressions over `@c.`, not only as integers, so the backend can render an agreement check (see Q3.6).
- Fable's lazy-computation fallback (same member names, computed on first read) is a sound answer if construction time ever matters; it changes no member.
- Final position: option A, members for result type and computed dims per line, folded sizes, value-to-region map, call site; `fold` and `literal_fits` exported.
- Confidence: medium-high.
- Changed: no, refined.

### Q3.6

- Strongest opposing argument (Fable): option C alone leaves the backend outcome of X1 in place, which `shared/bugs.md` X1 states is a rewrite that never matches or a panic in `merge` (`act-backend/act_backend/generic/src/ir/egraph.rs:403-409`), and decision 16 forbids a panic as an outcome.
- Concession: Fable is right on the backend; I treated the backend side as a non-match question for panel 4, but X1 names a panic, which is neither a defined error nor a defined behavior.
- Verified: the backend renders a shape check per assignment from `shape_vec` with `lets` binding `@c.` values (`isel_ir2isa_rewrites_rs_generator.py:593-603`), and sets the root shape from the declared type (`:702-708`), so `@c.` values are bound where a check could run.
- How the backend gets the check cheaply: if Q3.5 publishes each line's computed dims as folded expressions, the backend renders "declared dim equals computed dim" as one more integer comparison under the same `lets`, and raises the shared shape fault code (Q3.11) on mismatch.
- That needs no shape rules ported to Rust, only two expressions the front end already holds.
- Rebuttal on the front end: Fable's canonical-form comparison is acceptable only as far as `fold` goes; equal folded text agrees, two literals that differ reject, everything else is compatible.
- A front end that rejects "provable" symbolic disagreement needs an affine-equality procedure, which no item sizes, and `agree()` today does no folding (`act-oracle/taidl_to/generic/shapes.py:58-62`).
- Sonnet's identical-text step is the same rule before folding; the folded form is a strict improvement at no extra cost once S6 lands.
- Final position: C plus a narrow B: front end rejects literal mismatch and accepts equal folded forms, oracle keeps the call-time fault, backend renders a bound agreement check from published dims and raises the same code; no symbolic normalizer.
- Confidence: medium.
- Changed: yes, from C alone to C plus a narrow B.
- Still divides the panel: whether the front end goes beyond fold (Fable's "provable disagreement"), and whether the backend check is an error or a non-match (Sonnet's point 3).

### Q3.7

- All three agree on option A: run-time import at the pinned tag, with `taidl.shapes` and `taidl.diagnostics` importable without ANTLR.
- Concession to Fable: my round-1 claim that `shapes.py` is parser-free "by construction" holds only for its imports; `declared()` duck-types a parse tree (`act-oracle/taidl_to/generic/shapes.py:31-35`).
- So the rules must take the typed node model or plain `(type, dims)` tuples, which is the same port Q3.1 already sizes.
- The test I proposed (import `taidl.shapes`, assert `antlr4` not in `sys.modules`) still guards the property.
- Final position: option A, lazy package init (PEP 562), rule inputs are typed nodes or tuples, never trees.
- Confidence: high.
- Changed: no, refined.

### Q3.8

- Strongest opposing argument (Sonnet, with Fable): line 1 is the line that logs, CI summaries, `grep` and `str(error)` keep, so a spec id on line 3 is lost in every one-line channel; Clang puts the rule tag at the tail of line 1 (`[-Wunused-variable]`).
- My round-1 argument was that the code determines the id through the registry, so the suffix repeats a fact.
- Rebuttal fails on use: a reader of a CI log has the message, not the registry, and the suffix turns the code into a spec section with no lookup.
- The cost is a few characters against the 160-character rule (`diagnostics/scheme.md:50`), which a section id such as `[S9.4]` does not threaten.
- Concession: placement A, suffix on line 1, line 3 lists further sections only.
- All three agree on namespace D, with S9 rows numbered as subsections so E is a case of D, and F (a generator's `coverage.md` id) never in an `R` or fault message.
- Fable's note on decision 14 is adopted: docs say "section id", since "clause" has no precise use in the owner's papers as cited.
- Final position: placement A, namespace D; `TaidlError.spec` carries the same id as a field.
- Confidence: medium.
- Changed: yes, placement from B to A.
- Still divides the panel: nothing; a residual value judgment for the owner only if R44's form is reopened.

### Q3.9

- All three agree on match key D and renderer E.
- Fable's refinement is accepted: expectation and corpus rows carry `code` and section id, tests assert `code` and `object`.
- Fable's scope note is also accepted: D11 keeps rendering S9 (`taidl/docs/conformance.md`) in taidl, which is a different page; only the per-generator `coverage.md` gets one renderer, owned by T11.
- Final position: D for keys, E for `coverage.md`, D11 keeps S9.
- Confidence: high.
- Changed: no, refined.

### Q3.10

- Strongest opposing argument (Sonnet): an author should meet one name in prose, the exception class and the code letter; a class called Fault with letter `V` is a second word to learn, and the seed is unreleased, so a letter change costs nothing now.
- Rebuttal: the message form is `<subject> violates <rule>` (`diagnostics/scheme.md:57`), so `V` names the violated rule and Fault names the outcome; the letter and the message agree.
- Sonnet's collision example (`FF001`) arises only if the class letter becomes `F`; with `V` kept, a shared fault code reads `VF001`, which is unambiguous.
- Renaming to a third letter (for example `T` for trap) is free before release but buys only mnemonic fit, and every plan, seed row and E13 citation would change.
- Agreement across all three: class name Fault, no-write rule stated in the norm and in D8's tests, front-end findings are rejections, and the `Fault` alias at `taidl/taidl/operations.py:167-168` is renamed.
- Fable's evidence that the norm states an existing invariant is accepted: the oracle's `check_shapes` raises before any write (`act-oracle/taidl_to/generic/utils.py:771-786`).
- Final position: option A with `V` kept; alias renamed to `Rejection`.
- Confidence: medium.
- Changed: no.
- Still divides the panel: the class letter (`V` kept by Fable and me, renamed by Sonnet); this is a naming value judgment.

### Q3.11

- All three agree on option A: fault codes take scope `F` in `taidl/diagnostics.json`.
- Open sub-point between Sonnet and Fable: whether `VB001` is the shared 64-bit fault raised early (Sonnet) or a backend `U` row (Fable).
- Verified: `diagnostics/registry-seed.md:94` states `VB001` as "a value computed from attributes is decidable at generation", while `VO005` (`:91`) is the 64-bit range rule, and E13 lists both for the 64-bit case.
- These are two rules: out-of-range is the shared fault (one `F` code, raised at generation by the backend when it can decide it), and "not decidable at generation" is a backend limit, a `U` row.
- Final position: option A; the 64-bit rule is one shared `VF` code; `VB001` becomes a backend `U` row for undecidable values.
- Confidence: high.
- Changed: no, refined.

### Q3.12

- Strongest opposing argument (Sonnet): a coded replacement message needs stub methods kept in the API, which the shared rule against "just in case" placeholders forbids, and a release note plus the existing rejection serves the description writer.
- Rebuttal: no stub is needed; one table of removed names read by a class-level `__getattr__` on the builder classes raises one code naming the replacement, and the same table serves word-level removals at the `OPERATIONS` lookup (`taidl/taidl/operations.py:145` is where `cholesky` sits today).
- A table that drives a user-facing error is a feature with a test, not a placeholder; the shared rule targets dead re-exports and `// removed` comments.
- A generic rejection sends the user from the error to a release note to the fix; decision 16 asks that a rejection tie to a defined rule, and the replacement's section id does that.
- Against Fable's one code per removed construct: one code with the table keeps the registry from growing a row per removal, and each message still carries the replacement's section id.
- All three agree on E: RF013 to RF017 are not seeded, and the seed of moving rules waits for the v3 API.
- Final position: A and E, one `RF` code backed by a removed-name table; no converter; no stubs.
- Confidence: medium.
- Changed: no.
- Still divides the panel: Sonnet prefers C (generic rejection plus S8 note); Fable prefers a code per removed construct.

## Escalation to the owner

None.
