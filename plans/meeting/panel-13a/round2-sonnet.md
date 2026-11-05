# Panel 13a round 2: Sonnet

Lens: the architect's authoring experience and few dependencies (od:32 OD38, od:33 OD39, od:35 OD41).
Abbreviations as in round 1; f1 = round1-fable.md, o1 = round1-opus.md, p10 and p11 = panel-10 and panel-11 decision.md.
Checked this round: pipeline.rs:204-211, scheme.md:8-30, p10:15-19, p11:26-30, and a Python run for the mapping proxy.
Per question: strongest opposing view, answer, final, confidence, what still divides.
"All agree" means all three round-1 files hold the same position, which I adopt.

## Undisputed questions

- Q13a.1 (SF4): All agree: `operator.index` at the edge, exact `int` stored, `bool` and `float` rejected; Fable adds that a NumPy scalar is no `int` subclass. Confidence medium-high.
- Q13a.2 (SF8): All agree on the handle and the bare form; I adopt Fable's reason that `encode` matches the handle by identity, so a loop rebinding the name still works. Confidence high on the handle, medium on bare.
- Q13a.3 (SF11): All agree; I adopt Fable and Opus: a recording call outside an instruction raises at once. Confidence medium-high.
- Q13a.5 (SF16): All agree: printer and messages spell `@s.`, `@a.`, `@c.`; Python objects carry no prefix; the moderator forwards it as a reading. Confidence high.
- Q13a.7 (SF23): All agree: plain `True` accepted, plain `False` and empty `any_of`/`all_of` rejected. Confidence medium-high.
- Q13a.14 (LY-09): All agree: one `word` per ISA specification; a block-less instruction is excluded with a printed reason. Confidence high.
- Q13a.15 (LY-10): All agree: decode checks fixed bits, fields, links, and derived types, never a state-reading constraint; it stops at the lowest failing index. Confidence high.
- Q13a.17 (LY-16): All agree: `check` runs all three block checks; Fable's point that p11:28 postdates "generation time" in p10:17 is the cleanest citation. Confidence medium-high.
- Q13a.19 (LY-26): All agree: an entry is a `str`-keyed dict, possibly empty, checked by the front end. Confidence high.
- Q13a.20 (LY-28): All agree: no `d0` metadata and no `d0_meta` keyword. Confidence high.

## Disputed, surface

### Q13a.4 (SF12)
- Opposing (Fable, Opus): p11:28 says `check` collects every problem, so stopping a body at its first problem hides a second error and costs one run per fix.
- Concede: my stop rule breaks p11:28 inside a body, and I had no evidence that bodies rarely hold two errors.
- Of the two placeholder rules I take Opus's, skip any check with a placeholder input; Fable's "only cause is a placeholder" needs a cause definition per S9 row, the gap I raised.
- Final: signature problem reported once, body not run; other problems return a placeholder of the expected kind; checks with a placeholder input are skipped.
- Confidence: medium-high. Divides: Fable's cause wording against Opus's input wording.

### Q13a.6 (SF21)
- Opposing: Fable and Opus omit a plain negative start; OD34 (od:30) applies the implicit constraint to the evaluated address, not to attributes.
- Answer: a plain start is already an evaluated address, so a negative one faults at every issue; I keep it as a row.
- Final: reject plain size below 1, plain start below 0, plain start plus size past a declared dimension, and an equal-expression output pair; skip `d0`.
- Confidence: medium-high. Divides: whether the negative-start row is worth having.

### Q13a.8 (SF26)
- Opposing (Fable, Opus): a `Model.notes` member joins the U2 deprecation rule with no generator reading it, and the Q11.3 barrier exists so a generator cannot branch on lints.
- Concede: my only argument for `notes` was `-W ignore` hiding warnings, which is the architect's own filter; a second surface for one use runs against OD41.
- Final: `taidl.TaidlWarning` via `warnings.warn_explicit` at the author's line; no `Model.notes`; entries ordered by declaration, then call order.
- Confidence: medium-high. Divides: nothing.

### Q13a.9 (SF28)
- Opposing (Opus): use the default `run_name` and scan the returned globals for `Accelerator` objects, with no constructor hook.
- Concede: `runpy.run_path` already keeps `__main__` blocks inert by default, so my file-stem rule adds a rule for no gain.
- Concede: "every Accelerator built" needs a hook in the constructor; my evidence of several constructions was act-oracle/tests/test_kernel.py, whose constructions sit in functions, so no spec file is shown to need it.
- Keep: any spec problem (rejection, trace error, exception from the file) exits 1; no file, no accelerator, or bad arguments exits 2; Fable's code 5 splits a case no CI script treats differently.
- Final: globals scan in binding order, deduplicated by identity; exits 0, 1, 2; v3 examples guard generator calls under `__main__`.
- Confidence: medium. Divides: a trace-time error as 1 (me) or 2 (Fable, Opus).

### Q13a.10 (SF32)
- Opposing (Opus): my filter "all of the instruction's operators occur in the kernel" admits nearly every instruction in a large kernel, so the hint is noise.
- Opposing (Fable, Opus): name the uncovered operation and filter by it.
- Reply: pipeline.rs:206-209 shows the backend knows only "no candidate was extracted" or the first drop reason, never which operation lacked a candidate; naming it is new extractor work.
- Concede the precision point: filter by the uncovered operation if extraction can name it; otherwise give the kernel and the exclusion count, without the list.
- Final: advisory hint outside the match key; class `U`, rejection outcome (scheme.md:20).
- Confidence: medium. Divides: whether extraction can name the uncovered operation today.

### Q13a.11 (SF38)
- Opposing (Fable, Opus): tuples and `types.MappingProxyType` freeze the `Model` at every depth, so equality (object_model.md:9, :23) cannot drift.
- Reply, from a run this round: `json.dumps` of a `MappingProxyType` raises TypeError and `copy.deepcopy` of one raises "cannot pickle".
- An Allo emitter serializes metadata first, so each generator author would write a converter; tuples alone do serialize, as arrays.
- Answer: store the entry privately and immutably as they want, and return fresh plain dicts and lists on read; nothing mutable is reachable, and equality compares stored data.
- Final: see Q13a.18.
- Confidence: medium. Divides: read-only views against fresh plain copies.

## Disputed, layers

### Q13a.12 (LY-03)
- Opposing (Fable): accept an enumerated list only (`k*X`, `X*k`, `X`, `-X`, optional `+ c`), since p3:92-95 refused a normalizer and `(X+1)*8` equals `8*X+8` only by algebra.
- Reply: that list already enumerates commutativity of `*` and `+`, so it is a fold written out by hand; the recursive fold reads each node once and compares no two trees.
- Authoring cost: OD35 (od:31) gives `B*8` as its example, and `(B + 1) * 8` is how an architect writes a biased scale; refusing it sends the architect to hand algebra.
- Opus's `a*E + b*D + c0 == 0` also lets both attributes share a side, wider than the two forms of p10:19; I keep one bare attribute on one side.
- Final: tree linear in one attribute, folded to (k, c); sides swap; `//`, `%`, a second attribute, and links in `any_of` are not links.
- Confidence: medium. Divides: enumerated list (Fable), fold with a bare side (me), full linear form (Opus).

### Q13a.13 (LY-05)
- Opposing (Fable): bare `Comp` is `Comp[s64]` in every respect, the `Model` stores the type only, and the width check is the whole check.
- Concede on authoring grounds: OD38 makes 64-bit the default, so a bare attribute in a 64-bit slice is no mistake and forcing `Comp[s64]` there is friction; a bare one under an 8-bit slice fails the width check with the same message.
- Residual risk: p10:17 says "declared type"; the moderator reads that word, and p10:61 gives a reason the width check also meets.
- Final: bare equals `s64`, no flag on the `Model`; the S11.3 row is "type width differs from slice total".
- Confidence: medium. Divides: the weight of "declared" in p10:17.

### Q13a.16 (LY-11)
- Opposing (Opus): `byte_order` is a required keyword with no default, since a wrong default yields a silently wrong file, and hex words need no byte order.
- Reply: a required keyword puts a knob on every call that most architects never turn (OD41); a default written once in the language keeps two assemblers in agreement.
- Concede: the language form of a program is the word sequence in hex with a 0-based index, and bytes are a second form.
- I did not verify Fable's claim that every OD15 target is little-endian, nor either side's RISC-V citation.
- Final: little-endian default, `ceil(w/8)` bytes, high bits of the last byte zero, `byte_order` overrides.
- Confidence: medium. Divides: default little-endian against no default (Opus).

### Q13a.18 (LY-21)
- Opposing (Fable, Opus): tuples and read-only mappings at every depth.
- Answer: tuples for node collections (attributes, slices, fixed bits) as they say; the only split is metadata, as in Q13a.11.
- Final: tuples for node collections; metadata stored privately and immutably, read as fresh plain data.
- Confidence: medium. Divides: as Q13a.11.

### Q13a.21 (LY-31)
- Opposing (Opus): a generator may reject for a missing datum of its own namespace, since adding a key is something the architect can correct (OD40) and neutrality concerns only other generators.
- Reply: OD39 says metadata is optional; a whole-specification reject makes it required for that generator and withholds output for every instruction that needed none.
- OD40 sets the backend precedent: drop the candidate with a recorded reason, stop when nothing usable is left (p11:30 rejects an empty usable set).
- Concede: if exclusion leaves no instruction, the generator rejects, as the backend does.
- Final: exclusion naming node and namespace; empty usable set rejects; neutrality compared in deterministic mode (OD27).
- Confidence: medium-high. Divides: reject (Opus) against exclude (Fable, me).

### Q13a.22 (LY-32)
- Opposing (Opus): return `None`, since a `Report` type adds a member for one use and the umbrella wraps only in-tree generators.
- Reply: the T12 runner needs exclusions by `code` and `object` (p3:135-136), and the other routes are parsing stdout or a per-generator file.
- Cost to an outside author is one frozen object with `excluded` and `namespaces_read`, imported from `taidl`, which a generator already depends on for `Model`.
- Concede: positional-only `source` (Opus) ends the `model` against `accelerator` naming split; I adopt it.
- Final: `generate_<name>(source, /, base_dir=None, **options) -> Report`; writes only under `<base_dir>/targets/<accelerator>/<name>/`; no registry.
- Confidence: medium. Divides: `Report` against `None`.

### Q13a.23 (LY-34)
- Opposing (Fable): every generator registers one scope letter before release, as clang-tidy modules register a prefix.
- Reply: OD39 asks that new generators be easy to add, and generator_metadata.md:38 says an outside author touches neither `taidl` nor another generator; a letter in `taidl/diagnostics.json` breaks that.
- Concede: a generator the project ships registers its own letter (the assembler first), and an unknown key is class `U` with rejection outcome, not `K`, since scheme.md:20 and Q11.3 put architect-correctable problems there.
- Final: shipped generators register a letter; an outside generator uses scope `G` qualified by its namespace, class `U` for an unknown key.
- Confidence: low-medium. Divides: a letter for all (Fable), none for outsiders (Opus), `G` (me).

### Q13a.24 (LY-35)
- Opposing: Fable and Opus add a minimum minor that a generator needs; my "a newer minor proceeds" omitted the floor.
- Concede: an older minor lacks a member, so the generator states its major and minimum minor.
- Final: `major.minor` string, also `Model.contract_version`; reject outside the window naming both versions; no warn mode.
- Confidence: high. Divides: nothing.

## Escalation to the owner
- Q13a.5: confirm that OD21 holds as the printed spelling with bare names in the Python objects.
- Q13a.16 is withdrawn as an owner question: no decision fixes byte order, and a language default is reversible, so the moderator decides.
