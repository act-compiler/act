# Plan: optional layers and generator metadata

These items land the width layer, the encoding layer, the derived-attribute rule, and generator metadata (S11, S12, panel 10).
All land in v3: L37 first, then L38 and L39 with their consumers and the round-trip test as one unit that never ships in part; dates move, scope does not.
Sizes assume the shared 3-4 engineers; every item is additive, so an ISA specification without layers or metadata keeps its meaning.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| L37 | Width layer: a type per attribute, `s64` by default | L1, U2 | M | Q1 |
| L38 | Encoding layer: one encoding block per instruction | L37 | M | Q2 |
| L39 | Derived-attribute rule and the encode-decode round trip | L38 | M | Q2 |
| L40 | Generator metadata in the typed contract | U2, U3 | S | Q1 |
| L41 | Guide: adding a generator | L40 | S | Q2 |

## L37 Width layer

- What: an optional type per attribute, `Comp[t]` or `Addr[t]` with `t` from `u1` to `u63` and `s1` to `s64`, expanding to two range constraints each (S11.2, Q10.1).
- Changes: `taidl/instruction.py` takes the mapping and generalizes `INT64` (instruction.py:400) to the declared range; S1.3.10 names the type.
- Changes: the oracle faults where `warn_width` (act-oracle utils.py:268-275) only warns; B's allocator domain becomes the declared range, replacing `MAKE_INT_VAR(0, INT64_MAX)`.
- Breaking: no for an ISA specification without types; yes for B's lower bound 0, which decision 34 already overrules.
- Benefits: a field range has one spelling and one fault path shared by front end, oracle, and backend.
- Caveats: the type objects ship with U12; `u64` is rejected.
- Fail points: a domain the solver cannot hold; notice by a loud generation error, never a silent clamp.
- Order and dependencies: after L1 and U2; first of the layer items.
- Verification: T rows: a `u4` value of 16 faults in O; B never emits it; a key naming no attribute is rejected.

## L38 Encoding layer

- What: an optional `acc.encode(instr, word, fixed, fields)` block beside each instruction: word width, fixed bits, and ordered slices per encoded attribute (S11.3, Q10.2).
- Changes: `taidl` stores the block and `check` runs all three block checks (coverage, fixed-bit conflict, width per encoded attribute) and one `word` per ISA specification; a new assembler and disassembler generator reads it from the typed model and repeats none.
- Breaking: no, additive.
- Benefits: a binary form and its decoder come from the ISA specification, with no side file.
- Caveats: one fixed-length word per instruction; variable-length and multi-word encodings wait for a case study.
- Fail points: two instructions whose fixed bits agree on every position both fix; notice by the `check` conflict row.
- Order and dependencies: after L37; ships with L39, the generator, and the round-trip test as one unit; the generator's package home is a roadmap choice.
- Development: the encoding unit (L38, L39, B44, B45, T33, T34) is developed from wave 3 on the private heads and merges whole after the deletion pass (U14); the release gate (A31) and K20 are unchanged.
- Verification: every S9 layer row for the block rejects, naming the instruction and the bit.

## L39 Derived-attribute rule and round trip

- What: every attribute no slice names is derived, with exactly one link, an `==` under `require` that folds to `a*E + b*D + c0 == 0` with `a` or `b` in {1, -1}, stored as (E, k, c, orientation) (S11.4, Q10.3).
- Changes: `taidl` classifies attributes once a block exists and rejects each bad link naming the attribute; the generator encodes by the link and decodes by one subtraction and one exact division; B routes the link to the solver as a linear constraint.
- Breaking: no, additive; without a block nothing is derived.
- Benefits: decision 35's `A == B*8` addresses work with no solver in the assembler and no unencodable value from B.
- Caveats: chains, state in a link, and lookup links are rejected until a case study needs one.
- Fail points: `isel_ir2isa_rewrites_rs_generator.py:166-181` `attribute_factor` reads a link it does not invert; notice by the round-trip test.
- Order and dependencies: after L38, in the same unit.
- Development: with L38, from wave 3 on the private heads, merging whole after the deletion pass (U14).
- Verification: `decode(encode(x)) == x` on a corpus ISA specification with a small word, a scattered immediate, and an `A == B*8` link; a word matching no instruction gives S1.3.18.

## L40 Generator metadata in the typed contract

- What: a namespace-keyed dict of `str`-keyed dict entries on the accelerator, each buffer, and each instruction, deep-copied at declaration and read as a fresh copy on each typed node (S12, Q10.5).
- Changes: `Accelerator`, `acc.buffer`, and `@acc.instruction` take it as `meta=`; U2's typed nodes carry it under `CONTRACT_VERSION`; `check` (L43) lists the namespaces present as a note.
- Changes: the front end rejects only a non-identifier namespace or non-plain data, naming node and namespace; it has no registry of generators.
- Breaking: no, additive.
- Benefits: per-node data for one generator, such as an Allo partition hint, without a side file or a keyword that restates names.
- Caveats: a misspelled namespace is ignored by design and shows only in the listing; the member spelling follows panel 3.
- Fail points: a generator reads another's namespace; notice by the neutrality test.
- Order and dependencies: after U2 and U3; L41 builds on it.
- Verification: the neutrality test: for every corpus case with metadata, O and B output (files under the generator's directory and the printed exclusion list, in deterministic mode) is identical with it stripped.

## L41 Guide: adding a generator

- What: a documented extension path, in `taidl/docs/`, for a new generator such as act-allo (S12.4, decision 39).
- Steps the guide covers:
  - read the `Model` that `check` returns, and run its own checks as plain functions after it;
  - expose `generate_<name>(source, /, base_dir=None, **options)`, write under `<base_dir>/accelerators/<accelerator>/<name>/`, and return the frozen report;
  - state the supported `CONTRACT_VERSION` major and lowest minor;
  - own one metadata namespace, reject unknown keys in it, and report the namespaces read;
  - ship the expectations table with generated coverage (T11) and join the T12 runner;
  - follow the failure policy: reject only problems the architect can correct, with an error naming the clause.
- Changes: one guide page and a worked minimal generator; no change to `taidl` or another generator.
- Changes: the worked generator is the U11 template in `act/examples/generator_template/`; A27 refers to this guide for it.
- Breaking: no.
- Benefits: a new generator lands without touching `taidl`, the oracle, or the backend.
- Caveats: the guide follows U2's spelling and changes with `CONTRACT_VERSION`.
- Fail points: the worked generator drifts from the contract; notice by running it in CI against the corpus.
- Order and dependencies: after L40; T owns the CI run.
- Verification: the worked generator passes its expectations table and the neutrality test.
