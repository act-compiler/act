# Rulings in force

Each row states a rule that a term row in `terms.md` cannot carry.
Owner decisions 14, 19, 20, 21, and 32 settle every disagreement between sources: the owner's papers first when precise, otherwise StableHLO or MLIR.
Renames land together in the v3 release.

## Scope of the bans

| Topic | Ruling | Reason |
| --- | --- | --- |
| Generator concepts | The bans cover TAIDL concepts only; a rewrite condition may be called a "precondition", and the backend's own docs may say "global memory offset" for its placement offsets. | An XLA-HLO tensor and a rewrite-pattern variable are not TAIDL concepts. |
| Vendor words | "fragment", "shared memory", and "device memory" appear only in a vendor quote; in a TAIDL sentence, shared memory and register fragments are buffers and device memory is `hbm`. | One vocabulary covers tensor accelerators and GPU tensor-core instructions (Decision 15). |
| Hardware words | "scratchpad" and "register file" name hardware; the buffer that models one is a buffer. | The papers use "scratchpad" for the hardware only. |
| Papers' own words | A quotation or a citation of a paper keeps the paper's word, with the glossary term beside it. | Source column of the term rows. |

## Names

| Topic | Ruling | Reason |
| --- | --- | --- |
| The file and the language | The user's file is the "ISA specification"; the language document is the "TAIDL language specification"; bare "spec" names neither. | Decision 32. |
| Repo, distribution, and module names | `act-oracle`, `taidl-to`, `taidl_to`, `generate_oracle`, `act-backend`, `act_backend`, and `act-ecosystem` keep their spelling; prose says "test oracle generator" and "compiler backend generator". | Renaming them breaks every pin without adding meaning (Q7.11). |
| Generated directory | `targets/<Name>/` becomes `accelerators/<name>/` in both generators; the rename flips only on evidence that external scripts read `targets/`. | "target" is banned for an accelerator (Q7.11). |
| Reference operator for `broadcast` | `broadcast` keeps its name and takes `broadcast_in_dim`. | XLA `Broadcast` adds leading dimensions only. |
| Attribute values and names | An addressing attribute may be negative; implicit bounds apply to the evaluated address, never to attribute values; no tool infers meaning from an attribute name. | Decision 34. |
| Public API names | The v3 surface declares `acc.buffer(...)`, `acc.state(name, value)`, `@acc.instruction`, and `acc.encode(...)`; the v2 method renames of Q7.6 and Q7.10 (`add_buffer`, `add_state`, `computational_attrs`, `addressing_attrs`) apply to the converter's mapping and to keyword names where the surface repeats them. | Q7.6; Q7.10; Q11.2. |
| Names that stay | `update`, `Region`, `State`, and `hbm`; the `@a.`, `@c.`, and `@s.` prefixes belong to the removed string form. | Decisions 19 to 21; Q7.10; Q11.1. |
| Removed names | Each removed or renamed public method leaves a stub that only raises, naming its replacement and the clause id; `add_semantics` names the converter; no working alias. | Q7.12; Q3.12; Q11.4. |

## Errors and coverage

| Topic | Ruling | Reason |
| --- | --- | --- |
| Reject and fault | "reject" covers generation and kernel compile time; "fault" covers run time; the sentence names the stage. | Q7.8. |
| Unsupported input | A generator rejects an unsupported input with a coded error and never assumes it away. | Decision 27. |
| Coverage statuses | A generator's `docs/coverage.md` uses `supported`, `partial`, `excluded`, and `rejected`. | Q7.8; Q11.3; S1.2. |
| Per-generator choices | No case is left to each generator: a case is a fault, a rejection, or a defined behavior. | Decision 16. |

## Overlap, layers, and generator metadata

| Topic | Ruling | Reason |
| --- | --- | --- |
| Overlap | TAIDL assumes no hazard model; each same-buffer input-output pair has one relation, tied, apart, exact-or-apart (the default), or may overlap, declared by a marker or the default. | Decision 31; Q9.1; Q9.2. |
| Markers | A marker is shorthand for its constraints; a violated marker is a fault like any violated constraint. | Decision 30; Q9.3. |
| Layers | Widths, signedness, and encodings are optional layers over 64-bit integer attributes; a derived attribute never appears in the binary. | Decisions 35 and 38. |
| Generator metadata | Generator metadata is read by its own generator and ignored by all others; it never changes the meaning any other generator gives. | Decision 39. |

## Authoring surface and checks

| Topic | Ruling | Reason |
| --- | --- | --- |
| Surface | An ISA specification is a Python file whose instructions are decorated functions, run once by the tracer; the tracer never reads source, and a symbolic value raises on `bool()` instead of taking a silent meaning. | Decision 41; Q11.1. |
| Buffer keywords | `acc.buffer(name, access_dims, base_element_dims, element_type)` and `ops.constant(value, element_type, shape)`: the keywords are the identifiers of the terms access dimensions, base element shape, and element type; `access`, `unit`, and `dtype` are not keywords. | Q11.2; meeting/resolutions.md. |
| Spelling status | The spellings `any_of` and `Comp[u7]` of the Q11.2 sketch are provisional. | Q11.2 confidence. |
| Check | `check` runs once before any generator, returns the checked model, and is the only way to build one; a generator entry given an accelerator calls it. | Decision 42; Q11.3. |
| Reject and exclude | A fault the architect can correct is a rejection; a valid instruction a generator cannot model is an exclusion with a printed, stored reason; an empty usable set is a rejection; the oracle excludes nothing. | Decision 40; Q11.3. |
| Converter | The converter is a script on the taidl 2.x line, run once per ISA specification and checked by a model equality test; v3 has no string form. | Q11.4. |
