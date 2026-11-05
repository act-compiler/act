# Decisions

One entry per question the spec track debated.
Positions name the critic role that argued them: hawk (ambiguity), implementer (new generator), architect (accelerator coverage), minimalist, and other tracks (B, O, terminology).

| id | question | positions | verdict | rationale |
| --- | --- | --- | --- | --- |
| D1 | How does the spec give each operation's meaning? | (a) restate all 78 operations; (b) cite StableHLO and XLA-HLO, restate only TAIDL's changes | (b), with a pinned version (L2) and a per-operation shape table (L3) for the 75 words left after L33 | Restating invites divergence from the reference the generators already lower to. |
| D2 | Who checks shapes and element types? | (a) each generator; (b) the front end | (b) through the shared shape module of U4; literal and equal-folded dims are decided by the front end, others fault at the call (S1.3.8) | Owner decision 16; Q3.1, Q3.6. |
| D3 | Canonical term for a declared storage unit | "data model" vs "buffer" | "buffer"; "data model" names all buffers and state together | Owner decision 19. |
| D4 | Word for raising an error on an ISA specification | "refuse" vs "reject" | "reject" at generation and kernel compile time; "fault" at run time | Q7.8; L34 renames "refus" in messages, docstrings, READMEs, and public identifiers. |
| D5 | `@a.` in a size, backtick expression, or update | (a) allowed; (b) undefined; (c) rejected | (c), checked by L10 | Sizes and shapes must be address-free for B's allocator. |
| D6 | Integer `/` and `%` | floor (Python) vs truncation (C, Rust) | floor, `%` with the divisor's sign | Stated in S6.2; B maps explicitly (L6). |
| D7 | Integer overflow in an expression | per generator vs one rule | every operator yields a signed 64-bit integer, or the issue faults (S1.3.1) | Q1.2; B already emits checked arithmetic per operator. |
| D8 | Docs layout | one `spec.md` vs per-topic pages with section ids | per-topic pages plus `overview.md` index | Least churn; ids give stable citations. |
| D9 | Grammar in the spec | copied `.g4` rules vs naming the `.g4` files | no grammar: S2 and S3 state the Python surface and what tracing rejects | Owner decision 41; the grammars leave with L45. |
| D10 | Placement marks (`set_apart`, `set_no_partial_overlap`) | minimalist: remove, write constraints; keep as defined sugar | keep as shorthand that expands to constraints through the implicit-constraint mechanism: `set_apart` to `apart`, `set_no_partial_overlap` to `same or apart`, new `set_may_overlap` to nothing (S5.5, L35, L36) | Owner decision 30; Q9.1, Q9.2. |
| D11 | `constant` with a non-scalar shape | fill vs reject | fill every element | Single-literal operand cannot mean anything else. |
| D12 | `reduce_<reducer>` initial value | fixed vs per generator | the reducer's identity (S8.4) | Owner decision 16; matches O's `reduce_init` (`utils.py:439-452`). |
| D13 | Grammar rule name `instruction` for a semantics line | rename now vs later | no target: the grammar leaves with L45 | Owner decision 41. |
| D14 | New element types and masks | add now vs plan | plan: L17 in Q1, L18 after v3 | Masks have an idiom (D28); MX types have none. |
| D15 | Multi-core, async DMA syntax | new constructs vs idioms | idioms first (L24) | Describable today; avoid speculative syntax. |
| D16 | Timing and concurrency | in scope vs out | out of v3; program order is the only order, and a constraint used as a wait faults when false | Q2.6. |
| D17 | Names reserved by TAIDL-TO in the spec | minimalist: remove; implementer: users must see them | no table; S4.1 has one pointer sentence to TAIDL-TO's list, and the error names the word, the generator, and the list | Q2.10. |
| D18 | Order of reads and writes in one instruction | hawk: must be fixed; architect: hardware may stream | no read and write order and no hazard-order declaration in v3; a permitted overlap gets the semantics on the pre-issue state; each same-buffer input-output pair is tied, apart, exact-or-apart (default), or may overlap | Owner decision 31; Q9.1, Q9.3. |
| D19 | Region value line with another element type than its buffer | hawk: implicit bitcast or error; minimalist: no implicit conversion | error: rejected when literal, fault when backtick-dependent | `bitcast_convert` already says it explicitly; owner decision 16. |
| D20 | Name of the user's TAIDL file | glossary: "spec"; terminology: "ISA description" | "ISA specification"; the language document is the "TAIDL language specification" | Owner decision 32. |
| D21 | Parameter `computation_attr` vs term "computational attribute" | terminology and minimalist: keep the API name; hawk: one spelling | rename to `computational_attrs` and `addressing_attrs` in v3 (L30) | Q2.12, Q7.10. |
| D22 | Scope of banned synonyms (B objection) | B: bans clash with generator internals | bans apply only when naming TAIDL concepts in docs and messages | Generators have their own concepts, such as rewrite conditions. |
| D23 | Status words of a coverage table (B objection) | B: add supported, rejected, partial | `partial` and `rejected` rows; supported features need no row | One row per feature not fully supported keeps tables short. |
| D24 | Replace any component from scratch | none argued for it; then owner decision 41 | the string surface and its grammars are replaced by the Python surface; every check and table is kept | Owner decision 41; panel 11. |
| D25 | Rejection list in both S7 and S9 | minimalist: one list | S9 only | Two lists drift. |
| D26 | Where shape and type rules live (O argument) | O: move `TYPES` and `shapes.py` out of the oracle; minimalist: leave them | one shared item U4 moves the module and adds the checks; L3 stays the spec text | Q3.1; the generator copies the module into each oracle (owner decision 25). |
| D27 | First extension | architect: convolution; minimalist: types | `f4e2m1fn` and `f8e8m0fnu` only (L17); `scaled_dot` stays an idiom | Q2.4; two public MX sites, and the idiom is exact for power-of-two scales. |
| D28 | Write masks as syntax (L18) | architect: needed for SME and NKI; minimalist: expressible today | describable today: pass the output region as an input, `select` by a `pred` region, mark in place | Idiom works with no grammar change; SME is a CPU extension outside owner decision 15. |
| D29 | What an in-place mark means | storage reuse vs a hint | the output reuses the marked input's storage; a pair in different buffers or with unequal size text is rejected; it expands to `same(In k, Out)` plus a recorded tie | Owner decisions 17, 24, 30; Q9.2. |
| D30 | Which state expressions read | before updates vs after | every expression of an issue reads the pre-state; updates and outputs commit together | Q1.1. |
| D31 | Storage before the first issue | zero vs unspecified | a read of an unwritten on-chip element faults; `d0` holds the caller's `hbm` | Q1.4. |
| D32 | Float accumulation accuracy | fixed order vs tolerance | two Higham bounds plus an overflow clause for `dot`, `reduce_add`, `reduce_multiply` (L31) | Q1.5. |
| D33 | `set_inplace`, `set_apart`, `set_no_partial_overlap`, and three rare words removal | minimalist: cut; keep | marks stay as shorthand with the S5.5 expansions; the three words are removed (L33) | Owner decision 30; Q2.8. |
| D34 | `add_constraints` beside `constraints=` | minimalist: one spelling; keep both | `constraints=` only, removed in v3 (L33) | Two spellings of one list (`accelerator.py:66`, `instruction.py:321-323`); owner decision 10 batches the break. |
| D35 | May an addressing attribute be negative, and may a name carry meaning? | constrain attribute values vs only addresses | an attribute may be negative; implicit constraints apply to the evaluated address; names are plain variable names | Owner decision 34. |
| D36 | Authoring surface | string semantics vs Python by tracing vs Python by source reading vs C++ or MLIR | Python by tracing one call of each decorated function; standard library only (L42) | Owner decision 41; errors land on the author's line; Q11.1. |
| D37 | Region and marker spelling | slices vs `buffer.region(starts, size=sizes)`; markers in the decorator vs on `write` | `buffer.region` with `read` and `write`; markers as `write` keywords naming the read value | A region stores a start and a size; a marker pairs one input with one output by object; Q11.2. |
| D38 | Conditions | bitwise `&` and or-operator vs `any_of` and `all_of` | `require`, `any_of`, `all_of`; both bitwise operators raise | The or-operator binds tighter than a comparison; Q11.2. |
| D39 | What a generator does with an instruction it cannot model | reject the specification vs exclude the instruction | `check` rejects what the architect can correct; a generator excludes with a printed, recorded reason (L43) | Owner decisions 40 and 42; Q11.3. |
| D40 | Migration to the Python surface | converter vs dual acceptance vs by hand | a converter script on the 2.x line with a `Model` equivalence test per file (L44) | Owner decision 10 batches the break; Q11.4. |
| D41 | Keywords of `acc.buffer` and `ops.constant` | Q11.2 sketch: `access`, `unit`, `dtype`; glossary: the identifiers of its terms | `access_dims`, `base_element_dims`, `element_type`; the three sketch spellings are not keywords | Glossary renames; owner decision 19; meeting/resolutions.md. |
| D42 | Spellings `acc.d0`, `ops.and_`, and `(high, low)` bit slices | none argued; the drafts' spellings against panel 11 and the glossary | all three stand; "slice" names only a bit slice of the encoding block, never a region | `d0` is predeclared (owner decision 19); `and`, `or`, `not` are Python keywords; panel 10 orders slices high to low. |
| D43 | Signature of `ops.constant` | draft `dtype=` keyword vs positional sketch of Q11.2 | `ops.constant(value, element_type=s64, shape=[])`; the positional order is value, element type, shape | D41; owner decision 38 sets the `s64` default. |
