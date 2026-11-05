# Decisions

One entry per question the spec track debated.
Positions name the critic role that argued them: hawk (ambiguity), implementer (new generator), architect (accelerator coverage), minimalist, and other tracks (B, O, terminology).
An entry marked "pending panel" waits on the tech-lead panel for the ruling it names.

| id | question | positions | verdict | rationale |
| --- | --- | --- | --- | --- |
| D1 | How does the spec give each operation's meaning? | (a) restate all 78 operations; (b) cite StableHLO and XLA-HLO, restate only TAIDL's changes | (b), with a pinned version (L2) and a per-operation shape table (L3) | Restating invites divergence from the reference the generators already lower to. |
| D2 | Who checks shapes and element types? | (a) each generator; (b) the front end | (b) through L12; until then a generator rejects, and a backtick-dependent mismatch faults (S1.3.8) | Owner decision 16 leaves no undefined case; two generators that check differently accept different descriptions. |
| D3 | Canonical term for a declared storage unit | "data model" vs "buffer" | "buffer"; "data model" names all buffers and state together | Owner decision 19. |
| D4 | Word for raising an error on an ISA description | "refuse" vs "reject" | "reject" | Standard in language specs; "refuse" stays only in code until L26. |
| D5 | `@a.` in a size, backtick expression, or update | (a) allowed; (b) undefined; (c) rejected | (c), checked by L10 | Sizes and shapes must be address-free for B's allocator. |
| D6 | Integer `/` and `%` | floor (Python) vs truncation (C, Rust) | floor, `%` with the divisor's sign | Stated in S6.2; B maps explicitly (L6). |
| D7 | Integer overflow in an expression | per generator vs one rule | exact arithmetic; a final value outside signed 64 bits faults (S1.3.1) | Owner decision 16; O already computes exactly (`utils.py:268-271`), and the old two rules in S6.2 contradicted each other. |
| D8 | Docs layout | one `spec.md` vs per-topic pages with section ids | per-topic pages plus `overview.md` index | Least churn; ids give stable citations. |
| D9 | Grammar in the spec | copied `.g4` rules vs naming the `.g4` files | name the files; keep token tables and binding notes | The copy had already drifted: S3.2 omitted rule `compare` (`IDLV2Expr.g4:25`). |
| D10 | Placement marks (`set_apart`, `set_no_partial_overlap`) | minimalist: remove, write constraints; keep as defined sugar | keep, defined only by the constraints they append; removal pending panel (R58) | No owner decision removes them. |
| D11 | `constant` with a non-scalar shape | fill vs reject | fill every element | Single-literal operand cannot mean anything else. |
| D12 | `reduce_<reducer>` initial value | fixed vs per generator | the reducer's identity (S8.4) | Owner decision 16; matches O's `reduce_init` (`utils.py:439-452`). |
| D13 | Grammar rule name `instruction` for a semantics line | rename now vs later | later, with the renames pending panel (R2) | Renaming needs regenerating parsers and B and O visitor updates. |
| D14 | New element types and masks | add now vs plan | plan: L17 in Q1, L18 after v3 | Masks have an idiom (D28); MX types have none. |
| D15 | Multi-core, async DMA syntax | new constructs vs idioms | idioms first (L24) | Describable today; avoid speculative syntax. |
| D16 | Timing and concurrency | in scope vs out | out of scope (S1) | TAIDL gives functional meaning; timing models are separate tools. |
| D17 | Names reserved by TAIDL-TO in the spec | minimalist: remove; implementer: users must see them | keep in S4, labeled as TAIDL-TO's; removal pending panel (R61) | A user meets the error before reading O's docs. |
| D18 | Order of reads and writes in one instruction | hawk: must be fixed; architect: hardware may stream | all inputs are read before any output is written (S5.2) | TAIDL gives functional meaning (D16); streaming hazards are timing. |
| D19 | Region value line with another element type than its buffer | hawk: implicit bitcast or error; minimalist: no implicit conversion | error: rejected when literal, fault when backtick-dependent | `bitcast_convert` already says it explicitly; owner decision 16. |
| D20 | Name of the user's TAIDL file | glossary: "spec"; terminology: "ISA description" | "ISA description" | "spec" also means this language document. |
| D21 | Parameter `computation_attr` vs term "computational attribute" | terminology and minimalist: keep the API name; hawk: one spelling | keep the API name; rename pending panel (R2) | Owner decision 19 renames only the buffer and state calls. |
| D22 | Scope of banned synonyms (B objection) | B: bans clash with generator internals | bans apply only when naming TAIDL concepts in docs and messages | Generators have their own concepts, such as rewrite conditions. |
| D23 | Status words of a coverage table (B objection) | B: add supported, rejected, partial | `partial` and `rejected` rows; supported features need no row | One row per feature not fully supported keeps tables short. |
| D24 | Replace any component from scratch | none argued for it | no; every piece is kept or reshaped | Grammar and front end are sound; gaps are unstated meaning. |
| D25 | Rejection list in both S7 and S9 | minimalist: one list | S9 only | Two lists drift. |
| D26 | Where shape and type rules live (O argument) | O: move `TYPES` and `shapes.py` out of the oracle; minimalist: leave them | into the front end (L3, L12); the merged item is pending panel (R62) | Every generator must reject the same descriptions. |
| D27 | First extension | architect: convolution; minimalist: types | low-precision and block-scaled types (L17) | Broadest coverage per cost; convolution is describable as `dot` today. |
| D28 | Write masks as syntax (L18) | architect: needed for SME and NKI; minimalist: expressible today | describable today: pass the output region as an input, `select` by a `pred` region, mark in place | Idiom works with no grammar change; SME is a CPU extension outside owner decision 15. |
| D29 | What an in-place mark means | storage reuse vs a hint | output region 0 reuses input region k's storage; buffer and sizes checked, starts fault if unequal | Owner decision 17. |
| D30 | Which state constraints read | before updates vs after | before, except placement-mark constraints, which read the state their starts read | Matches O: constraints at the pre-state, starts after updates; the all-after alternative is pending panel (R7). |
| D31 | Storage before the first issue | zero vs unspecified | every buffer but `d0` is 0; `d0` holds the `hbm` contents | Owner decision 16 leaves no undefined read. |
| D32 | Float accumulation and transcendental accuracy | fixed order vs tolerance | open (S1.3.15), pending panel (R8) | A fixed order binds XLA-backed oracles to one schedule; a tolerance needs agreed values. |
| D33 | `set_inplace`, `set_apart`, `set_no_partial_overlap`, and three rare words removal | minimalist: cut; keep | keep in this draft; pending panel (R5, R58, R60) | Owner decision 17 names the in-place mark; the others wait on the panel. |
| D34 | `add_constraints` beside `constraints=` | minimalist: one spelling; keep both | `constraints=` only, removed in v3 (L30) | Two spellings of one list (`accelerator.py:66`, `instruction.py:321-323`); owner decision 10 batches the break. |
