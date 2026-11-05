# Decisions

One entry per question the spec track debated.
Positions name the critic role that argued them: hawk (ambiguity), implementer (new generator), architect (accelerator coverage), minimalist, and other tracks (B, O, terminology).

| id | question | positions | verdict | rationale |
| --- | --- | --- | --- | --- |
| D1 | How does the spec give each operation's meaning? | (a) restate all 70 operations; (b) cite StableHLO and XLA-HLO, restate only TAIDL's changes | (b), with a pinned version (L2) and a per-operation shape table (L3) | Restating invites divergence from the reference the generators already lower to. |
| D2 | Who checks shapes and element types? | (a) each generator; (b) the front end | (b) eventually (L12); until then undefined (S1.3) | Two generators that check differently accept different specs. |
| D3 | Canonical term for a declared memory | "data model" (API name) vs "buffer" | "data model" | Matches `add_data_model`; "buffer" collides with runtime buffers in B and O. |
| D4 | Word for raising an error on an ISA description | "refuse" vs "reject" | "reject" | Standard in language specs; "refuse" stays only in code until L26. |
| D5 | `@a.` in a size, backtick expression, or update | (a) allowed; (b) undefined; (c) rejected | (c) as the rule, undefined until L10 lands | Sizes and shapes must be address-free for B's allocator. |
| D6 | Integer `/` and `%` | floor (Python) vs truncation (C, Rust) | floor, `%` with the divisor's sign | Already stated in S6.2; B maps explicitly (L6). |
| D7 | Integer overflow in an expression | undefined vs implementation-defined | implementation-defined, documented per generator | Generators already differ and document it (S6.2). |
| D8 | Docs layout | one `spec.md` vs per-topic pages with section ids | per-topic pages plus `overview.md` index | Least churn; ids give stable citations. |
| D9 | Grammar in the spec | EBNF rewrite vs verbatim `.g4` rules | verbatim, `.g4` normative | One source; L8 checks the copy. |
| D10 | Placement marks (`set_apart`, `set_no_partial_overlap`) | minimalist: remove, write constraints; keep as defined sugar | keep, defined only by the constraints they append | Generators see only constraints; marks save error-prone text. |
| D11 | `constant` with a non-scalar shape | fill vs reject | fill every element | Single-literal operand cannot mean anything else; O and B confirm (L4). |
| D12 | `reduce_<reducer>` initial value | fix now vs implementation-defined | implementation-defined until L4 | Not verified in B or O code this session. |
| D13 | Grammar rule name `instruction` for a semantics line | rename now vs later | later (L15) | Renaming needs regenerating parsers and B and O visitor updates. |
| D14 | New element types and masks | add now vs plan | plan (L17, L18), no grammar edit this quarter's first week | Owner rule: no grammar changes in this meeting. |
| D15 | Multi-core, async DMA syntax | new constructs vs idioms | idioms first (L24) | Describable today; avoid speculative syntax. |
| D16 | Timing and concurrency | in scope vs out | out of scope (S1) | TAIDL gives functional meaning; timing models are separate tools. |
| D17 | Names reserved by TAIDL-TO in the spec | minimalist: remove, generator-specific; implementer: users must see them | keep in S4, labeled as TAIDL-TO's, until O lists them in its coverage table (L7) | A user meets the error before reading O's docs. |
| D18 | Order of reads and writes in one instruction | hawk: must be fixed; architect: hardware may stream | all inputs are read before any output is written (S5.2) | TAIDL gives functional meaning (D16); streaming hazards are timing. |
| D19 | Region value line with another element type than its data model | hawk: implicit bitcast or error; minimalist: no implicit conversion | error, undefined until L12 checks it | `bitcast_convert` already says it explicitly. |
| D20 | Name of the user's TAIDL file | glossary: "spec"; terminology: "ISA description" | "ISA description", pending owner | "spec" also means this language document. |
| D21 | Parameter `computation_attr` vs term "computational attribute" | terminology and minimalist: keep the API name; hawk: one spelling | rename in v3 (L15); disagrees with the terminology reference | The owner allows one breaking release; consistency is a stated goal. |
| D22 | Scope of banned synonyms (B objection) | B: bans clash with generator internals | bans apply only when naming TAIDL concepts in docs and messages | Generators have their own concepts, such as rewrite conditions. |
| D23 | Status words of a coverage table (B objection) | B: add supported, rejected, partial | `partial` and `rejected` rows; supported features need no row | One row per feature not fully supported keeps tables short. |
| D24 | Replace any component from scratch | none argued for it | no; every piece is kept or reshaped | Grammar and front end are sound; gaps are unstated meaning. |
| D25 | Rejection list in both S7 and S9 | minimalist: one list | S9 only | Two lists drift. |
| D26 | Where shape and type rules live (O argument) | O: move `TYPES` and `shapes.py` out of the oracle; minimalist: leave them | into the front end (L3, L12, with O6, O7) | Every generator must reject the same descriptions. |
| D27 | First extension | architect: convolution; minimalist: types | low-precision and block-scaled types (L17) | Broadest coverage per cost; convolution is describable as `dot` today. |
| D28 | Write masks as syntax (L18) | architect: needed for SME and NKI; minimalist: expressible today | describable today, syntax later: pass the output region as an input, `select` by a `pred` region, mark in place | Idiom works with no grammar change; L18 drops to Q2. |
