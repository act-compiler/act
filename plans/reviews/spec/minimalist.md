# Minimalist review of the TAIDL spec and plan

The language is small and sound; the plan around it is not.
The grammar has 78 operation words (`IDLV2.g4:45-74`), two marks nobody uses, and a plan of 28 items with three L sizes for a track that shares 3-4 engineers with B, O, A, and T.
Every count below comes from the 28 files with `add_instruction` across the four repos, examples and tests included.
Owner rulings 16 (no undefined behavior) and 17 (in-place is true aliasing) are applied where marked.

## Verdicts

| Item or construct | Verdict (keep / merge / cut / defer) | Reason |
| --- | --- | --- |
| Data models, regions `[data model, starts, sizes]` | keep | Every instruction in every example uses them; `check_regions` is 15 lines (`instruction.py:115-129`). |
| `d0` as a special case | merge | `d0` is declared with no dimensions (`accelerator.py:31`) and then given one by `count = 1 if buffer == 'd0'` (`instruction.py:123`); declare it as an ordinary data model with one addressed dimension and the rule goes. |
| Two attribute lists, `@c.` and `@a.` | keep | B's allocator needs address-free sizes (D5); two generators read `comp_attr` and `parameters`. |
| State variables and updates | keep, do not extend | B refuses every read or update (`act-backend/act_backend/spec_checks.py:372-384`); no example or tutorial uses them; they stay as the only way to describe configuration registers. |
| Constraints | keep | O checks them at every call (`act-oracle/docs/coverage.md`); every rejected run-time case under ruling 16 can be stated as one. |
| `add_constraints` beside `constraints=` | merge | Two spellings of one list (`accelerator.py:66`, `instruction.py:321-323`); keep the keyword. |
| `set_inplace` | merge, derive it | Under ruling 17 an in-place input and the output share storage, so they are the same region; an output region spelled as an input region is in place, and the call says nothing more (softmax, `taidl/examples/mini_trn.py:92-93`). |
| `set_apart`, `set_no_partial_overlap` | cut | No example or tutorial calls them; the only callers are B and O tests (`act-backend/tests/test_generation.py:220-221`, `act-oracle/tests/test_constraints.py:41`); each writes one constraint text the user can write (`instruction.py:216-218`); 70 lines of front end (`instruction.py:174-244`), half of S5.5, and five S9 rows go. |
| Semantics block, `%In<k>`, `%Out<k>` | keep | Checked both ways (`instruction.py:142-161`, `:290-307`); the strongest part of the design. |
| `ROOT` | defer | It carries no information, since S7.1 fixes which line has it, but it keeps a semantics pasteable as XLA-HLO text; revisit only with another grammar change. |
| Presumed reshape (S7.2) | cut | An implicit rule with `partial` support in O; one explicit `reshape` line replaces it, and 196 `reshape` lines show users write that line anyway. |
| 75 operation words with no or list attributes | keep | One row each in `OPERATIONS` (`operations.py:41-165`); `copy` alone has 50 uses. |
| `cholesky`, `triangular_solve`, `batch_norm_inference` | cut | Each appears once, in a B or O test; no tensor accelerator has such an instruction; they alone need the `BOOL`, `TRANSPOSES`, and `NUMBER` forms (`operations.py:18-23`, `:145-156`) and O's `f16` partial row. |
| `reduce_<reducer>` family | keep | TAIDL's own; it is what keeps bodies out of the grammar. |
| 19 element types | keep | O supports all; the lexer rule is six lines (`IDLV2.g4:37-42`). |
| S1.3 table | keep, shrink | Under ruling 16 the `kind` column has two values, `error at the call` or a defined result; every `undefined` row becomes one of them. |
| S2 and S3 copied `.g4` rules | merge | A copy of a normative file kept equal by a CI test (L8) is a second source; keep the token tables and precedence notes, link the two `.g4` files for the rules. |
| S4 "Names reserved by generators" | cut | TAIDL-TO's list; O's `coverage.md` already points at O's README for it. |
| S6 "Parsed trees" | merge | Front-end API (`constraint_asts`, `instr_expression_asts`), not language; move to the `taidl` README. |
| S8.2, S8.3, S8.5, S9 | keep | One table each, mirrored by code or by a single list. |
| S10 banned synonyms column | simplify | Keep the terms; keep bans only for the four words that collide (`spec`, `buffer`, `parameter`, `op`). |
| L1, L2 | keep | S each; ids and a pinned reference are what every other item cites. |
| L3 and L12 with O6, O7 | merge | Two L items that restate what `act-oracle/taidl_to/generic/shapes.py` (543 lines) already computes; move that module into `taidl` and write the table from it; one M. |
| L4 | simplify | O already starts each reducer at its identity (`coverage.md`); write that sentence; S. |
| L5 | simplify | O already rejects an out-of-bounds region at the call (`coverage.md`); under ruling 16 every region row is `error at the call`; S. |
| L6 | keep | S; already stated in S6.2. |
| L7 | cut from this track | O's `coverage.md` exists; B owns its own. |
| L8 | simplify | Keep the `TYPES` and `OPERATIONS` list check; drop the grammar-copy diff once S3 links the `.g4`. |
| L9 | defer | `tests/test_examples.py` already runs the examples; per-section examples are polish. |
| L10, L11, L13 | keep | S each; they turn a stated rule into a check. |
| L14 | cut the code | Narrow S9 to "the literal `0` directly", as its own caveat offers; `instruction.py:414` stays. |
| L15 | shrink | Rename `parameters` and delete `num_dim_str`, which no generator reads; the rest is churn in three repos. |
| L16 | defer | O checks the literal at generation; no generator disagrees today. |
| L17 | keep, shrink | `f4e2m1fn`, `f8e8m0fnu`, and `scaled_dot` cover Blackwell MX and Trainium2; defer `f6*`, `s1`, `u1`, each with at most one named user. |
| L18 | cut | D28 already gives the idiom; SME and SVE are CPU vector extensions, outside the owner's scope. |
| L19 | defer | Idiom: cover the span as the region, then `slice` with a stride in the semantics. |
| L20 | defer | D27 already says `dot` plus `pad` and `slice` describes it; add it at the second real ISA description that asks. |
| L21 | cut | Saturation is `clamp` then `convert`; toward-zero is `convert`'s float-to-integer rule; stochastic rounding needs random state, which S8.5 excludes. |
| L22 | cut | It names no accelerator ("any design with custom reductions"); it is the largest grammar change and the one that lets bodies in. |
| L23 | defer to late Q2 | The one extension with real need and no idiom (Trainium indirect DMA, SparseCore), but it costs B its static addresses; second extension, after L17. |
| L24 | keep | S; add the strided and saturation idioms to it. |
| L25, L26 | keep | S and M; error texts change once, in v3. |
| L27, L28 | cut | L27 is other tracks' work; L28 admits an allow list that grows. |

## What the language does well and must not lose

- One line per value in XLA-HLO spelling, so a semantics is readable by anyone who reads HLO (`IDLV2.g4:8`).
- A separate entry rule per place in the expression grammar, so a constraint cannot be passed as a start (`IDLV2Expr.g4:5-7`, `instruction.py:387-393`).
- One data table for every operation word's operands and attribute forms (`operations.py:41-165`), checked at the call that takes the text.
- Errors that name the instruction, the place, the line, and the column (`instruction.py:366-383`).
- `%In<k>` and `%Out<k>` tied to regions by position and checked in either order (`instruction.py:99-100`, `:254-257`).
- The reducer fixed in the word, which keeps every body out of the grammar.
- S8.5, which explains every absence in one list; it is the page a new generator author reads first.

## Speculative generality in the extensions

- Fewer than two real accelerators: L22 names none, L18 names two CPU extensions outside scope plus two masks the D28 idiom covers, and L21's stochastic case has one user and no reference operation.
- Describable today: L18 (D28), L19 (span plus strided `slice`), L20 (D27), L21 (`clamp` plus `convert`).
- Of eight proposals, only L17 and L23 add something no idiom gives; L24 is a doc.
- The plan's own rule, "none changes the grammar this quarter without its own review" (`extensions.md:4`), is the right one; apply it to the whole of Q2 as well.

## Order and sizes for 3-4 engineers

- Each plan file says "sizes assume two engineers"; four tracks cannot each have two of 3-4.
- 28 items with three L sizes is two quarters of one track alone; the cuts above leave 15 items and no L.
- The six A tutorials still write `ENTRY load_rm {` and `%In1 = ... parameter(0)` (`act/tutorials/pldi26/exercise3/solution/QKV.py:25-26`), which `IDLV2.g4:6` and `instruction.py:294-302` reject; every "notice by A's tutorial run" fail point is void until A moves them to v2.15, so that comes first.
- One breaking batch, early in Q1: L10, L11, L13, the `d0` merge, the mark and word cuts, the presumed-reshape cut, and the two renames; B and O migrate once.
- Ruling 16 makes L5 and the S1.3 rewrite one S item: every former `undefined` row reads `error at the call`, and O already behaves so.
- Ruling 17 fixes the in-place meaning in one sentence and removes `set_inplace` from the API; B reads the derived flags where it read the given ones (`act-backend/act_backend/instruction_info.py:112`).
- L3's "70 words" is wrong; the grammar has 78, as O's `coverage.md` says.
- Order after the batch: L2, then the merged shape item, then L17, then L23 if Q2 has room; docs items L9, L16 last.

## Cuts, in one list

- `set_apart` and `set_no_partial_overlap`, with their S5.5 and S9 text.
- `set_inplace` as a call; the flag is derived from equal region texts.
- `add_constraints`; `constraints=` stays.
- Presumed reshape.
- `cholesky`, `triangular_solve`, `batch_norm_inference`, and the `BOOL`, `TRANSPOSES`, and `NUMBER` forms they alone need.
- The S4 reserved-names table and the copied grammar rules of S3.
- L7, L14's code change, L18, L21, L22, L27, L28.
