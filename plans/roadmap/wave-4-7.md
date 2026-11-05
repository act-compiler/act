# Waves 4 to 7

Wave 4 ends at M3, wave 5 at M4, wave 6 at M5; wave 7 follows M5.

## Wave 4: generators read the front end's checks

42 items, weight 105.

| ID | Item | Depends on | Size | Note |
| -- | ---- | ---------- | ---- | ---- |
| B8 | Name and test each IR-to-IR rewrite rule | B6 | S | -- |
| B11 | Copy of a shared tied input as a step of candidate building | B10, B30 | M | -- |
| B12 | Emit plain code without layout twins; rustfmt at generation | B6 | S | -- |
| B14 | Checks as one pass producing that model | B13, B21 | M | -- |
| B15 | Keep `docs/coverage.md` in sync by a test; document S1.3 cases | B14, L7 | S | -- |
| B17 | Operations beyond the 18, `concatenate` of any arity | B31, B6, B7 | M | -- |
| B19 | Several output regions | B11, B2 | L | -- |
| B20 | `slice` stride, `constant` expressions, float literal range | B6 | S | -- |
| B32 | Guards on bare-variable IR-to-IR rules; phases only if node-limit stops remain | B29, B8 | M | -- |
| B33 | Target constants out of IR-to-IR rewrite 0 | B6 | S | -- |
| B34 | Search budget measurement and candidate deduplication | B10, B29 | S | -- |
| B35 | Overlapping IR-to-ISA rules: declaration-index tie-break, duplicate warning | B29, B6 | S | -- |
| B36 | Drop a candidate that reads unwritten on-chip storage | B10 | S | -- |
| O1 | Runtime code out of `API_FILE` into `runtime/`; split what remains of `utils.py` | O20 | M | -- |
| O2 | One instruction record from the visitor, consumed by checks and emission | O1, O5 | M | -- |
| O3 | Emitter calls in generic form instead of string templates; one word table | O2 | M | absorbs O4 (Q8.6) |
| O7 | Delete the oracle's static checks; run the front end's; copy its shape module | O6, U4 | M | Q8.4 step c |
| O13 | Differential reference: NumPy over the instruction record, module-text goldens | O2 | M | -- |
| O15 | Apply the naming table: identifiers, comments, docs | L25, L27, O1 | M | -- |
| O24 | Call-time errors: checked 64-bit values, constraint overflow, initial state | L5, L6, O20 | S | -- |
| O26 | Unwritten on-chip reads raise at the call; zero fill goes | L5, O20 | M | -- |
| O32 | Width layer: the oracle checks attribute types as implicit range constraints | O10, O24 | S | -- |
| O33 | Encoding layer, derived attributes, generator metadata: ignored; neutrality test; binary input later | O32 | S | -- |
| P3 | Named typed rewrite rules with priorities | B6, B8 | M | -- |
| P4 | Phased saturation schedule | B6, P3 | M | -- |
| P5 | Skipped-pattern and stop-reason reports | P3 | S | -- |
| P10 | Reusable check extractors | B13, P3 | M | -- |
| P11 | Alignment as a buffer property; no new state kind in v3 | B16, C11, P8 | S | -- |
| P12 | Named and oracle-tested expansion rules | P3, P7 | S | -- |
| D3 | Recode and reword the front-end messages | D1, D2, L26 | M | -- |
| T6 | Front-end verdict runner in taidl | D3, T5 | M | -- |
| T7 | Seed the corpus from existing front-end tests (S2 to S7) | T1, T5, T6 | L | -- |
| T9 | Property-based tests for shapes, element types, and names | T6, T8 | M | -- |
| T26 | Norm catalogue and coverage report | D14, T5, T6 | M | -- |
| T29 | Program runner over an oracle interface | D8, T5, T6 | M | -- |
| T30 | Overlap matrix, rejection, and cross-check cases | D8, T29, T5, T6 | M | -- |
| T32 | Width layer cases (range faults, `u64` rejection) | D16, L37, T29, T5, T6 | M | -- |
| D4 | Move run-time-independent checks into the front end | B24, D14, D3, L10, L14, L16, L3, L5, U4 | L | -- |
| D5 | Delete duplicated checks in generators | B14, D4, O6, U10 | M | absorbs B23 (Q8.4 step c) |
| D6 | Unsupported-feature registries | B15, D1, L7, O14, O17 | M | -- |
| D10 | Registry coverage test and wording lint | D1, T5, T6 | M | -- |
| D13 | Machine-readable output and code explanations | D1, D7, D8 | S | -- |

## Wave 5: v3 released on released pins

23 items, weight 66.
The encoding unit (L38, L39, B44, B45, T33, T34) lands whole before U8, and never ships in part (Q10.4).
Release order: U8 releases taidl 3.0; taidl-to 3.0 and act-backend 2.0 pin it and release; A23 releases act 3.0 on those pins.

| ID | Item | Depends on | Size | Note |
| -- | ---- | ---------- | ---- | ---- |
| T12 | Expectation runner: supported, or unsupported with the documented code | D6, T11, T6 | M | -- |
| T13 | Differential tests oracle vs backend on the supported intersection | B29, T12, T29, T3, T8 | L | -- |
| T4 | Delete tests that pin generated text or internal names | T11, T13, T14, T20 | S | -- |
| T18 | Final deletion pass of PR-defense tests | T13, T14, T20, T4 | M | -- |
| T23 | Check the shape module copied into each generated oracle | T12, U4 | S | -- |
| T24 | Nightly grammar fuzzer | T6, T9 | M | -- |
| T25 | Mutation pilot over front-end and oracle fault checks | T12, T6 | S | -- |
| T27 | CI tiers, budgets, and deterministic mode | B29, T12 | M | -- |
| T31 | Backend overlap rows: allocation per relation, copy for a tied shared input | B29, T13, T30 | M | -- |
| T35 | Generator metadata cases (neutrality, namespaces) | L40, T12, T5, T6, U2 | M | -- |
| T36 | Failure policy cases (decision 40) | D16, T12, T13, T27, T29 | M | -- |
| U1 | Close the two emitter divergences | L6, O3 | S | -- |
| U11 | Adding a generator: template generator and expectations | L40, L41, T11, T12, U10, U2, U3 | M | -- |
| D11 | Generate S9 and the unsupported tables | B15, D3, D6, L7, L8, O14 | S | -- |
| L38 | Encoding layer: one encoding block per instruction | L37 | M | encoding unit: lands whole before U8 (Q10.4) |
| L39 | Derived-attribute rule and the encode-decode round trip | L38 | M | encoding unit: lands whole before U8 (Q10.4) |
| B44 | Derived attributes solved in closed form in the allocator | B30, B38 | M | encoding unit: lands whole before U8 (Q10.4) |
| B45 | Assembler and disassembler generated from the encoding layer, landing last as one unit | B44 | L | encoding unit: lands whole before U8 (Q10.4) |
| T33 | Encoding layer cases and assembler round trip | L38, L39, T29, T32, T6 | L | encoding unit: lands whole before U8 (Q10.4) |
| T34 | Derived attribute cases (closed-form links, rejected links) | T33 | M | encoding unit: lands whole before U8 (Q10.4) |
| U8 | Coordinated v3 release train | U10, U2, U3, U4, U5, U6, U7 | M | taidl 3.0 released first (owner 33) |
| T16 | CI wiring: corpus travels with the pinned taidl release | T12, T5 | S | pins the released taidl 3.0 |
| A23 | Release ACT 3.0 | A15, A16, A17, A20, A26 | M | act 3.0 released last, on released pins (owner 33) |

## Wave 6: the tutorial on the release

7 items, weight 16.

| ID | Item | Depends on | Size | Note |
| -- | ---- | ---------- | ---- | ---- |
| A24 | Write one tutorial on the v3 release | A10, A11, A23, A25 | L | written on the released v3 |
| A3 | Remove news, roadmap, and release-history lines from README | A23 | S | -- |
| A12 | Glossary and ASCII pass on the new tutorial | A24 | M | -- |
| A13 | Give every exercise of the new tutorial a runnable solution | A24 | S | -- |
| A14 | Align paper titles and setup text in the new tutorial | A24 | S | -- |
| A21 | Run the new tutorial's solutions in CI | A24 | M | -- |
| A22 | Decide the tutorial lint scope | A24 | S | -- |

## Wave 7: follow-ons after M5

10 items, weight 30.
Each enters with its reason: other-core.md marks it later and no scheduled item depends on it (Q8.1).

| ID | Item | Depends on | Size | Note |
| -- | ---- | ---------- | ---- | ---- |
| U9 | One copy of docker and CI scripts | -- | S | follow-on after M5 (Q8.7) |
| L9 | Worked examples per section | L3 | M | later |
| L18 | Write masks on output regions: an idiom in L24 | L24, L3 | M | later |
| L19 | Strided regions, first in line after v3 | L5 | M | later |
| L20 | `convolution` with window attributes: an idiom in L24 | L3 | M | later |
| L21 | Rounding and saturation attributes on `convert`: an idiom in L24 | L3 | S | later |
| L22 | Sub-computations for `reduce`, `scatter`, `sort`, `map` | L3 | L | later |
| L23 | Indirect regions: starts read from a buffer, joined with the data-dependent addressing study | L5 | M | later |
| L28 | Banned-synonym lint over the D10 word list | D10, L26 | S | later |
| P6 | Cost-aware extraction | B10, P4 | L | later; held until R20 is ruled (Q8.6) |
