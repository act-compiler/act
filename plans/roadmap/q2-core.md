# Quarter 2, core tracks

Plan items scheduled for quarter 2 in the core tracks, by track.

## language

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| L8 | Spec consistency checks in CI | L1 | S | -- |
| L14 | Test the plain-zero divisor rule | U12 | S | -- |
| L38 | Encoding layer: one encoding block per instruction | L37 | M | -- |
| L39 | Derived-attribute rule and the encode-decode round trip | L38 | M | -- |
| L41 | Guide: adding a generator | L40 | S | -- |

## backend

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| B5 | Reader checks HLO annotations against the inferred shapes | B7 | S | -- |
| B10 | One k-best extractor on (class, buffer) | B29, B3, B37, B9 | M | -- |
| B11 | Copy of a shared tied input as a step of candidate building | B10, B30 | M | -- |
| B12 | Emit plain code without layout twins; rustfmt at generation | B6 | S | R19 |
| B16 | Read-only state folded at compile time | B13, B24, L1 | M | -- |
| B17 | Operations beyond the 18, `concatenate` of any arity | B31, B6, B7 | M | -- |
| B18 | Element types `s2`, `s4`, `u2`, `u4`, `f8e4m3fn`, `f8e5m2` | B7, L17 | S | -- |
| B19 | Several output regions | B11, B2 | L | -- |
| B20 | `slice` stride, `constant` expressions, float literal range | B6 | S | -- |
| B32 | Guards on bare-variable IR-to-IR rules; phases only if node-limit stops remain | B29, B8 | M | -- |
| B33 | Target constants out of IR-to-IR rewrite 0 | B6 | S | -- |
| B34 | Search budget measurement and candidate deduplication | B10, B29 | S | -- |
| B36 | Drop a candidate that reads unwritten on-chip storage | B10, L46 | S | -- |
| B42 | Input validation and a documented input specification | B46, B5 | M | -- |
| B43 | Input formats other than XLA HLO text: detection, including portable artifacts; readers in B47 and B53 | B42 | S | -- |
| B44 | Derived attributes solved in closed form in the allocator | B30, B38 | M | -- |
| B45 | Assembler and disassembler generated from the encoding layer, landing last as one unit | B44 | L | -- |

## oracle

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| O3 | Emitter calls in generic form instead of string templates; one word table | O2 | M | R30 |
| O7 | Delete the oracle's static checks; read only the checked `Model`; copy the front end's shape module | O6, U10, U4 | M | R31, R62 |
| O12 | Drop the matmul-precision flag; drop x64 after a feed test | -- | S | R32 |
| O13 | Differential reference: NumPy over the instruction record, module-text goldens | O2 | M | -- |
| O26 | Unwritten on-chip reads raise at the call; zero fill becomes the declared-value fill | L46, L5, O20 | M | -- |
| O28 | Build the C++ debug handler once per install and jax version | -- | S | -- |
| O29 | Study: migration plan of the oracle onto fsim-lib | -- | M | -- |
| O33 | Encoding layer, derived attributes, generator metadata: ignored; neutrality test; binary input later | O32 | S | -- |

## tests

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| T4 | Delete tests that pin generated text or internal names | T11, T13, T14, T20 | S | -- |
| T9 | Property-based tests for shapes, element types, and names | T6, T8 | M | -- |
| T13 | Differential tests oracle vs backend on the supported intersection | B29, T12, T29, T3, T8 | L | -- |
| T15 | Golden policy: one regenerate, one record, one tolerance | B29, T14 | S | -- |
| T18 | Final deletion pass of PR-defense tests | T13, T14, T20, T4 | M | -- |
| T23 | Check the shape module copied into each generated oracle | T12, U4 | S | -- |
| T24 | Nightly surface fuzzer | T6, T9 | M | -- |
| T25 | Mutation pilot over front-end and oracle fault checks | T12, T6 | S | -- |
| T31 | Backend overlap rows: allocation per relation, copy for a tied shared input | B29, T13, T30 | M | -- |
| T32 | Width layer cases (range faults, `u64` rejection) | D16, L37, T29, T5, T6 | M | -- |
| T33 | Encoding layer cases and assembler round trip | L38, L39, T29, T32, T6 | L | -- |
| T34 | Derived attribute cases (closed-form links, rejected links) | T33 | M | -- |
| T36 | Failure policy cases (decision 40) | D16, T12, T13, T27, T29 | M | -- |

## act

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| A3 | Remove news, roadmap, and release-history lines from README | A23 | S | -- |
| A12 | Glossary and ASCII pass on the new tutorial | A24 | M | -- |
| A17 | Check sibling pins and dependency floors against act | A15 | M | -- |
| A19 | Cover state variables and constraints in the end-to-end test | -- | M | -- |
| A21 | Run the new tutorial's solutions in CI | A24 | M | -- |
| A23 | Release ACT 3.0 | A15, A16, A17, A20, A26, A28, A31, U8 | M | -- |
| A24 | Write one tutorial on the v3 release | A10, A11, A23, A25, A28 | L | -- |
| A26 | Gate the tag with a fresh-venv install | A25 | S | -- |
| A27 | Document how to add a new generator | A25, A8, U10, U2, U3 | M | -- |
| A28 | Converter in the last 2.x release and act's own ISA specifications | A16, U13 | M | -- |
| A31 | Release gate: the M4 evidence passes on the private heads | B44, B45, L38, L39, T12, T13, T33, T34, T36, T37, T38, U14 | S | -- |
