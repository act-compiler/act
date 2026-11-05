# Quarter 1, supporting tracks

Plan items scheduled for quarter 1 in the supporting tracks, by track.

## casestudies

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| C1 | Typed-view idiom and byte order | C10, F6 | S | -- |
| C2 | Strided regions: evidence that admits L19 | C11 | S | -- |
| C3 | Instruction families and indexed state: idiom | C10 | S | -- |
| C4 | Mode-dependent semantics: idiom | C11 | S | -- |
| C6 | Added element types: evidence for L17 | L17 | S | -- |
| C9 | Float attributes and bit operators: idiom | C10 | S | -- |
| C10 | Modeling cookbook and scope notes | -- | S | -- |
| C11 | Case studies as an executable corpus | -- | M | R16 |
| C13 | Defined-behavior audit of the case studies | C11 | S | -- |

## priorart

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| P1 | Numbered per-operation constraints in S8 | -- | M | -- |
| P2 | Three-word taxonomy (rejected, fault, defined value) for every S1.3 item | P1 | S | -- |
| P3 | Named typed rewrite rules with priorities | B6, B8 | M | -- |
| P4 | Phased saturation schedule | B6, P3 | M | -- |
| P5 | Skipped-pattern and stop-reason reports | P3 | S | -- |
| P7 | Executable examples per spec row | P1, P2 | M | -- |
| P12 | Named and oracle-tested expansion rules | P3, P7 | S | -- |

## shared

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| U2 | Typed front-end contract with a version | L15 | M | -- |
| U3 | One final validation that publishes derived facts | L5, U2 | M | -- |
| U5 | Element-type table and literal check | L16, O8, U2 | M | -- |
| U10 | Generators call `validate()` first | U3 | S | -- |

## diagnostics

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| D1 | Registry, classes and message format | L1, L25 | M | -- |
| D2 | Call sites and exception classes in the front end | D1, U3 | M | -- |
| D3 | Recode and reword the front-end messages | D1, D2, L26 | M | -- |
| D6 | Unsupported-feature registries | B15, D1, L7, O14, O17 | M | -- |
| D8 | Oracle run-time and call diagnostics | D1, D14, L6, O11, O6 | M | -- |
| D9 | Silent fallbacks become errors | B24, B26, D1, L5 | S | -- |
| D10 | Registry coverage test and wording lint | D1, T5, T6 | M | -- |
| D14 | Fault table and run-time norm | L5, L6 | S | -- |

## semantics

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| F1 | Typing rules that U4 implements in the front end | -- | M | -- |
| F2 | Region aliasing rules and in-place mark meaning | -- | M | -- |
| F3 | Total integer arithmetic: overflow and zero divisors fault | -- | S | -- |
| F6 | Byte order, sub-byte packing, size of d0 | -- | S | -- |
| F8 | Front-end rejections the spec already implies | -- | S | -- |
| F9 | Executable reference interpreter of the exact classes | F1, F3 | M | -- |
