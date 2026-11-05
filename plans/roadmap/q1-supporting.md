# Quarter 1, supporting tracks

Plan items scheduled for quarter 1 in the supporting tracks, by track.

## casestudies

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| C1 | Typed views of storage | -- | M | -- |
| C2 | Strided and multi-dimensional regions | C1 | L | -- |
| C3 | Instruction families and indexed state | -- | M | -- |
| C4 | Mode-dependent semantics | -- | M | -- |
| C9 | Float attributes and bit operators | -- | S | -- |
| C10 | Modeling cookbook and scope notes | -- | S | -- |
| C11 | Case studies as an executable corpus | -- | M | R16 |
| C13 | Defined-behavior audit of the case studies | C11 | S | -- |

## priorart

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| P1 | Numbered per-operation constraints in S8 | -- | M | -- |
| P2 | Three-word taxonomy for every S1.3 item | P1 | S | -- |
| P3 | Named typed rewrite rules with priorities | B6, B8 | M | -- |
| P4 | Phased saturation schedule | B6, P3 | M | -- |
| P5 | Skipped-pattern and stop-reason reports | P3 | S | -- |
| P7 | Executable examples per spec row | P1, P2 | M | -- |
| P12 | Named and oracle-tested expansion rules | P3, P7 | S | -- |

## diagnostics

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| D1 | Registry, classes and message format | L1, L25 | M | -- |
| D2 | Call sites and exception classes in the front end | D1 | M | -- |
| D3 | Recode and reword the front-end messages | D1, D2, L26 | M | -- |
| D14 | Run-time violation table and norm | L5, L6 | S | -- |

## semantics

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| F1 | Static shape and element type checker in the front end | -- | L | -- |
| F2 | Region aliasing rules and in-place mark meaning | -- | M | -- |
| F3 | Total integer arithmetic: overflow and zero divisors reject | -- | S | -- |
| F6 | Byte order, sub-byte packing, size of d0 | -- | S | -- |
| F8 | Front-end rejections the spec already implies | -- | S | -- |
| F9 | Executable reference interpreter of the exact classes | F1, F3 | M | -- |
