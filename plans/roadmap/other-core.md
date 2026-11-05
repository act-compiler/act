# Later, core tracks

Plan items scheduled for later in the core tracks, by track.

## language

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| L9 | Worked examples per section | L3 | M | -- |
| L12 | Shape and element type checker: merged into U4 | U4 | - | R62 |
| L13 | Integer literal lexing: no target after L45 | L45 | - | -- |
| L18 | Write masks on output regions: an idiom in L24 | L24, L3 | M | -- |
| L19 | Strided regions, first in line after v3 | L5 | M | -- |
| L20 | `convolution` with window attributes: an idiom in L24 | L3 | M | -- |
| L21 | Rounding and saturation attributes on `convert`: an idiom in L24 | L3 | S | -- |
| L22 | Sub-computations for `reduce`, `scatter`, `sort`, `map` | L3 | L | -- |
| L23 | Indirect regions: starts read from a buffer, joined with the data-dependent addressing study | L5 | M | -- |
| L27 | Apply the glossary to B, O, and A docs | L25 | M | -- |
| L28 | Banned-synonym lint over the D10 word list | D10, L26 | S | -- |

## backend

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| B22 | Merged into B21 | -- | - | -- |
| B23 | Cut: line agreement (D3) is U4's | -- | - | -- |

## oracle

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| O4 | One word table: folded into O3 | O3 | - | R30 |

## act

| ID | Item | Depends on | Size | Rulings |
| -- | ---- | ---------- | ---- | ------- |
| A6 | Comment pass on docker/ scripts | -- | S | -- |
| A13 | Give every exercise of the new tutorial a runnable solution | A24 | S | -- |
| A18 | Add a scheduled compatibility build | A16 | M | -- |
