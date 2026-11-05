# Response to the ambiguity review

Rows answer `reviews/spec/ambiguity.md` in its order: N0 is the proposed norm, T1-T36 the findings table, G1-G4 the grammar bullets, M1-M11 the missing S1.3 entries, and S1-S6 the missing S9 entries.
"Pending panel" names the tech-lead ruling the change waits on.

| Finding | Disposition | Reason | Changed |
| --- | --- | --- | --- |
| N0 one norm for decision 16 | accepted | Owner decision 16 requires one defined norm; the fault wording gives one | S1.3, S5.6, S10 `fault` |
| T1 S7.5 `constant` literal fit | accepted | B and O disagree today (`spec_checks.py:94-107`, `utils.py:398-426`); front end skips operands (`instruction.py:424`) | S7.5, S1.3.9, S9; L16 |
| T2 S6.2 overflow rules contradict | accepted | Two rules in one section; O computes exactly (`utils.py:268-271`) | S6.2, S1.3.1; L6, D7 |
| T3 S5.3 constraint vs start state | accepted | Placement constraints compare starts (`instruction.py:216-218`) that read post-update state; the all-after variant is pending panel (R7) | S5.3, S5.6; D30 |
| T4 S8.4 float accumulation order | pending panel (R8) | A fixed order binds XLA-backed oracles to one schedule; the tolerance alternative needs agreed values | S1.3.15, S8.4; L31, D32 |
| T5 S8.4 reduce initial value | accepted | Matches O's `reduce_init` (`utils.py:439-452`); owner decision 16 | S8.4; L4, D12 |
| T6 storage before first write | accepted | No line of spec or code states it; owner decision 16 | S5.6; D31 |
| T7 S5.5 in-place aliasing | accepted | Owner decision 17; front end checks only the flag count (`instruction.py:163-172`) | S5.5, S1.3.6, S9; L5, F21, D29 |
| T8 byte order and sub-byte packing | accepted | O stores `pred` as one byte and packs sub-byte types (`utils.py:880-908`) | S8.4, S1.3.14; F6 |
| T9 negative start, `d0` bound | accepted | A negative start is parsed (`instruction.py:136`); `d0` now has one access dimension sized by `hbm` | S4.3, S5.2, S1.3.3; F24, L5 |
| T10 version pin | partly | Pin is the oracle's jax (L2: `act-env:py3.14-jax0.11.2`), not a literal release string, so one pin serves spec and oracle | S8.4; L2 |
| T11 integer edge cases in a semantics | accepted | O computes XLA CPU values for these today (semantics holes H17); owner decision 16 | S8.4, S1.3.12 |
| T12 S7.2 partial instruction never checked | accepted | `check_values` runs only when both are given (`instruction.py:99-100`, `254-257`) | S5.2, S9; L11, F27 |
| T13 S4.5 late `add_data_model` | accepted | Accepted today and the list is shared by reference (`accelerator.py:47-58`, `instruction.py:81`) | S4.5, S9; L11 |
| T14 S7.4 dimension 0 | accepted | `dim : INT` accepts 0 (`IDLV2.g4:23`) | S7.4, S1.3.7, S9; L16, F28 |
| T15 S7.2 count or type mismatch | accepted | Owner decision 16; backtick shapes are known only at issue | S7.2, S8.3, S1.3.8; L12 |
| T16 overlapping outputs | accepted | Owner decision 16 | S5.2, S1.3.4; L5 |
| T17 failed constraint | accepted | Owner decision 16 | S5.3, S1.3.5 |
| T18 computed divisor 0 | accepted | Owner decision 16; O already rejects at the call (`utils.py:292-301`) | S6.2, S1.3.2 |
| T19 `@a.` in another place | accepted | Statically checkable (`instruction.py:339-358`) | S6.1, S9; L10 |
| T20 initial value past 64 bits | accepted | Statically checkable; only the type is checked (`accelerator.py:69`) | S4.4, S9; L16 |
| T21 S6.3 short circuit | accepted | A fault in an unevaluated arm must not depend on the generator | S6.3, S5.6 |
| T22 `compare` default `type` | accepted | O uses the StableHLO default by element type (`oracle_visitor.py:114-118`) | S8.4 |
| T23 `dot` mixed types | partly | One operand type is stated; how the result type accumulates is part of T4, pending panel (R8) | S8.4; L31 |
| T24 S8.3 checked operand | accepted | `select` and `clamp` first operands are `pred` and the bound | S8.3 |
| T25 `INT` base | accepted | A C reader with base 0 reads `010` as 8 | S2.1, S6.2 |
| T26 `-inf`, `-nan`, NaN pattern | accepted | `NUMBER` takes the sign (`IDLV2.g4:91`) | S2.1, S7.5 |
| T27 least 64-bit literal | accepted | Literals are unsigned (`instruction.py:408`) | S6.2 |
| T28 `(0)` and `-0` divisors | accepted | The check compares token text (`instruction.py:414`); the spec now matches it and faults the rest | S6.2, S9; L14, F10 |
| T29 semantics with no `%Out` | accepted | Rejected today (`instruction.py:311-318`) | S7.1, S9 |
| T30 front end may reject more | accepted | Two front ends must accept one set | S9 |
| T31 S7.1 bare-word sentence | accepted | Mixed lexer fact and placement | S7.1 |
| T32 starts compared by text | accepted | `getText` keeps parentheses and order (`instruction.py:214`) | S5.5 |
| T33 S8.5 run-time shapes | accepted | Backtick dimensions depend on attributes and state | S8.5 |
| T34 Python keyword list | partly | Soft keywords allowed; the list is the running Python's (`names.py:11`), since no Python version is pinned | S4.1 |
| T35 non-`str` list entries | accepted | `strings` does not check entries (`instruction.py:452-456`) | S5.1, S9; L16, F25 |
| T36 `bool` and tuple in S9 | accepted | Code rejects `bool`, accepts a tuple (`accelerator.py:52-53`, `69`) | S4.3, S4.4, S9; L16 |
| G1 S3.2 omits `compare` | accepted | Evidence the copy drifts; the page now names the `.g4` files | S3; D9, L8 |
| G2 S2.2 longest match and `WS` | accepted | `IDLV2Expr.g4:62` | S2.2 |
| G3 S2.1 `COMMENT` and `WS` rows | accepted | `IDLV2.g4:96-97` | S2.1 |
| G4 `-inf` and `-nan` lex as `NUMBER` | accepted | `IDLV2.g4:91` | S2.1 |
| M1 late buffer declaration | accepted | As T13 | S4.5, S9 |
| M2 regions without semantics | accepted | As T12 | S5.2, S9 |
| M3 dimension 0 | accepted | As T14 | S7.4 |
| M4 `constant` literal range | accepted | As T1 | S7.5 |
| M5 negative start, `d0` past end | accepted | As T9 | S5.2 |
| M6 contents before first write | accepted | As T6 | S5.6 |
| M7 float summation order | pending panel (R8) | As T4 | S1.3.15; L31 |
| M8 integer edge cases | accepted | As T11 | S8.4 |
| M9 `inf` at `f8e4m3fn`, NaN pattern | accepted | O rejects the first (`utils.py:411`) | S7.3, S7.5 |
| M10 `and` and `or` evaluation | accepted | As T21 | S6.3 |
| M11 non-`bool` mark flags | accepted | Truthiness is used (`instruction.py:168`, `203`) | S5.5, S9; L16, F26 |
| S1 no `%Out<k>` line | accepted | As T29 | S9 |
| S2 start or size neither `int` nor `str` | accepted | `instruction.py:136-137` | S9 |
| S3 Python `int` start past 64 bits | accepted | `instruction.py:408` | S9 |
| S4 `%In01` | accepted | `instruction.py:150-161` | S7.2, S9 |
| S5 name not a `str` | accepted | `names.py:9` | S4.1, S9 |
| S6 non-`str` list entry | accepted | As T35 | S9 |
