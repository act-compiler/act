# Width layer, derived attributes, and the assembler unit

Panel 10 adds an optional width layer (a type `s<w>` or `u<w>` per attribute, default `s64`) and an optional encoding layer (an encoding block per instruction, with the derived-attribute rule).
The backend reads the types and the links; it never reads the encoding block and keeps printing text assembly (panel 10, Q10.2).
B38 is rewritten below; B44 and B45 are new.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| B38 | Allocator attribute domains follow the width layer (D15); default `s64` | width layer in the front end | S | Q1 |
| B44 | Derived attributes solved in closed form in the allocator | B38, B30, encoding layer in the front end | M | Q2 |
| B45 | Assembler and disassembler generated from the encoding layer, landing last as one unit | B44, encoding layer, round-trip corpus | L | Q2 |

## Order inside v3

- Layer 1 (width layer) lands first with B38; layer 2 (block, derived rule, B44, B45, round-trip test) lands last as one unit (panel 10, Q10.4).
- The unit never ships in part; if time runs short the v3 date moves and the scope does not.
- Until the width layer lands, B38 uses the default `s64` domain; an ISA specification with no types behaves as today apart from negative values becoming selectable.

## B38 Allocator attribute domains (D15)

- What: each addressing attribute's solver variable spans its declared range: `s<w>` gives -2^(w-1) to 2^(w-1) - 1, `u<w>` gives 0 to 2^w - 1, and an untyped attribute is `s64`; this replaces `MAKE_INT_VAR(0, INT64_MAX)` (`malloc_instructions_h_generator.py:456`).
- What: computational attributes (`:451`, emitted as `int64`) take the same declared range as a check on the matched value, so a value outside its type is never selected.
- What: the implicit bounds (start at or above 0, end within the buffer) apply to the evaluated address only (owner decisions 34, 38).
- What: a domain the solver cannot hold is a generation error naming the attribute; the domain is never clamped silently.
- Preserves: every address bound the allocator enforces today; attribute names stay plain variable names (owner decision 34).
- Benefits: a start such as `1024 - @a.addr_out` is selectable; the search never picks a value the type forbids, so an encoded program never holds an unencodable value.
- Caveats: floor division and remainder over a negative attribute keep the divisor-sign rule (`solver.h:46-80`); `solver.h:33` already uses `INT64_MAX - 1` as headroom, so the full `s64` domain fits; a wider domain slows search, measured under B34.
- Fail points: a solve that picks an address outside its buffer; noticed by a kernel test with a negative attribute value and one with a `u8` attribute.
- Order and dependencies: the type read needs the front end's typed model (U2); the default-domain part has no dependency.
- Verification: a kernel whose only valid attribute value is negative compiles and the oracle run agrees; a `u8` attribute never exceeds 255 in the `--log` allocation; the generated C++ no longer contains `INT64_MAX` for an attribute.

## B44 Derived attributes in closed form

- Applies only to an instruction with an encoding block; the front end has checked that each derived attribute D has exactly one link `E == k*D + c` or `D == k*E + c`, with E one encoded attribute and k, c integer literals (panel 10, Q10.3).
- What: the allocator gets E as the variable, with its type domain; D is never a solver variable.
- What: for `D == k*E + c` the generator emits D as the expression `k*E + c`.
- What: for `E == k*D + c` the generator emits D as `(E - c) / k` and adds the constraint `(E - c) % k == 0`, using the floor operations of `solver.h:26-33`.
- What: the link is a linear constraint in the model, routed to the solver as `malloc_instructions_h_generator.py:510` routes address constraints.
- Preserves: the text assembly the backend prints; a specification without a block, which has no derived attribute.
- Benefits: search never emits a value the decoder cannot invert, and no generator runs a solver for a derived attribute.
- Caveats: the closed form needs k nonzero (checked by the front end); a link naming state or another derived attribute is rejected there, not here.
- Fail points: an inexact division accepted; noticed by a kernel test whose only valid E has no integer partner and must drop the candidate with a reason.
- Order and dependencies: after B38 and B30; before B45, which shares the corpus ISA specification.
- Verification: the corpus ISA specification (an `A == B*8` link) compiles a kernel; every printed `A` is a multiple of 8; the oracle run agrees.

## B45 Assembler and disassembler generator

- What: one generator takes a `Model` or an `Accelerator`, reads only the `Model` that `check` returns, and emits an assembler and a disassembler per ISA specification with an encoding block; it exposes `generate_<name>(accelerator, base_dir=None, **options)` like any generator (owner decision 39).
- What: generation checks reject, naming the instruction and the bit or attribute: a word bit covered by no slice or by two, two instructions whose fixed bits a word matches, an encoded attribute whose type width differs from its slice total.
- What: encode evaluates the link; decode inverts it by one subtraction and one exact division; a word matching no instruction is the defined decode error (owner decision 40).
- What: the unit lands with the round-trip test `decode(encode(x)) == x` on a corpus ISA specification with a small word, a scattered immediate, and an `A == B*8` link, and with the expectations table of the generator tests.
- Preserves: the backend and the oracle, which never read the block.
- Benefits: a binary program for the accelerator, from the same ISA specification, with the decoder as its inverse.
- Caveats: package home (a new package or inside act-backend) is a packaging choice for the roadmap owner; it needs no allocation, so it shares no code with the allocator.
- Fail points: a scattered slice order reversed; noticed by the round-trip corpus.
- Order and dependencies: last of the v3 backend set; depends on B44 and on the front end's layer 2 items; ships with them or not at all.
- Verification: the round-trip passes on the corpus; each generation check has a failing ISA specification; the neutrality test (output of oracle and backend equal with the block stripped) passes.
