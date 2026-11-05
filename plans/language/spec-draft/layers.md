# Optional layers

This page is S11 of the TAIDL language specification: the width layer, the encoding layer, and the derived-attribute rule.
Each layer is optional and adds to the one below it; an ISA specification without them means exactly what S4 to S8 say.
Each declaration below is shown in the Python surface of S2; the `Model` carries it as typed nodes (S4.6).

## S11.1 Layers and the default

| layer | adds | needs |
| --- | --- | --- |
| 0 | the language of S4 to S8; every attribute and operand is a signed 64-bit integer | nothing |
| 1, the width layer | a type per attribute | layer 0 |
| 2, the encoding layer | an encoding block per instruction, and with it the derived-attribute rule | layer 1 for every encoded attribute |

- A bare `Comp` or `Addr` has type `s64` and no declared type; `Comp[s64]` declares one, and the attribute node records which (S4.6).
- An instruction with no encoding block has no derived attribute and no binary form; the assembler and disassembler generator excludes it with a printed reason.
- A layer changes what every conforming generator must do; generator metadata (S12) never does.
- A generator that does not read a layer handles the constraints of that layer as it handles any constraint (S5.3), and excludes an instruction whose constraint it cannot hold (S9).

## S11.2 The width layer

- An attribute declares its type in its annotation, such as `addr: Addr[u16]` or `off: Comp[s8]`, with type objects of `taidl.types`.
- A type is `s<w>` for w in 1 to 64, the range -2^(w-1) to 2^(w-1) - 1.
- A type is `u<w>` for w in 1 to 63, the range 0 to 2^w - 1.
- `u64` is rejected, since every value lies in signed 64 bits.
- A type is a closed range: a value never wraps, truncates, or sign-extends.
- A type is an implicit constraint (S5.3): it expands to two range constraints on the attribute, like a marker.
- An attribute value outside its type at issue is a fault (S1.3.10).
- It is also a failed constraint (S5.3); when a type constraint and a constraint of another source both fail on one issue, S1.3.10 is reported.
- A type bounds the attribute's value at issue and nothing else: an expression over typed attributes is a signed 64-bit integer (S6.2).
- A type never changes folding, shape inference, or the checks of S7.4 and S7.5.
- No name implies a type (S5.1); the implicit region constraints still bind the evaluated start only (S5.2).
- A compiler backend takes an addressing attribute's declared range, `s64` by default, as its search domain.
- A compiler backend treats the type of a computational attribute as a condition of a match: a kernel value outside it drops that candidate with a recorded reason.
- A type constraint is never a reason to exclude an instruction.

## S11.3 The encoding layer

- An instruction may carry one encoding block, declared beside it; it is pure bit placement, read by an assembler and disassembler generator.

```python
acc.encode(gemm_acc, word=32, fixed=[((31, 25), 0x09)],
           fields={'flag': [(24, 24)], 'a1': [(23, 16)], 'a2': [(15, 8)], 'out': [(7, 0)]})
```

- `acc.encode(instr, word, fixed, fields)` takes the instruction's handle, which `@acc.instruction` returns and `acc.encode` matches by identity (S5.1), the word width, the fixed bits, and the fields.
- In a family the call sits in the loop body, once per member; a member with no call has no block.
- An `instr` that is not an instruction of `acc`, a second block for one instruction, or a call before the instruction is declared is rejected (S9).
- The block never edits the instruction; a field key names a parameter of `instr`, and an unknown key is rejected.
- The block gives the word width in bits, fixed and explicit, an `int` of at least 1 with no upper limit; encodings are fixed length and one word.
- All blocks of one ISA specification share one `word`; a block with another one is rejected.
- The block gives the fixed bits, as `((high, low), value)` pairs: the opcode and every reserved bit with its value.
- The block gives, per encoded attribute, an ordered list of slices `(high, low)` whose concatenation is the field.
- The list runs from the most significant part of the field to the least: the first slice supplies the top bits of the field.
- Bit 0 is the least significant bit of the word; a slice `(high, low)` has `high >= low` and holds bits `high` down to `low`.
- The list need not descend in word position, as in `'imm': [(31, 31), (7, 7), (30, 25), (11, 8)]`.
- A field holds its attribute value as is: no scale, offset, or implied bit; a signed type packs as two's complement.
- A scale, an offset, or an implied low bit is a derived attribute with a link (S11.4).
- The oracle and the compiler backend never read the block; they read the types and constraints it implies.
- A binary program is a sequence of `word`-bit unsigned words indexed from 0; a message shows a word in hexadecimal.
- Its byte form is `ceil(word/8)` bytes per word, little-endian by default, with the top `8*ceil(word/8) - word` bits of each word zero.
- `byte_order='big'` is an option of the assembler and disassembler generator.

Checks, each rejected with an error naming the instruction and the bit or attribute; `check` runs all three, and no generator repeats one:

- Every bit of the word is covered by exactly one slice or one fixed bit.
- No two instructions conflict; two conflict when their fixed bits agree on every bit position both fix.
- Each encoded attribute has a declared type whose width equals the total width of its slices.

## S11.4 The derived-attribute rule

- The slice list of the block names the encoded attributes; every other attribute of that instruction is derived.
- A derived attribute D has exactly one link to one encoded attribute E.
- A link is an `==` directly under `require` whose two sides fold, over `+`, `-`, unary `-`, and `*` by a plain integer, to `a*E + b*D + c0 == 0` with `a` or `b` in {1, -1}.
- The `Model` stores the link as (E, k, c, orientation), `E == k*D + c` or `D == k*E + c`, so a generator never sees its spelling; `E + 1 == 8*D` is a link with `a` = 1.
- A condition with `//`, `%`, or an attribute other than E and D, or one inside `any_of`, is not a link.
- A derived attribute may be negative and may carry its own type.
- Rejected, with an error naming the derived attribute:
  - a derived attribute with no link;
  - a derived attribute with two links;
  - a link whose fold leaves signed 64 bits, or gives k = 0;
  - a link naming another derived attribute, or two encoded attributes.
- Every other condition is an ordinary constraint, including one that names a derived attribute, such as `D < 100`, `D == 5`, or `E1 == E2 + 1`.
- A condition of the link form that names two derived attributes is rejected as naming another derived attribute.
- For `E == k*D + c`, encode computes E = k*D + c and decode computes D = (E - c) / k.
- For `D == k*E + c`, decode computes D = k*E + c and encode computes E = (D - c) / k.
- The division is exact; a value with no integer partner is a violated constraint (S1.3.5).
- For example, `A == B*8` with A encoded: decode of A = 12 names no B and faults; encode of any B in range succeeds.
- Decode of a word matches the fixed bits, reads each field (two's complement for a signed type), applies each link with exact division (S1.3.5), and checks each derived value against its type (S1.3.10).
- Decode evaluates no other written constraint; the oracle checks it at issue.
- A word that matches the fixed bits of no instruction is a decode error (S1.3.18).
- Decode of a binary program stops at the lowest failing word index.
- A compiler backend takes each encoded attribute as a variable over its type and each link as a linear constraint, so it never emits an unencodable value.
- When the value a kernel gives a derived attribute has no integer encoded partner, or the partner is outside its type, the backend drops that candidate with a recorded reason.
