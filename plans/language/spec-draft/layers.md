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

- An attribute with no declared type has type `s64`.
- An ISA specification with no encoding block has no derived attribute and no binary form.
- A layer changes what every conforming generator must do; generator metadata (S12) never does.
- A generator that does not read a layer still applies its constraints, since each type expands to constraints (S11.2).

## S11.2 The width layer

- An attribute declares its type in its annotation, such as `addr: Addr[u16]` or `off: Comp[s8]`, with type objects of `taidl.types`.
- A type is `s<w>` for w in 1 to 64, the range -2^(w-1) to 2^(w-1) - 1.
- A type is `u<w>` for w in 1 to 63, the range 0 to 2^w - 1.
- `u64` is rejected, since every value lies in signed 64 bits.
- A type is a closed range: a value never wraps, truncates, or sign-extends.
- A type is an implicit constraint (S5.3): it expands to two range constraints on the attribute, like a marker.
- An attribute value outside its type at issue is a fault (S1.3.10).
- No name implies a type (S5.1); the implicit region constraints still bind the evaluated start only (S5.2).
- A compiler backend takes an addressing attribute's declared range, `s64` by default, as its search domain.

## S11.3 The encoding layer

- An instruction may carry one encoding block, declared beside it; it is pure bit placement, read by an assembler and disassembler generator.

```python
acc.encode(gemm_acc, word=32, fixed=[((31, 24), 0x12)],
           fields={'a1': [(23, 16)], 'a2': [(15, 8)], 'out': [(7, 0)]})
```

- `acc.encode(instr, word, fixed, fields)` takes the instruction function, the word width, the fixed bits, and the fields.
- The block never edits the instruction; a field key names a parameter of `instr`, and an unknown key is rejected.
- The block gives the word width in bits, fixed and explicit; encodings are fixed length and one word.
- The block gives the fixed bits, as `((high, low), value)` pairs: the opcode and every reserved bit with its value.
- The block gives, per encoded attribute, an ordered list of slices `(high, low)`, high to low, whose concatenation is the field.
- A field holds its attribute value as is: no scale, offset, or implied bit; a signed type packs as two's complement.
- A scale, an offset, or an implied low bit is a derived attribute with a link (S11.4).
- The oracle and the compiler backend never read the block; they read the types and constraints it implies.

Checks, each rejected with an error naming the instruction and the bit or attribute; `check` runs the first and the third, and the generator that reads the blocks the second:

- Every bit of the word is covered by exactly one slice or one fixed bit.
- No word matches the fixed bits of two instructions.
- Each encoded attribute has a declared type whose width equals the total width of its slices.

## S11.4 The derived-attribute rule

- The slice list of the block names the encoded attributes; every other attribute of that instruction is derived.
- A derived attribute D has exactly one link: a written constraint `require(E == k*D + c)` or `require(D == k*E + c)`.
- In a link, E is one encoded attribute, k and c are plain integers, and k is nonzero.
- A derived attribute may be negative and may carry its own type.
- Rejected, with an error naming the derived attribute:
  - a derived attribute with no link;
  - a derived attribute with two links;
  - a link that is not affine in that form;
  - a link naming another derived attribute, or two encoded attributes.
- Encode evaluates the link; decode inverts it by one subtraction and one exact division.
- A value with no integer partner, or outside its field, is a violated constraint (S1.3.5).
- For example, `A == B*8` with A encoded: decode of A = 12 names no B and faults; encode of any B in range succeeds.
- A word that matches the fixed bits of no instruction is a decode error (S1.3.18).
- A compiler backend takes each encoded attribute as a variable over its type and each link as a linear constraint, so it never emits an unencodable value.
