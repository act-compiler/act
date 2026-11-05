# act-oracle layers and generator metadata

TAIDL v3 adds optional layers to an ISA specification: a width layer, an encoding layer, and derived attributes (panel 10).
An ISA specification may also carry generator metadata (owner decision 39).
The oracle checks the width layer as constraints and ignores the rest.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| O32 | Width layer: the oracle checks attribute types as implicit range constraints | L width layer, O10, O24 | S | Q1 |
| O33 | Encoding layer, derived attributes, generator metadata: ignored; neutrality test; binary input later | O32, T | S | Q2 |

## O32 Width layer

- Breaking: yes.
- Changes: `taidl_to/instruction_codegen.py`, `runtime/` checks, `tests/`; preserves the check of a default attribute.
- What: a type `s<w>` or `u<w>` on an attribute expands to two range constraints through the front end (owner decision 30, Q10.1); the oracle receives them in its constraint list.
- What: the oracle checks them as it checks any constraint (O10): before any load, at each call and iteration, one violated-constraint fault with the clause id.
- What: an untyped attribute is `s64`; the range is the layer-0 check that O24 makes strict, so `warn_width` and its `UserWarning` go (D23).
- What: a value never wraps, truncates or sign-extends; the oracle keeps attribute values as exact integers.
- What: a typed attribute does not change the region constraints: they bind the evaluated start, never the attribute (owner decision 34).
- What: the oracle adds no width rule of its own; a type in the typed model is read from the constraint list, never from a name.
- Benefits: the oracle, the allocator and the front end share one violated-constraint path.
- Caveats: a derived attribute is an ordinary attribute with a written link constraint; the oracle checks it, and runs no encode or decode.
- Fail points: a type check that reads the attribute type separately from the constraint list; a test pins `u8` at 255 and 256, `s8` at -128 and -129.
- Order and dependencies: after the language item for the width layer, O10 and O24.
- Verification: one test per boundary, each naming the clause; an untyped attribute behaves as before.

## O33 Layers the oracle ignores

- Breaking: no.
- Changes: `tests/`, `README.md`, `docs/coverage.md`; preserves every generated oracle.
- What: the oracle never reads the encoding block; a derived attribute is checked only as a written constraint.
- What: the oracle never reads generator metadata; it reads a metadata entry only when the entry is its own, and no such entry is defined.
- What: a neutrality test generates the oracle for every corpus case carrying an encoding block or generator metadata, and for the same case with both stripped, and asserts identical output.
- What: binary input waits for the assembler and disassembler generator (panel 10 Q10.2, Q10.4): a decoder in front of the oracle turns words into instruction calls, and the oracle runs on the calls.
- What: a word that matches no instruction is the decoder's defined error, naming the word; the oracle adds no binary parsing.
- What: the decoder is a roadmap item of the language and generator tracks; this item lists only the oracle's side, the call-level entry that already exists.
- Benefits: adding a generator or a layer needs no oracle change (owner decision 39).
- Caveats: if a second generator needs a datum the oracle also needs, it is promoted to a layer with front-end validation (Q10.5), not read from metadata.
- Fail points: an oracle that reads a metadata key by accident; the neutrality test catches it.
- Order and dependencies: after O32; the test needs the corpus cases that carry data (T).
- Verification: stripped and unstripped oracles are byte-identical across the corpus.
