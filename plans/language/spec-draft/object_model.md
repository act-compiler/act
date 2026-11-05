# Object model

This page is S4.6 of the TAIDL language specification: what a generator reads from an accepted ISA specification.
The typed contract, its node classes, its members, and `CONTRACT_VERSION` are shared plan item U2; the `taidl` README documents each member with its type.

## S4.6 The `Model`

- `taidl.check(acc)` returns a `Model`, and only `check` constructs one (S9).
- A `Model` is frozen: no member of it or of its nodes can be set after `check` returns.
- Every collection the front end builds in a node is a tuple: attributes, regions, constraints, slices, fixed bits, and the encoded and derived lists.
- Every read of a node's metadata returns a fresh deep copy in the types of S12.1, dicts and lists, so nothing mutable is reachable in the `Model`.
- It holds one typed node per buffer, state variable, and instruction, and per region, constraint, state update, and value of each instruction.
- A generator reads only the `Model`, never the `Accelerator` or a trace record.

| node | members |
| --- | --- |
| accelerator | its name, `base_dir`, metadata, and its buffer, state, and instruction nodes |
| buffer | its name, access dimensions, base element shape, element type, initial value or `None`, metadata, and whether it is `d0` (`is_hbm`) |
| state variable | its name and its value before the first issue |
| instruction | its name, attributes, input regions, output regions, constraints, state updates, values, overlap relations, encoding block, encoded and derived attributes, links, and metadata |
| attribute | its name, its kind, its type, and whether the type was declared |
| region | its buffer, start expressions, and size expressions |
| constraint | its condition and its source tag |
| value | its tensor operator, operands, operation attributes, inferred element type, and inferred shape |

- The buffer nodes include `d0`, marked `is_hbm`.
- A generator's `base_dir=None` means the accelerator node's `base_dir`.
- An instruction node lists its attributes in parameter order.
- It lists its input regions in `read` order and its output regions in `write` order, as two tuples; a region object that is read and written gives one node in each.
- Its constraints include those every source of S5.3 expands to, each tagged with its source.
- The tag is one of `written`, `region`, `output_pair`, `relation`, `type`.
- A `relation` constraint names its marker and pair; a `type` constraint names its attribute; a `type` constraint exists only for a declared type.
- Every implicit constraint is recorded unfolded, one per source, so two `Model` objects of one ISA specification hold equal tuples.
- An implicit constraint of a `d0` region refers to the `hbm` size as one expression leaf.
- Each same-buffer input-output pair carries its relation and, for `inplace`, the tie (S5.5); a backend reads that record.
- An instruction node holds its encoding block, or `None`: the word, the fixed bits as (high, low, value), and per encoded attribute its slices.
- It lists the encoded and the derived attributes, and per derived attribute its link as (encoded attribute, k, c, orientation); the link also stays among the constraints.
- The block is `None` and the tuples are empty when the encoding layer is absent.
- Expressions and conditions are typed trees over plain integers, attribute and state references, and the `hbm` size.
- Generator metadata sits on the accelerator, buffer, and instruction nodes, by namespace (S12).
- `CONTRACT_VERSION` is a string `major.minor`, also `Model.contract_version`; an added member bumps the minor, and a removed one is deprecated for one minor, then bumps the major.
- A generator states its major and its lowest minor, and rejects a `Model` outside that window with an `X` error naming both versions.

## Call sites

- Every node records its call site: the file and line of the first frame outside `taidl`; a whole-instruction node records the function's first line.
- A constraint that `check` expands takes the call site of its source: a region's `buffer.region` call, a relation's `write`, or, for a type, the function's first line.
- Code with no file records `<string>` as the file.

## Printing

- `str(model)` and `str(instruction)` give text whose semantics parses as XLA-HLO and whose other lines are comments; the text is not part of the contract.
- The text shows each instruction's attributes with their types, regions, constraints, state updates, encoding block and links, and the metadata of each node with keys sorted.
- The text spells references `@s.<name>`, `@a.<name>`, and `@c.<name>`, as every message does (S2.2).
- The text is deterministic: the same `Model` prints the same text.
- The printed text answers "what did the front end understand" and is a log format; no front end reads it back.
- Two `Model` objects are equal when every member except call sites and `base_dir` is equal, with tuples compared in order.
