# What the current design gets right

- The language is a dialect of XLA-HLO text with a closed word list, so every operation has a published reference semantics (IDLV2.g4:45-74, operations.md:9-25).
- Semantics blocks are SSA with one `ROOT` and a fixed `%In<k>`/`%Out<k>` discipline, which gives each instruction a pure function body (instruction.py:268-318, operations.py:171-217).
- Regions are the only memory interface and are read before any write, so an instruction's effect is a single simultaneous update of memory and state (instructions.md:136, instruction_codegen.py:732-737).
- Expressions are a tiny total-by-construction integer language with floor division fixed in the spec and implemented identically in both generators (expressions.md:199-200, quotient.txt, expression_writer.py:479-481).
- Literals past i64 and division by the literal 0 are rejected at parse time, in every place a text can stand (instruction.py:403-416, 419-437).
- Reference names are resolved against the instruction's own attribute lists and the declared state at parse time, so no generator sees a dangling `@c.`, `@a.`, or `@s.` (instruction.py:339-358).
- Placement marks desugar to ordinary constraints that generators see as text, so no generator needs to know the marks (instruction.py:190-231, instructions.md:167).
- The spec already enumerates its undefined and implementation-defined cases in one table, which makes the holes in `holes.md` a finite list to close (overview.md:27-45).
- Each generator publishes a coverage table citing spec sections, so a disagreement is attributable to a row (act-backend docs/coverage.md, act-oracle docs/coverage.md).
- The oracle checks shapes, windows, sizes, and constraints at every call and names the loop iteration, which is the executor behavior the rulings make normative (utils.py:651-665, 698-712, 733-746, 796-804).
- The backend refuses what it cannot honor by name before writing anything, never giving a feature another meaning (spec_checks.py:214-577).
