# What the current design gets right

- The language is a dialect of XLA-HLO text with a closed word list, so every tensor operator has a published reference semantics (IDLV2.g4:45-74, S8.1).
- Semantics blocks are SSA with one `ROOT` and a fixed `%In<k>`/`%Out<k>` discipline, which gives each instruction a pure function body (instruction.py:268-318, operations.py:171-217).
- Regions are the only memory interface, so an instruction's effect is a single update of memory and state at the end of its issue (S5.6, instruction_codegen.py:162-181).
- Expressions are a tiny integer language with floor division fixed in the spec and computed by the same rule in both generators (S6.2, quotient.txt, expression_writer.py:15-24); they are total only once overflow and a computed zero divisor are faults (plan F3).
- Literals past i64 and division by the literal 0 are rejected at parse time, in expressions, dims, and attributes (instruction.py:403-437); a computed divisor is a fault at issue.
- Reference names are resolved against the instruction's own attribute lists and the declared state at parse time, so no generator sees a dangling `@c.`, `@a.`, or `@s.` (instruction.py:339-358).
- Placement marks desugar to ordinary constraints that generators see as text, so no generator needs to know the marks (instruction.py:190-231, S5.5).
- The spec enumerates its former undefined and implementation-defined cases in one table, which makes the holes in `holes.md` a finite list to close (S1.3).
- Each generator publishes a coverage table citing spec sections, so a disagreement is attributable to a row (act-backend docs/coverage.md, act-oracle docs/coverage.md).
- The oracle checks shapes, regions, sizes, and constraints at every call and names the loop iteration, which is the executor behavior the rulings make normative as faults (utils.py:651-665, 698-712, 733-746, 796-804).
- The backend rejects what it cannot apply by name before writing anything, never giving a feature another meaning (spec_checks.py:22-418).
