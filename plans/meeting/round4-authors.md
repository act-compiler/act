# Round 4: authors apply panel 10 and owner decision 40

Each author revises its own folders; the writing rules of round 3 still hold.

## Inputs

- Panel 10 decision: plans/meeting/panel-10/decision.md (width layer, encoding layer, derived attributes, generator metadata, all in v3, encoding layer last).
- Owner decision 40: nothing fails without a genuine cause.
  - The front end and the generators reject only problems the architect can correct, with a clear error.
  - A generated oracle stops with an error only on invalid assembly code or binary input, and otherwise runs correctly.
  - A generated backend drops a failing candidate with a recorded reason and stops only when the input kernel (StableHLO, XLA HLO, torch-mlir, or torch.fx) is invalid or outside the input specification.
- Vocabulary: "generator metadata" (one generator's block is a "metadata entry"); "width layer", "encoding layer", "derived attribute", "marker", "shorthand"; never "add-on".
- Owner decision 39: adding a new generator must be easy; plan for a documented extension path.

## Writing rules

- Replace text in place; docs state what is; ASCII only; one sentence per line; bullets by default; tables for per-item data.
- New plan items take the next free number in your track; keep existing IDs.
- Files under about 150 lines; one README.md index per folder; only README.md files link.
- Use Edit and Write for files; no git; read-only shell otherwise.
