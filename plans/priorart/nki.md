# AWS NKI ISA documentation style

## What it is

- Each `nki.isa` instruction has one page: signature, one-line formula such as `dst = stationary.T @ moving`, the engine (Tensor, Vector, Scalar, GpSimd, DMA), feature sections (performance mode, tiling, accumulation, transpose), memory types (inputs from SBUF, output to PSUM), data-type rules, layout, tile-size limits, profiler view, parameters, and one runnable example.
- Rules are quoted sentences: "The partition dimension sizes of the stationary and moving tiles must be identical", "if one of the input tiles is tfloat32/float32, the other tile must also be tfloat32/float32".
- Undefined behavior is named per feature: "Using accumulate=True on an uninitialized PSUM location is undefined behavior on all targets."
- Hardware generations are named inline: "float8_e4m3fn is added on NeuronCore-v4", "Moving free size is capped at 512 on v3".
- Source: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/nki.isa.html and nki.isa.nc_matmul.

## TAIDL today, for comparison

- A TAIDL instruction's facts are spread over calls and have no rendered page; the mini_trn example (act-oracle/examples/mini_trn/mini_trn.py) is the closest thing to documentation.
- Constraints such as NKI's dtype pairing are TAIDL constraints over `@c.` attributes (S5.3) but element-type pairing across regions has no place to be stated, since a data model has one element type (S4.3).
- Per-generation limits are state or constraint values in TAIDL; NKI shows the owner's target audience expects them in prose beside the instruction.

## Borrow

- A generated instruction page: from one `Accelerator`, emit per-instruction documentation with the signature, data models read and written, attributes with constraints, state updates, the semantics block, undefined cases, and one example kernel line with its oracle result.
- Engine as a declared instruction property: a tag the cost model and scheduling can read, declared in the instruction, not inferred.
- Per-feature undefined-behavior notes: each S1.3 case that applies to an instruction is listed on its page.
- Generation variants as one accelerator with a read-only state variable, documented on the page as "on v4".

## Avoid

- Prose as the only source of a rule: NKI's constraints are sentences a compiler cannot check; in TAIDL every page sentence is rendered from a constraint or a table row.
- Performance claims in the ISA spec: "2x throughput" belongs in a cost model, not in the semantics.

## Relevance

- Medium: a documentation model, not a semantics model.
