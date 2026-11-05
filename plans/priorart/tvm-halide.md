# TVM tensorize, TensorIR intrinsics, and Halide

## What it is

- TVM tensorize declares an intrinsic twice: a `te.compute` description of its semantics and an `intrin_func(ins, outs)` that emits the call, returning `(body, reduce_reset, reduce_update)`; `decl_tensor_intrin` binds them with `decl_buffer` strides and `offset_factor`.
- TensorIR keeps the same pair as a description block and an implementation block, matched structurally against a scheduled block.
- Halide separates algorithm from schedule; `FindIntrinsics.cpp` lifts arithmetic to intrinsics with a `rewrite(pattern, replacement, predicate)` table where order matters and each lifted intrinsic needs a matching lowering.
- Sources: https://github.com/apache/tvm/blob/v0.16.0/gallery/how_to/work_with_schedules/tensorize.py, https://halide-lang.org, https://github.com/halide/Halide/blob/main/src/FindIntrinsics.cpp.

## TAIDL today, for comparison

- TAIDL declares semantics once and generates both matcher and emitter, which removes tensorize's description-versus-implementation drift.
- The reduce split (reset and update) is what TAIDL expresses through accumulate-style instructions with an input that is also written (`set_inplace`, S5.5).
- Halide's ordered rule table is the shape of `ir2ir_rewrites.txt` (act-backend/act_backend/generic/src/isel/rewrites/ir2ir_rewrites.txt:1-10), but egg applies rules without order.

## Borrow

- Explicit stride and alignment contract: `offset_factor` and symbolic `strides` tell the matcher what layouts an intrinsic accepts; TAIDL regions have starts and sizes but no stride or alignment word, and the allocator's assumptions live in generated C++ (`generate_known_integer_checks`, act-backend/act_backend/isel_ir2isa_rewrites_rs_generator.py:528).
- Partial-reduction protocol: name in the spec how an instruction that accumulates over part of a contraction is composed, so the backend's `(add (dot ...) ?d)` split rule (ir2ir_rewrites.txt:2) has a stated contract.
- Failure modes as documentation: the tensorize tutorial lists what breaks a match; the backend's coverage.md should list what makes an instruction unselectable.

## Avoid

- Two descriptions per instruction that can drift; TensorIR mitigates with structural matching, TAIDL avoids it by construction.
- Ordered peephole tables: Halide's comments warn that the simplifier undoes matches and that order changes results; keep rules unordered with explicit priorities only where needed.

## Relevance

- Medium.
