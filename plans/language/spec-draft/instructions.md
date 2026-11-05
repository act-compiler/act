# Instructions

This page is S5 of the TAIDL language specification.
An instruction reads its input regions, its attributes, and the state, and writes its output regions and the state.
This instruction is the running example of S5 to S7:

```python
from taidl import Accelerator, Comp, Addr, read, write, require, update, ops
from taidl.types import u1, u8, bf16, f32

acc = Accelerator('mini_trn')
d1 = acc.buffer('d1', access_dims=[128], base_element_dims=[64], element_type=bf16)
accum = acc.buffer('accum', access_dims=[1024], base_element_dims=[64], element_type=f32)
calls = acc.state('calls', 0)

@acc.instruction(meta={'allo': {'engine': 'tensor'}})
def gemm_acc(flag: Comp[u1], a1: Addr[u8], a2: Addr[u8], out: Addr[u8]):
    require(a1 != a2)
    a = read(d1.region([a1], size=[64]))
    b = read(d1.region([a2], size=[64]))
    target = accum.region([1024 - out], size=[64])
    old = read(target)
    prod = ops.convert(ops.dot(a, b, lhs_contracting_dims=[1], rhs_contracting_dims=[0]), f32)
    keep = ops.broadcast(ops.compare(ops.constant(flag), ops.constant(1), direction='EQ'), [64, 64], dimensions=[])
    write(target, ops.select(keep, old, ops.constant(0, f32, [64, 64])) + prod, inplace=old)
    update(calls, calls + 1)
```

## S5.1 Instruction attributes

- `@acc.instruction(name=None, meta=None)` declares the function below it as an instruction (S3.1).
- Each parameter of the function is one instruction attribute, in parameter order; an issue gives each a value by name (S5.6).
- The annotation gives the kind: `Comp` for a computational attribute, `Addr` for an addressing attribute.
- `Comp[t]` and `Addr[t]` add a type `t` of the width layer (S11.2); a bare `Comp` or `Addr` has type `s64`.
- A parameter with no annotation, another annotation, a default value, or `*args` or `**kwargs` is rejected.
- Each attribute takes an integer value in its type each time the instruction is issued; a negative value is legal unless its type is `u<w>`.
- An attribute name is a plain variable name; no tool infers meaning from it.
- A computational attribute configures what the instruction computes and may appear in any expression.
- An addressing attribute may appear only in a start or a constraint (S6.1).
- A start is any expression over attributes and state, such as `1024 - out`.

## S5.2 Regions: inputs and outputs

- `buffer.region(starts, size=sizes)` makes a region: a list of one start per access dimension and, by keyword, a list of one size each.
- `d0` has one access dimension, so its regions are `acc.d0.region([s], size=[n])`.
- A start or a size is a plain `int` or an integer expression (S6).
- `read(r)` makes `r` the next input region and returns its region value; input region k is the k-th `read`, counting from 0.
- `write(r, value, ...)` makes `r` the next output region and writes `value` to it; output region k is the k-th `write`.
- A region object may be both read and written, as `target` above; each `read` and each `write` is one region of the instruction.
- Along each access dimension, a region covers the indices from its start to its start plus its size, exclusive.
- Along the base element shape, a region covers every index.
- The region shape is the sizes followed by the base element shape.
- An input region and an output region of one buffer overlap only as their declared relation allows (S5.5).
- Every region carries implicit constraints (S5.3) on its evaluated starts and sizes, never on attribute values: each evaluated start is at least 0, each size at least 1, and each start plus size at most its access dimension; a violated one is a fault (S1.3.3).
- For `d0` the access dimension is the byte count of the `hbm` contents (S5.6).
- Two output regions of one instruction that overlap are a fault (S1.3.4).
- An instruction with no `read` and no `write` only changes state.

## S5.3 Constraints

- `require(c1, c2, ...)` adds each condition (S6.3) as a constraint; several calls add to one conjunction.
- One constraint mechanism holds four sources: constraints the ISA author writes, the implicit constraints of every region and every output pair (S5.2), the constraints each input-output pair's relation expands to (S5.5), and the range constraints of each declared type (S11.2).
- A constraint of any source reads the state before the issue, as every expression of the issue does (S5.6).
- Issuing an instruction where a constraint of any source does not hold is a fault (S1.3.5).

## S5.4 State updates

- `update(var, expr)` records that state variable `var` takes the value of the integer expression `expr`.
- Updates are one parallel assignment: every update expression reads the state before the issue, and all updates take effect together at the end of the issue.
- Two updates of one state variable in one instruction are rejected.
- A read of `var` after `update(var, ...)` in the same function still reads the state before the issue.
- An update that needs another update's new value writes it out: with `update(a, a + 64)`, the new `a` is `a + 64`.
- Post-increment addressing is start `ptr` with `update(ptr, ptr + 64)`.
- S5.6 gives the full order of an issue.

## S5.5 Overlap between inputs and outputs

TAIDL assumes no hazard model; the ISA author declares how an instruction behaves when an input and an output of one buffer overlap.
Each same-buffer pair of an input region and an output region has exactly one relation, declared by a marker or given by the default.
The front end expands that relation into constraints of S5.3, through the mechanism that gives every region its implicit constraints.

| relation | meaning | marker on `write` |
| --- | --- | --- |
| tied | the output reuses the storage of the input | `inplace=x` |
| apart | no shared element | `apart=[x, ...]` |
| exact-or-apart | equal starts and equal sizes in every access dimension, or apart | `no_partial_overlap=[x, ...]`, or no marker |
| may overlap | no overlap restriction | `may_overlap=[x, ...]` |

- Each marker names read values, the results of `read`; it pairs that input region with the region this `write` writes.
- An undeclared same-buffer input-output pair is exact-or-apart.
- A pair in different buffers gets no relation and no constraint.
- Two output regions of one instruction stay apart; that is an implicit constraint of S5.3 (S1.3.4).
- Any further rule, such as two inputs that must differ, is a constraint the ISA author writes, such as `require(a1 != a2)`.
- A value under any permitted overlap is the semantics evaluated on the state before the issue (S5.6); no read and write order is assumed.

### Expansions

Every term is the evaluated start or size expression of a region, never an attribute value or name (S5.1).

- `apart(R, S)`: along some access dimension, `R.start + R.size <= S.start` or `S.start + S.size <= R.start`.
- `same(R, S)`: along every access dimension, equal starts and equal sizes; a dimension whose start and size are the same recorded expression is skipped.
- `inplace=x`: `same(In x, Out)`, and the tie is recorded for the backend.
- `apart=[x]`: `apart(In x, Out)`.
- `no_partial_overlap=[x]`: `same(In x, Out) or apart(In x, Out)`, the explicit spelling of the default.
- `may_overlap=[x]`: no constraint.
- The `Model` records, per same-buffer input-output pair, its relation and, for `inplace`, the tie; a backend reads that record.

### Rejected by `check`

The recorded expressions alone decide these cases, so `check` rejects them (S9).

- A marker naming a value that is not a `read` result of this instruction.
- An in-place marker that pairs an input and an output in different buffers.
- A placement marker on an input in another buffer than the output.
- An input named by more than one marker of one `write`, or more than one input tied to one output.
- An in-place input whose size expressions differ from the output's.
- `apart` on an input whose start and size expressions all equal the output's.
- An input whose start and size expressions all equal the output's, in the same buffer, without `inplace`.

### Violation

- An issue where any constraint fails, implicit, from a marker, or written, is the one fault of S1.3.5, before any write.
- A run-time start depends on attributes and state, so `check` never rejects a pair for an overlap it cannot decide from the recorded expressions.
- A front-end lint warns on an unmarked same-buffer pair of an instruction whose semantics is not elementwise; the semantics is unchanged.
