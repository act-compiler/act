# Panel 13b decision: semantics, instructions, and execution

The questions are the debate rows of plans/reviews/spec-gaps/instructions.md (G) and semantics.md (SG), instructions first.
Binding: all owner decisions (OD), and panels 1, 2, 9, joint-1-5, and joint-unwritten.
Spec paths are under plans/language/spec-draft.

| Question | Decision | Confidence |
| --- | --- | --- |
| Q13b.1 (G3) | `d0` carries no generator metadata; an hbm entry goes in the accelerator's `meta` | high |
| Q13b.2 (G7) | The decorator returns an instruction handle; `acc.encode` takes it; calling it raises | medium |
| Q13b.3 (G12) | Five S9 rejections of text that faults at every issue, as a closed list | medium-high |
| Q13b.4 (G21) | Keep D5: an address reaches state only through a computational attribute | medium-high |
| Q13b.5 (G23) | A run yields the final `hbm` contents or the fault report | high |
| Q13b.6 (G26) | Rejected; see Q13b.15 | medium-high |
| Q13b.7 (SG5) | A plain integer is a non-`bool` value that `operator.index` accepts, held as `int` | medium |
| Q13b.8 (SG6) | A plain `bool` is a condition leaf that never faults; an always-false `require` is rejected; empty calls raise | medium-high |
| Q13b.9 (SG7) | A `float` is its binary64 value, an `int` or `Fraction` its exact value, each rounded once; no `str` | medium-high |
| Q13b.10 (SG9) | The signature table, with `ops.iota(element_type, shape, iota_dimension=...)` and no default | medium-high |
| Q13b.11 (SG11) | Every attribute that names an axis is a plain `int` | high |
| Q13b.12 (SG15) | Option B: `ops.dot(..., element_type=None)`, with its meaning stated | medium-high |
| Q13b.13 (SG22) | Name `pow`, `rSqrt`, `rootn(x, 3)`, and the limits; key both clauses to the correctly rounded value | medium-high |
| Q13b.14 (SG26) | Nearest by value, a tie up; `f4e2m1fn` NaN gives +0 as TAIDL's own value; L17 tests both rows | medium-high |
| Q13b.15 (SG31) | A `read` with no `write` is rejected (S9), by panel 2 Q2.9 R11 | medium-high |

## Rule set

- R1, S4.3: "`d0` carries no generator metadata of its own; an entry about hbm goes in the accelerator's `meta`."
- R2, S5.1: "The decorator returns the instruction's handle, whose one attribute is `name`; `acc.encode` takes the handle, and calling it raises at the author's line."
- R3, S5.5 "Rejected by `check`" and S9, with no promise of completeness:
  - a plain start below 0 or a plain size below 1;
  - a plain start plus a plain size past a plain access dimension, `d0` excepted;
  - two output regions of one buffer whose start expressions all equal;
  - an exact-or-apart pair whose start expressions all equal and whose sizes differ as two plain integers in some dimension;
  - an input tied to more than one output.
- R4, S5.4: "An address held in state is loaded from a computational attribute; an addressing attribute never reaches state (D5)."
- R5, S5.6: "A run yields the `hbm` contents after the last issue, or the fault report; the final state and on-chip buffers are observable only through a generator's own interface."
- R6, S5.2, S7.1, S9: "An instruction with a `read` and no `write` is rejected (S9); an instruction that touches memory without an architectural effect, such as a prefetch, has no `read` and no `write`."
- R7, S6.2, S4.3, S4.4: "A plain integer is a value other than a `bool` that `operator.index` accepts; its value is the `int` that call returns, and the `Model` holds that `int`."
- R7 also, S7.5: "`True` and `False` are values only at `pred`."
- R8, S6.3: "A plain `bool` in `require`, `any_of`, or `all_of` is a condition with that value; it never faults, and every other comparison still evaluates."
- R8 also: "A `require` that is false whatever its symbolic comparisons give, such as `require(False)` or `require(all_of(False, c))`, is rejected (S9); `require()`, `any_of()`, and `all_of()` raise at the author's line."
- R9, S7.5: "A plain number is its exact value rounded once to nearest, ties to even: a `float`'s exact value is its binary64 value, and an `int` or a `fractions.Fraction` is exact."
- R10, S7.5 table: `ops.constant(value, element_type=s64, shape=[])`, `ops.iota(element_type, shape, iota_dimension=d)`, `ops.convert(x, element_type)`, `ops.bitcast_convert(x, element_type)`, `ops.reshape(x, shape)`, `ops.broadcast(x, shape, dimensions=...)`, `ops.dot(..., element_type=None)`.
- R10 also: "`iota_dimension` has no default; a target may also be passed by its keyword."
- R11, S8.2, S8.4: "An attribute that names an axis is a plain `int`: `dimensions`, `iota_dimension`, `index_vector_dim`, the `dot` and `gather` dimension lists, the `transpose` `permutation`, and the `concatenate` and `reverse` dimensions; a size, a bound, a padding, or a bit count may be an integer expression."
- R12, S8.4: "`ops.dot` takes `element_type=T`, by default its operands' type; the exact products are summed exactly, an integer T wraps the sum to T, a float T takes the joint 1-5 bound with u of T and n = K, and the operands and T are all integer or all float."
- R12 also: the running example becomes `ops.dot(a, b, ..., element_type=f32)` (instructions.md:23).
- R13, operation_meaning.md:69 and :75: "The IEEE result is the correctly rounded value of y at t, with `pow` for `power`, `rSqrt` for `rsqrt`, `rootn(x, 3)` for `cbrt`, and the mathematical value for `erf` and `logistic`."
- R13 also: "When the IEEE result is an infinity, a zero, or NaN, r equals it; otherwise r meets the bound, and within it r may be 0 or a subnormal"; the `f4e2m1fn` clause of :69 stays.
- R14, operation_meaning.md:53: "the power of two nearest by value, so 1.5 * 2^k rounds up"; :52 keeps "NaN gives +0", which the oracle lowers as an explicit `select`.
- R14 also: L17 tests every `f8e8m0fnu` and `f4e2m1fn` row against the pinned stack, the sign of zero included.

## Q13b.1 (G3): generator metadata for hbm

- OD39 places metadata at three levels, `Accelerator(..., meta=None)` carries the accelerator level (declarations.md:19, :22), and hbm is one per accelerator, so `d0_meta=` would be a second spelling with one use site.
- Dissent: none.

## Q13b.2 (G7): what the decorator returns

- Opus's decisive point: the S3.2 row raises only for `read`, `write`, `require`, or `update` (grammar.md:84), and a body that calls none is legal (semantics.md:13), so a called unchanged function returns silently.
- Only a handle's `__call__` gives every call one outcome (OD16); the function route needs two `check` rows and still leaves that case open.
- The handle has `name` and no methods, so it stays within OD41.
- Dissent: Fable holds the unchanged function, keyed by object identity, with a `check` rejection of a function decorated twice.

## Q13b.3 (G12): text-decidable rejections

- Panel 9 step 9 keeps static rejection "only where text decides" (panel-9/decision.md:93); each row of R3 is decided by the recorded trees and faults at every issue, so it is within that rule and OD40.
- Row 3 needs no size equality: sizes are at least 1, so equal starts in every dimension share the start element (all three in round 2).
- Row 4 is narrowed: size trees such as `n` and `m`, or `2 * m` and `m + m`, may evaluate equal, and texts that evaluate equal satisfy `same` (panel-9/decision.md:67).
- Row 5: a tie forces each output onto the input's storage (OD17), so two tied outputs overlap at every issue.
- Corpus: panel 9's F5 row (panel-9/decision.md:119) keeps a symbolic size and expects a fault; a plain-size twin expects rejection; each new row has one rejection case.
- Dissent: none.

## Q13b.4 (G21): addressing attributes into state

- The backend refuses every instruction that updates state (act-backend spec_checks.py:370-378, read), so an `Addr` spelling buys no v3 feature.
- State may appear in an update (expressions.md:12), so marking addressing state is a fixpoint over every update, and one new set-base instruction can turn an accepted size elsewhere into an error.
- Rejecting now and allowing later changes no accepted specification; the reverse order breaks them.
- Fable's rule, as that fixpoint, goes to the backend plan's B16 row as the relaxation to weigh with state support; normative text carries no plan status (SG13).
- Dissent: none in final positions.

## Q13b.5 (G23): what a run yields

- The oracle returns only the declared outputs of `executable.execute` (act-oracle decorator.py:190-194, read), and the backend closed loop compares hbm only.
- A faulting run is compared by its fault report; Q1.12's "every earlier issue has committed" stays the meaning of the run, and R5 adds no duty to expose that hbm.
- Dissent: none.

## Q13b.6 and Q13b.15 (G26, SG31): a `read` with no `write`

- Panel 2 Q2.9 binds: "an instruction may omit semantics only when it has no regions; a semantics defines at least `%Out0`" (panel-2/decision.md:125).
- A `read` with no `write` has a region and defines no output, so the binding text rejects it under either reading of "omit".
- Panel 11 reasoning 10 removes the type reconciliation of the string surface (panel-11/decision.md:92); it repeals no R11 clause, and panel 11 is not binding here.
- The cited prefetches have no architectural effect ("merely a hint", Intel `PREFETCHh`), so they are written with no regions, as Opus and Fable conceded.
- Fable's other classes (a send out of SBUF, a copy into another CTA) carry no citation and take no weight.
- Dissent: Fable, and Sonnet in round 2, hold that R11 lapsed with the string surface; recorded for the joint moderator below.

## Q13b.7 (SG5): `bool` and NumPy integers

- Probed (NumPy 2.5.3): `operator.index(np.int64(4))` returns an `int`, `operator.index(np.bool_(True))` raises `TypeError`, and `operator.index(True)` returns 1, so excluding Python `bool` by type excludes both bools.
- Probed: `np.int64(4) + S()` reaches `S.__radd__` with an `int`, while `size=[np.int64(4)]` reaches the front end as `np.int64`; only `operator.index` accepts the same value in both places.
- Table-driven families pass NumPy scalars, and rejecting them fails without a genuine cause (OD40); the tree still holds exact `int`.
- Dissent: Opus and Sonnet named exact `int` as acceptable; no evidence favours it beyond one shorter sentence.

## Q13b.8 (SG6): plain `bool` and empty conditions

- Rejecting a plain `True` sends the author to `assert`, which Python drops under `-O`, and fails a helper such as `def fits(s): return s <= 64` only when its argument is plain (OD40).
- Sonnet's S6.3 objection is met by the leaf rule: every other comparison still evaluates, so `any_of(True, a // b == 1)` faults where `b` is 0 (expressions.md:36).
- An always-false `require` faults at every issue, the R3 class, so `check` rejects it.
- G19's `assert` note is replaced by R8.
- Dissent: Fable would still reject a plain `bool` inside `any_of`, which the leaf rule makes unnecessary.

## Q13b.9 (SG7): plain float constants

- The front end never reads source text (grammar.md:4), so semantics.md:78's "exact decimal value" cannot hold for a `float`.
- Panel 1 Q1.10 binds single rounding and names `Fraction` or hex bits as the single-rounding spellings (panel-1/decision.md:144-146); R9 applies that to the value TAIDL receives.
- An `int` must not pass through `float()`, as the oracle's `mlir_literal` does today (act-oracle utils.py:408, read), so the oracle needs an exact rounder, which also serves `Fraction`.
- L17 tests `1 + 2**-8 + 2**-40` at `bf16` and `2**53 + 2**29 + 1` at `f32`.
- Dissent: none in final positions; all three drop `str`.

## Q13b.10 (SG9): signatures and `iota` order

- `ops.constant` (semantics.md:74) and `jax.lax.broadcasted_iota(dtype, shape, dimension)` put the type first, and StableHLO makes `iota_dimension` required, so it has no default.
- Dissent: none.

## Q13b.11 (SG11): symbolic axes

- S7.4 infers every result shape, and S6 has no selection over integer expressions, so a symbolic axis leaves the result's dimension order unstateable.
- The oracle test with `lhs_contracting_dims={`@c.k`}` (act-oracle tests/test_templates.py:292, read) breaks, which OD10 allows.
- Dissent: none.

## Q13b.12 (SG15): result type of `dot`

- Panel 1 Q1.5 step 4 rests on "the author writes a wider result type, which `sh_dot` supports since operand and result types are independent" (panel-1/decision.md:87), and joint 1-5 step 8 repeats it (joint-1-5/decision.md:63).
- Option A would remove that premise; B keeps the v2 capability, so it is no new construct under Q2.3, and no accumulator declaration (joint-1-5/decision.md:80).
- Opus's meaning sentence closes the integer case, and makes `dot(a, b, element_type=T)` equal `dot(convert(a, T), convert(b, T))` wherever both converts are exact.
- Dissent: none in final positions; no backend match of either form was run.

## Q13b.13 (SG22): which IEEE function

- `pow(1, inf)` is 1 and `powr(1, inf)` is NaN, so the function must be named, and StableHLO names IEEE `pow`.
- SG22's "the bound applies when y is finite and nonzero" is unsatisfiable at overflow: `exponential(88.8)` at `f32` has a finite y above the largest `f32`, whose IEEE result is +inf.
- Keying both clauses to the correctly rounded value partitions the inputs and covers `erf` and `logistic`, which have no IEEE function.
- Dissent: none in final positions.

## Q13b.14 (SG26): `f8e8m0fnu` rounding and `f4e2m1fn` NaN

- Run here, ml_dtypes 0.6.0: 1.5, 3, and 6 give 2, 4, and 8 at `f8e8m0fnu`, and a positive NaN gives byte 0x8, which is -0, at `f4e2m1fn`.
- MX leaves the NaN result to the implementation, so OD16 needs TAIDL's own value, and +0 is the simple one.
- Dissent: none in final positions.

## Records for the joint moderator

- Q13b.15 applies panel 2 Q2.9 R11 as written; Fable's amendment, a legal `read` with no `write` in the traced surface, goes to a joint moderator of panels 2 and 13b, recommended: keep R11, since no cited in-scope instruction needs it.
- Q13b.9 reads Q1.10's "exact decimal" as the exact value TAIDL receives; a joint moderator of panels 1 and 13b confirms that reading.
- Q13b.8 and panel 13a's SF23 are one question; a different panel 13a ruling goes to a joint moderator.

## Escalation to the owner

- None.
