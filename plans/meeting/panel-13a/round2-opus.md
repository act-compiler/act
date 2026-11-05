# Panel 13a round 2: Opus (a precise, checkable contract)

Plan paths are relative to /home/user/act/plans, spec pages sit under language/spec-draft/, and code paths are relative to /home/user.
Binding: owner decisions (OD), panels 3, 10, 11, and 12; "run in this round" marks a Python 3.13 experiment the moderator can repeat.

### Q13a.1 (SF4): NumPy integers
- Opposing: none on the rule; Sonnet keeps `bool` legal as an `ops.constant` value at `pred`, which I concede, since `pred` is no integer position.
- Final: any non-`bool` object `operator.index` accepts, stored as exact `int` (docs.python.org/3/library/operator.html); confidence medium-high; divides: nothing.

### Q13a.2 (SF8): handle or function; bare form
- Opposing (Fable): a handle with no `__call__` makes a call Python's own `TypeError` in the author's frame, with no extra rule.
- Reply: a `__call__` raising `taidl.TraceError` keeps that, since `TraceError` subclasses `TypeError` (grammar.md:87), and adds the `R` code and `[S3.1]`.
- Final: a handle `acc.encode` matches by identity; a call raises `TraceError` naming S3.1; bare form as grammar.md:52; confidence high; divides: the code on the call error.

### Q13a.3 (SF11): reuse of all-plain regions
- Opposing: none; the one shared risk, an S5.5 rule keyed on region identity, does not exist, since S5.5 compares expressions (conformance.md:56-57) and each call records its own node (object_model.md:27).
- Final: as round 1; confidence high; divides: nothing.

### Q13a.4 (SF12): tracing after a collected problem
- Opposing (Sonnet): "not reported when its only cause is a placeholder" needs a cause per S8 row, so front ends differ on `ops.add(bad, x)`; ending the body at the first problem is derivable.
- Reply: concede that a per-check skip needs such a table, and replace it with a per-call rule: a call with a placeholder argument checks nothing, records nothing, and returns a placeholder of its result kind.
- Each call has one result kind (`read` and `ops` give a tensor value, `buffer.region` a region), so `ops.add(bad, x)` reports nothing in every front end.
- Against ending the body: panel 11 binds "it collects every problem" (panel-11/decision.md:28), and ending a body raises through author frames, where a `try/except Exception` catches it.
- Final: a signature problem is reported once and the body not run; any other problem is recorded and returns a placeholder; `bool()` on one raises at once (grammar.md:76); confidence medium-high; divides: Sonnet's first-problem rule.

### Q13a.5 (SF16): reading of OD21
- Opposing: Sonnet asks the owner to confirm; but all three readings keep `@s.` in every printed form and change no owner text, so no escalation criterion applies (roles.md, Moderator).
- Final: as round 1; confidence high; divides: only whether the reading is forwarded.

### Q13a.6 (SF21): an always-faulting case rejects
- Opposing (Sonnet): a plain start below 0 also faults at every issue; concede, since every evaluated start is at least 0 (instructions.md:55, S1.3.3), a bound that covers `d0` too.
- The equal-output row is exact: equal sizes either reach 1 and overlap (S1.3.4) or fall below 1 (S1.3.3).
- Final: reject a plain size below 1, a plain start below 0, a plain start plus size past a declared dimension (never `d0`), and equal output regions of one buffer; confidence high.

### Q13a.7 (SF23): plain `True` and `False` in `require`
- Opposing (Fable, Sonnet): a plain `True` is dropped, not recorded; concede, since conditions have no boolean leaf (object_model.md:37) and a recorded `True` adds one every generator must read.
- Final: `all_of` drops `True` and is `False` with any `False`; `any_of` drops `False` and holds with any `True`; a `require` argument folding to `False` rejects (S6.3); empty `any_of()` and `all_of()` reject.
- instructions.md:63 ("a check on plain values only is a Python `assert`") needs the matching edit; confidence medium-high; divides: nothing.

### Q13a.8 (SF26): entry order and the warning channel
- Opposing (Sonnet): warnings vanish under `-W ignore`, and `Model.notes` gives a generator author data without parsing stderr.
- Reply: the lint is optional and "not part of conformance" (instructions.md:128), so `Model.notes` makes two conforming front ends return unequal `Model` objects (object_model.md:52).
- A runner gets structured records through `warnings.catch_warnings(record=True)`, each with category, file, and line (docs.python.org/3/library/warnings.html); I concede Sonnet's `warn_explicit` at the node's call site.
- Final: entries by declaration, then call order; the lint is a `taidl.TaidlWarning` emitted only after `check` finds no problem, so `-W error` never hides a `Rejection`; the namespace note is a print (conformance.md:25); confidence medium-high; divides: `Model.notes`.

### Q13a.9 (SF28): the check command and two accelerators
- Opposing (Fable, Sonnet): check every `Accelerator` built, in construction order, under the file's stem; a globals scan misses one held in a list or built by a factory.
- Reply on scope: concede; `variants = [build(n) for n in (4, 8)]` binds no `Accelerator` in globals, and binding order mishandles rebinding.
- The command installs a collector for its run only, a context variable `Accelerator.__init__` appends to, so `taidl.check` from Python keeps no registry.
- Reply on the name: hold `'<run_path>'`; with `run_name='json'` an `import json` inside the file returned the file itself (run in this round), since `run_path` places it at `sys.modules[run_name]` (CPython Lib/runpy.py).
- Reply on codes: one per phase, since `check` runs no author code (bodies run at decoration, grammar.md:51):
  - 0: every check returned;
  - 1: a `check` raised `Rejection`, after every accelerator was checked;
  - 2: running the file raised, a `TraceError` included, its traceback printed as is;
  - 3: a `check` raised anything else, an internal error;
  - 4: a usage error, such as bad arguments or an unreadable file;
  - 5: the file built no `Accelerator`.
- Codes 3, 4, and 5 keep pytest's internal error, usage error, and "No tests were collected" (docs.pytest.org/en/stable/reference/exit-codes.html).
- Confidence: medium-high on scope and name, medium on codes; divides: the stem, and Sonnet's one code for every spec problem.

### Q13a.10 (SF32): a kernel that needs an excluded instruction
- Opposing (Sonnet): "excluded instructions whose operators all occur in the kernel" needs only the list and the kernel, is deterministic, and stays outside the match key, which I concede (panel-3/decision.md:135).
- Hold the uncovered operation: the extractor keeps a subtree only when every child has an extraction (act-backend/act_backend/generic/src/isel/extractor/slow.rs:116), so one bottom-up pass finds the lowest uncovered one.
- "All operators occur" admits most excluded instructions on a large kernel; the message also states the stop cause, saturation or the limit, as pipeline.rs:205-219 does, since only saturation shows no covering exists.
- Final: code and kernel object; the message names the lowest uncovered operation and the stop cause, then the excluded instructions whose operators include it; confidence medium; divides: the hint set.

### Q13a.11 (SF38): metadata frozen, tuples visible
- Opposing (Sonnet): a proxy breaks what an emitter first does with an entry; concede, since `json.dumps`, `copy.deepcopy`, and `copy.copy` each raise `TypeError` on a `MappingProxyType` (run in this round).
- Tuples are worse: `(1,) == [1]` is `False`, a silent wrong branch, and S12.1 rejects a tuple as plain data (generator_metadata.md:12-13).
- Keep Fable's timing: the copy is taken at the declaration call (generator_metadata.md:23), so a family loop mutating one dict publishes each iteration's value.
- Final: each read returns a new deep copy in S12.1 types, checkable as `n.meta == n.meta` and `n.meta is not n.meta`; U2.md:12 then says "a fresh plain-data copy per read"; confidence high; divides: Fable's view.

### Q13a.12 (LY-03): which link spellings count
- Opposing (Fable): S6.1 compares trees structurally and panel 3 refused a normalizer (panel-3/decision.md:92-95), so a link is an enumerated list and `(X+1)*8` is not one.
- Reply: LY-03's own list leaves `4 + 8*D` and `4 - D` undecided and rejects the affine `-(8*D)`; a fold has no shape cases for two front ends to split on.
- Panel 10 refuses "a nonaffine link" (panel-10/decision.md:20), so its test is affinity, and its "integer literals" (:19) can only mean plain `int`, since the tracer sees values, not text (grammar.md:40).
- The fold compares no two trees, and the tree stays the constraint evaluated as recorded (expressions.md:26); (k, c) drives only encode and decode; I concede Sonnet's "one side is exactly one attribute".
- Final: an `==` under `require`, one side one attribute X, the other folding in exact integers over `+`, `-`, unary `-`, and plain `*` to k*Y + c, k nonzero, k and c in signed 64 bits; confidence medium-high; divides: fold or list.

### Q13a.13 (LY-05): bare `Comp` against `Comp[s64]`
- Opposing (Fable): bare `Comp` is `Comp[s64]` in every respect; TableGen's `bits<n>` knows no declared-versus-default split.
- Reply: panel 10 rules a "declared type" (panel-10/decision.md:17) because "a default 64-bit attribute names no rule for which bits enter a field" (:61), so it binds the distinction.
- TableGen agrees: `bits<n>` takes n in every declaration (llvm.org/docs/TableGen/ProgRef.html), so every encoded width is written; the draft already records the flag (object_model.md:19) and ties `type` constraints to it (:30).
- Final: as round 1; confidence medium-high; divides: Fable's reading of panel 10.

### Q13a.14 (LY-09): one word width
- Opposing: none; final as round 1; confidence high.

### Q13a.15 (LY-10): what decode checks
- Opposing (Fable): decode reads fixed bits, fields, and links only, since a field of width w holds exactly its type.
- Reply: true for encoded attributes, but a derived attribute "may carry its own type" (layers.md:71), and a link can compute a value outside it.
- Decode as the exact inverse of encode is checkable by hypothesis (OD28): `decode(encode(x)) == x` for typed x and `encode(decode(w)) == w` for decodable w; the second fails without the derived type check.
- Final: fixed bits, fields, exact link inversion, and each derived value's type (S1.3.10), no written constraint, lowest failing index; confidence high; divides: the derived type check.

### Q13a.16 (LY-11): byte order
- Opposing (Sonnet): a default left to each generator lets two assemblers emit incompatible files; Fable adds that RISC-V fixes parcels little-endian whatever the data endianness (lkml.iu.edu/2202.3/03720.html).
- Reply: concede; a normative default gives each file one meaning across conforming generators, the property round 1 sought by a required keyword, at no cost per call.
- Precision: "the high bits of the last byte zero" names the wrong byte under a big-endian override, so state the padding per word.
- Final: words indexed from 0, hex in messages; each word is `ceil(word/8)` bytes, little-endian by default, top `8*ceil(word/8) - word` bits zero; `byte_order='big'` overrides; confidence medium-high; divides: the default's home.

### Q13a.17 (LY-16): who checks fixed-bit disjointness
- Opposing: none; panel 10's "at generation time" (panel-10/decision.md:17) yields to panel 11's `check` running every front-end rule (panel-11/decision.md:28).
- Final: as round 1, with conformance.md:110 moved into the Layers table; confidence high.

### Q13a.18 (LY-21): frozen collections
- Opposing: none on node collections; every collection the front end builds (attributes, regions, constraints, fixed bits, slices) is a tuple.
- Final: tuples for those, metadata as Q13a.11; confidence high; divides: as Q13a.11.

### Q13a.19 (LY-26): an entry is a dict
- Opposing: none; final as round 1; confidence high.

### Q13a.20 (LY-28): metadata for `d0`
- Opposing: none; final as round 1; confidence medium-high.

### Q13a.21 (LY-31): may a generator require metadata
- Opposing (Fable, Sonnet): OD39 makes metadata optional, so absence means defaults, and a datum with no default excludes the instruction.
- Reply: concede; round 1 read "optional" as optional to the language only, but OD39 names no exception, and it outranks the Q11.3 split round 1 leaned on.
- Final: exclusion naming node, namespace, and key; when every instruction is excluded, conformance.md:111 rejects; neutrality as LY-31 in deterministic mode (OD27); confidence medium-high.

### Q13a.22 (LY-32): entry point, output place, return value
- Opposing (Fable, Sonnet): returning `None` leaves the T12 runner to re-parse printed text or the crate to assert exclusions by `code` and `object`.
- Reply: concede, the change round 1 named; S12.3 also has a generator report the namespaces it reads (generator_metadata.md:32), and only a return value gives both as data.
- Final: `generate_<name>(source, /, base_dir=None, **options)` returns a frozen `taidl` report with `output_dir`, `excluded` (fields of a `Rejection` entry, conformance.md:19), and `namespaces_read`; output under `<base_dir>/targets/<accelerator>/<name>/`.
- Exclusions print in the two-line warning form, not through `warnings`, since `-W error` would abort a half-written output; confidence high on the report, medium on `/`; divides: `/`.

### Q13a.23 (LY-34): scope letters
- Opposing (Sonnet): a letter in a shared list is a change per generator (generator_metadata.md:43), so outside generators share scope `G`; an unknown key is class `K`.
- Reply: concede `G`; a code is class, scope, and three digits (diagnostics/scheme.md:9), so round 1's "no letter" was no code, and each generator keeps its own `diagnostics.json` (:131).
- Hold `U`: `K` is run input such as kernels and binaries (:16), while a bad key sits in the ISA specification, the `U` rejection outcome (:20-23).
- Final: act-shipped generators register a letter (:26), any other uses `G`, and a bad key is `U` with the rejection outcome; confidence medium-high; divides: the class.

### Q13a.24 (LY-35): contract version mismatch
- Opposing (Sonnet): a generator states its major only, and any minor proceeds.
- Reply: a generator using a member added in 3.2 meets a 3.1 `Model` in a development install and fails later with `AttributeError`, not the `X` error naming both; but U2.md:6 keeps the version on an addition, so an addition must bump the minor.
- Final: as round 1, with that U2 edit; confidence high; divides: the lower bound.

## Escalation to the owner

- None.
