# Generator metadata

This page is S12 of the TAIDL language specification: data an ISA specification carries for one generator.
Generator metadata never changes what S5 to S8 mean; a datum that would is a layer (S11).

## S12.1 Metadata entries

- The accelerator, each buffer, and each instruction may carry generator metadata, as the `meta=` keyword of `Accelerator`, `acc.buffer`, and `@acc.instruction`.
- For example, `@acc.instruction(meta={'allo': {'engine': 'tensor'}})`.
- `d0` carries no metadata; a datum about hbm goes in the accelerator's entry.
- `meta` is a dict from a namespace, one generator's name, to that generator's metadata entry.
- A metadata entry is a `str`-keyed dict of plain data, possibly empty; another entry type is rejected, naming the node and the namespace.
- Plain data is exactly: `None`; `bool`; `int` within signed 64 bits; a finite `float`; `str`; a list of plain data; a dict with `str` keys and plain-data values.
- The type is tested exactly, so a tuple, a set, a subclass, or a self-containing list is rejected.
- A namespace is an identifier (S4.1).
- A generator's namespace is the `<name>` of its `generate_<name>`.
- The namespaces `oracle` and `backend` belong to the test oracle generator and the compiler backend generator; no other generator may use them.
- Run options, such as a cost model or an output directory, are keywords of a generator's entry point, not metadata.
- For example, an Allo partition choice on a buffer is the metadata entry under `allo`.

## S12.2 What the front end does

- The front end rejects only a namespace that is not an identifier and an entry that is not a `str`-keyed dict of plain data, naming the node and the namespace.
- It deep-copies each entry at its declaration call and never reads its contents; later changes to the author's object do not reach the `Model`.
- `check` publishes the metadata on each node of the `Model`; every read returns a fresh deep copy of dicts and lists (S4.6).
- `Model.metadata_namespaces` is the sorted tuple of the namespaces on any node; `python -m taidl check` prints it as a note (S9).
- The typed contract's `CONTRACT_VERSION` (S4.6) covers the container; each generator owns the schema of its namespace.

## S12.3 What a generator does

- A generator reads only its own namespace and ignores every other namespace silently.
- It rejects a key its schema does not hold, or a value of the wrong form, in its own namespace, with an error naming the node and the key, as one of its checks after `check` (S9).
- It reports the namespaces it reads, so a misspelled namespace shows in the front end's listing and in no generator's.
- Absent metadata means defaults.
- A datum a generator cannot default excludes that instruction or buffer, naming the node, the namespace, and the key.
- An accelerator-level datum it cannot default rejects with class `U`, naming the same; an empty usable set rejects (S9).
- Neutrality: for every ISA specification, each other generator's output is identical with one generator's metadata stripped.
- Output means the files under that generator's directory and its printed exclusion list, compared in its deterministic mode.
- A datum that two generators read is promoted to a layer with front-end validation.

## S12.4 Adding a generator

- A new generator takes the `Model` that `check` returns, or an `Accelerator` on which it calls `check` once (S9).
- It exposes `generate_<name>(source, /, base_dir=None, **options)`, as `generate_oracle` and `generate_backend` do, `source` a `Model` or an `Accelerator`; its body reads only the `Model`.
- It returns a frozen report, a `taidl` class, with `output_dir`, `excluded`, and `namespaces_read`.
- It writes only under `<base_dir>/accelerators/<accelerator>/<name>/` and the paths its report lists.
- It prints each exclusion in the two-line warning form; no registry lists the generator.
- Its own checks are plain functions it calls after `check`: no registry and no checker subclass.
- An act-shipped generator registers one scope letter before release; an outside generator uses scope `G`, its codes keyed by its namespace in its own `diagnostics.json`.
- An unknown key or a bad value in its namespace is class `U`, the rejection outcome.
- It rejects what the architect can correct and excludes, with a printed and recorded reason, an instruction it cannot model (S9).
- It ships an expectations table: one row per corpus case, with one status of S1.2: `supported`, `partial`, `excluded`, or `rejected` with a coded error.
- It touches neither `taidl` nor another generator.
- The generator guide in the `taidl` docs walks through these steps:
  1. Copy the generator template.
  2. Rename it to `generate_<name>`.
  3. Replace the emitter.
  4. Fill the expectations table.
  5. Run the corpus runner.
