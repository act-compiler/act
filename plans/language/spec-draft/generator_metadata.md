# Generator metadata

This page is S12 of the TAIDL language specification: data an ISA specification carries for one generator.
Generator metadata never changes what S5 to S8 mean; a datum that would is a layer (S11).

## S12.1 Metadata entries

- The accelerator, each buffer, and each instruction may carry generator metadata, as the `meta=` keyword of `Accelerator`, `acc.buffer`, and `@acc.instruction`.
- For example, `@acc.instruction(meta={'allo': {'engine': 'tensor'}})`.
- Generator metadata is a mapping from a namespace, one generator's name, to that generator's metadata entry.
- A metadata entry is plain data: `None`, `bool`, `int`, `float`, `str`, and lists and `str`-keyed dicts of these.
- A namespace is an identifier (S4.1).
- Run options, such as a cost model or an output directory, are keywords of a generator's entry point, not metadata.
- For example, an Allo partition choice on a buffer is the metadata entry under `allo`.

## S12.2 What the front end does

- The front end rejects only a namespace that is not an identifier and an entry that is not plain data, naming the node and the namespace.
- It stores each entry as given and never reads its contents.
- `check` publishes the metadata on each node of the `Model` and lists the namespaces present, as a note.
- The typed contract's `CONTRACT_VERSION` covers the container; each generator owns the schema of its namespace.

## S12.3 What a generator does

- A generator reads only its own namespace and ignores every other namespace silently.
- It rejects an unknown key in its own namespace with an error naming the node and the key, as one of its checks after `check` (S9).
- It reports the namespaces it reads, so a misspelled namespace shows in the front end's listing and in no generator's.
- Neutrality: for every ISA specification, each other generator's output is identical with one generator's metadata stripped.
- A datum that two generators read is promoted to a layer with front-end validation.

## S12.4 Adding a generator

- A new generator takes the `Model` that `check` returns, or an `Accelerator` on which it calls `check` once (S9).
- It exposes `generate_<name>(model, base_dir=None, **options)`, as `generate_oracle` and `generate_backend` do; its body reads only the `Model`.
- Its own checks are plain functions it calls after `check`: no registry and no checker subclass.
- It rejects what the architect can correct and excludes, with a printed and recorded reason, an instruction it cannot model (S9).
- It ships an expectations table: one row per corpus case, supported, excluded, or rejected with a coded error (S1.2).
- It touches neither `taidl` nor another generator.
- The guide that walks through these steps is plan item L41.
