# Generator metadata

This page is S12 of the spec: data an ISA specification carries for one generator.
Generator metadata never changes what S5 to S8 mean; a datum that would is a layer (S11).

## S12.1 Metadata entries

- The accelerator, each buffer, and each instruction may carry generator metadata.
- Generator metadata is a mapping from a namespace, one generator's name, to that generator's metadata entry.
- A metadata entry is plain data: `None`, `bool`, `int`, `float`, `str`, and lists and `str`-keyed dicts of these.
- A namespace is an identifier (S4.1).
- Run options, such as a cost model or an output directory, are keywords of a generator's entry point, not metadata.
- For example, an Allo partition hint on a buffer is the metadata entry under `allo`.

## S12.2 What the front end does

- The front end rejects only a namespace that is not an identifier and an entry that is not plain data, naming the node and the namespace.
- It stores each entry as given and never reads its contents.
- `validate()` publishes the metadata on each typed node and lists the namespaces present, as a note.
- The typed contract's `CONTRACT_VERSION` covers the container; each generator owns the schema of its namespace.

## S12.3 What a generator does

- A generator reads only its own namespace and ignores every other namespace silently.
- It rejects an unknown key in its own namespace with an error naming the node and the key.
- It reports the namespaces it reads, so a misspelled namespace shows in the front end's listing and in no generator's.
- Neutrality: for every ISA specification, each other generator's output is identical with one generator's metadata stripped.
- A datum that two generators read is promoted to a layer with front-end validation.

## S12.4 Adding a generator

- A new generator takes the typed model after `validate()`.
- It exposes `generate_<name>(accelerator, base_dir=None, **options)`, as `generate_oracle` and `generate_backend` do.
- It ships an expectations table: one row per corpus case, supported or unsupported with a coded error (S1.2).
- It touches neither `taidl` nor another generator.
- The guide that walks through these steps is plan item L41.
