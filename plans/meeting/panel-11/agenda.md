# Panel 11 agenda: authoring surface and the check function

Binding owner decisions: 10 (breaking changes in one v3 release), 16, 28 (dependencies), 30, 34, 38, 39, 41, and 42.
Owner decision 41: ISA specifications move off semantics written as text inside Python strings to a clean embedded language (for example a Python meta-DSL, or C++ or MLIR); nothing fancy, as few dependencies as possible, and comfortable for an architect.
Owner decision 42: a check function validates an ISA specification on its own; it always runs once before any generator, followed by that generator's own additional checks (for example the backend's machine-state requirements; none for the oracle).
Today: /home/user/taidl/examples/mini_trn.py writes semantics as HLO-like text in `add_semantics("""...""")`, regions as nested string lists such as `[['d0', ['@a.addr_in'], ['@c.n * 128']]]`, and the text is parsed by ANTLR grammars in /home/user/taidl/taidl/antlr4/.

### Q11.1: What is the authoring surface for an ISA specification in v3?

- Options: A. a Python embedded DSL (decorated functions whose bodies build operations through ordinary Python expressions, as NKI, Allo, Exo, or JAX tracing do); B. a standalone textual language with its own small parser; C. an MLIR dialect with TableGen-defined operations; D. a C++ embedded DSL; E. other.
- Criteria: readability for an architect, unambiguous meaning, dependencies, error messages that point at the author's line, tooling (editor support, debugging), and how generators consume it (the typed contract of panel 3).
- Evidence: /home/user/taidl/examples/, /home/user/taidl/taidl/*.py, spec-draft, /home/user/act/plans/priorart/ (exo.md, nki.md, mlir-dialects.md, sail.md), case studies; external docs of NKI, Allo, Exo, Triton, JAX, MLIR ODS (load WebSearch and WebFetch via ToolSearch).

### Q11.2: How do regions, attributes, state, constraints, markers, and the optional layers look in that surface?

- Context: today regions and attribute references are strings; owner decision 34 makes attributes plain variables whose names carry no meaning; decisions 30 and 38 add markers, implicit constraints, and optional width and encoding layers; decision 39 adds generator metadata.
- Options: argue a concrete sketch of one instruction from examples/mini_trn.py and one accumulate instruction in the proposed surface.

### Q11.3: What does the check function check, and how do generator-specific checks attach?

- Context: panel 3 decided `validate()` publishes derived facts on a typed model; decision 42 requires one check before any generator plus each generator's own checks.
- Options: A. `check(spec)` in taidl returns the validated typed model, and each generator registers an extra check that runs on it; B. each generator subclasses one checker; C. other.
- Questions within: what the backend's extra checks are (machine state, supported operations); what the oracle's are (expected none); how a check reports errors (decision 40: only problems the architect can correct, with a clear error at the author's line).

### Q11.4: How do existing ISA specifications migrate, and what is the order of work?

- Options: A. a one-time converter from the string form to the new surface, run on examples and tests, then the string form is removed in v3; B. both forms accepted in v3, string form removed later; C. manual migration only.
- Constraint: decision 10 (one coordinated breaking release).
