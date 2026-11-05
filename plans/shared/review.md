# What works across the repos and should stay

Paths are relative to /home/user.

## Front end

- The front end parses every text once, when the call that takes it runs, and keeps the tree beside the text (taidl/taidl/instruction.py:80, :98, :327-329); both generators consume trees, not strings, almost everywhere.
- Every fault names the instruction, the place, the line and the column (taidl/taidl/instruction.py:366-383), so a generator never has to explain a syntax error.
- One operation table drives the form checks of every word (taidl/taidl/operations.py:41-165), and the backend already reads it rather than copying it (act-backend/act_backend/isel_ir2isa_rewrites_rs_generator.py:253).
- Reference checks resolve `@a.`, `@c.`, `@s.` against declarations at parse time (taidl/taidl/instruction.py:340-358), so neither generator re-checks names.
- Placement marks are compiled into ordinary constraints with their trees (taidl/taidl/instruction.py:190-231), so generators see one constraint form.
- The data model rejects bad dims and types at the edge (taidl/taidl/accelerator.py:37-43).

## Generators

- Both generators refuse before writing anything, and remove the previous output first (act-backend/act_backend/generator.py:90-98, :124-131; act-oracle/taidl_to/generator.py:30-43).
- The oracle stages its output and moves it into place complete (act-oracle/taidl_to/generator.py:59-87).
- The backend's refusals are one module with one function per rule (act-backend/act_backend/spec_checks.py:22-418).
- The backend's strict template fill refuses a missing or unknown key (act-backend/act_backend/template_loader.py:69-81, :127-138).
- The oracle's shape rules are complete per word and run at generation and at each call (act-oracle/taidl_to/generic/shapes.py:450-543; act-oracle/taidl_to/generic/utils.py:771-786); they are the best seed for a shared checker.
- The oracle's literal check per element type is exact, including types with no infinity (act-oracle/taidl_to/generic/utils.py:380-436).
- The backend's checked Rust writer floors and refuses overflow as the spec says (act-backend/act_backend/expression_writer.py:79-123).

## Release and packaging

- Each generator imports only `taidl`, never the other generator (act/act/generators/__init__.py:4-15).
- All three consumers pin one front-end tag, v2.15 (act-backend/pyproject.toml:16; act-oracle/pyproject.toml:16; act/pyproject.toml:16), which matches taidl/taidl/__init__.py:3.
- The umbrella checks that every version it names agrees (act/tests/check_versions.py:31-60).
- The ANTLR tool and runtime are pinned together (taidl/taidl/antlr4/generate.sh:7-8; taidl/pyproject.toml:15).
- The four repos share one image tag and one runner pattern (taidl/docker/act.env:3; act/docker/act-env.env:2).
- An end-to-end test runs both generators on one ISA description (act/tests/test_end_to_end.py:69-78).
