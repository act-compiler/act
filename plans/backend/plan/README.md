# act-backend plan

Plan for the next two quarters, item ids B1 to B56, B1 to B46 sized in overhaul.md against the roadmap's three to four engineers.
Items B47 to B56 are the long-term platform, in roadmap waves 8 to 10 after M5.
B22 is merged into B21 and B23 is cut.

- [review.md](review.md): the backend as it stands, what to keep, verdict per component
- [bugs.md](bugs.md): bugs and defects D1 to D22 with severity and fix sketch
- [overhaul.md](overhaul.md): target structure per module and the ordered steps in two lanes
- [decisions.md](decisions.md): decisions in force with their evidence and the panel or owner decision behind each
- [architecture.md](architecture.md): B1 to B3, module layout, typed handoff, test gate
- [search.md](search.md): B29, B32, B34, deterministic search, rule guards, search budget
- [parser.md](parser.md): B4 and B5, the two parsers
- [isel.md](isel.md): B6 to B9, B31, B33, B35, typed payloads, analysis, rewrite values, rule overlap
- [extractor.md](extractor.md): B10, B11, B30, B36, B37, B39, B40, extraction, unwritten storage, pair relations in the allocator
- [generators.md](generators.md): B12 to B14 and B46, Python generators, templates, checks after `check`, exclusion report
- [coverage.md](coverage.md): B15 to B20, coverage gaps including state
- [defects.md](defects.md): B21 to B26, fixes for bugs D1 to D8
- [failures.md](failures.md): B41 to B43, dropped candidates, input validation including a kernel with no covering, input formats (bugs D17 to D22)
- [encoding.md](encoding.md): B38, B44, B45, width-layer domains, derived attributes, assembler unit (bug D15)
- [naming.md](naming.md): B27 and B28, terminology and renames
- [platform.md](platform.md): B47, B48, B50, B53 to B56, StableHLO importer, dialect, torch adapter, C++ HLO reader, one C++ driver
- [egraph.md](egraph.md): B49, B51, B52, the in-house C++ e-graph library, matcher, and the engine swap
