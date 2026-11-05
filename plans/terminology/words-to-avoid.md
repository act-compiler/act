# Words to avoid

Each word was found in the repos unless the second column says otherwise.

| Word | Seen in | Replacement |
| --- | --- | --- |
| refuse, refuses, refused | `act-oracle/README.md` (section titles "What an oracle refuses", "What a kernel refuses", "What a call refuses"), `act-oracle/docs/operations.md`, `act-backend/README.md:48`, `taidl/docs/{semantics,expressions,declarations,operations}.md`, `taidl/taidl/instruction.py:116`, `taidl/taidl/names.py` | reject, rejects, rejected |
| machine code | `act-backend/README.md:16,19` | assembly code |
| tile (for a region or its value) | `act-oracle/README.md:67,122`, `act-oracle/docs/operations.md` | region, region value |
| window | `act-oracle/README.md:122,129` | region |
| dtype (outside NumPy or PyTorch names) | `act-oracle/README.md:96,102,105` | element type |
| seamless support | `act-backend/README.md:17` | the concrete claim, such as "plugs into XLA" |
| leverages | `act/README.md:17` | uses |
| just (as in "just from TAIDL specifications") | `act/README.md:18,20`, `act-oracle/README.md:3` | "from an ISA description alone", or delete |
| TAIDL specification, ISA specifications | `act/README.md:12,18,20`, `act-oracle/README.md:3,15` | ISA description |
| honored, respected | not found in the repos | applies, holds, or the concrete check |
| simulator (unqualified) | general | test oracle, or functional simulator on first use |
