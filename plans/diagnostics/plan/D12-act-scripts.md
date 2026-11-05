# D12 act umbrella: scripts and entry points

- What: one `fail()` shell helper (stderr, `error[XA...]`, exit 1) used by `act/docker/*.sh`, the tutorial scripts and `taidl/docker/*.sh` (E21).
- What: wrap the lazy generator imports in `act/act/generators/__init__.py:5-16` to raise `XA` with the install command (E22).
- What: name the file in `check_versions.py` failures (E20).
- Benefits: one spelling of "unknown argument", one stream, one exit status.
- Caveats: the tutorial scripts exist in three copies until A11 merges them.
- Fail points: a copy left behind; noticed by a grep test over `act/**/*.sh`.
- Order and dependencies: D1, A6, A15.
- Verification: each script run with a bad flag prints the common line on stderr and exits 1.
