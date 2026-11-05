# D13 Machine-readable output and code explanations

- What: `--diagnostics=json` on the backend binary and an environment variable for Python tools, printing one JSON object per diagnostic.
- What: `explain(code)` in `taidl` and `--explain CODE` in the binary print the rule, the spec section and one example from the registry.
- Benefits: editors and CI read codes without parsing text.
- Caveats: optional; skip if the schedule slips.
- Fail points: JSON schema drift; noticed by a schema test against the registry fields.
- Order and dependencies: D1, D7, D8.
- Verification: a test parses the JSON output of one error per class.
