# Grammar

This page is S2 and S3 of the spec.
The ANTLR4 grammars are normative and are not copied here: `taidl/antlr4/IDLV2.g4` (semantics) and `taidl/antlr4/IDLV2Expr.g4` (expressions), in the `taidl` repo at the release this spec describes.
This page states what the rules do not show at a glance: token precedence, entry rules, and operator binding.

## S2 Lexical structure

### S2.1 Semantics tokens

- Whitespace (space, tab, CR, LF) separates tokens and is otherwise skipped.
- A comment runs from `//` to the end of its line and is skipped; there is no block comment.
- Tokens are matched longest first; at equal length, the earlier rule in the table wins.
- So a bare word that is exactly a keyword, a type word, an operation word, `inf`, or `nan` is never an `IDENTIFIER`; with a leading `%` it is one.

| token | form | note |
| --- | --- | --- |
| `ENTRY`, `ROOT` | the words themselves | keywords |
| `TYPE` | one of the 19 element type words of S7.3 | |
| `OPERATOR` | one of the operation words of S8.1 | |
| symbols | `= ( ) { } [ ] : ; ,` | |
| `INT` | `[0-9]+` | decimal; leading zeros do not change its value |
| `NUMBER` | `-?` then digits with optional `.digits` and exponent, or `-?` then `inf` or `nan` | a bare unsigned integer is `INT`, not `NUMBER` |
| `EXPRESSION` | a backtick, any characters but a backtick, a backtick | its inside is parsed by the expression grammar |
| `IDENTIFIER` | optional `%`, then `[a-zA-Z_]`, then `[a-zA-Z0-9_.]*` | may contain dots |
| `COMMENT`, `WS` | `//` to end of line; whitespace | skipped |
| `ERROR_CHAR` | any other single character | no rule accepts it, so the text is rejected |

### S2.2 Expression tokens

- Tokens are matched longest first; at equal length, the earlier rule in the table wins.

| token | form | note |
| --- | --- | --- |
| `AND`, `OR` | `and`, `or` | keywords |
| `ADDRESSING` | `@a.` then a name | a reference to an addressing attribute |
| `COMPUTATIONAL` | `@c.` then a name | a reference to a computational attribute |
| `STATE` | `@s.` then a name | a reference to a state variable |
| `ASSIGN` | `=` | only in a state update |
| comparisons | `== != < <= > >=` | grammar rule `compare` |
| arithmetic | `+ - * / %` and `( )` | |
| `INT` | `0` or `[1-9][0-9]*` | decimal, unsigned; no leading zero |
| `WORD` | a name | no rule accepts it, so the text is rejected |
| `WS` | whitespace | skipped |
| `ERROR_CHAR` | any other single character | rejected |

- A name in an expression is `[a-zA-Z_][a-zA-Z0-9_]*`: unlike a semantics `IDENTIFIER`, it has no dot.

## S3 Grammar

### S3.1 Semantics grammar

- The entry rule of `IDLV2.g4` is `module`.
- The rule `operation` is one line of a semantics (S7.1).
- The rule `name` accepts `IDENTIFIER` or `OPERATOR`, so an attribute name such as `slice` is read by its text.

### S3.2 Expression grammar

The front end parses each text by the entry rule of its place in `IDLV2Expr.g4`:

| place | entry rule |
| --- | --- |
| a start or a size of a region | `expression_text` |
| a backtick expression in a semantics | `expression_text` |
| a constraint | `constraint_text` |
| a state update | `update_text` |

- `*`, `/`, and `%` bind tighter than `+` and `-`; each binary operator groups from the left.
- Unary `-` binds tighter than every binary operator.
- `and` binds tighter than `or`.
- A comparison takes exactly two operands, so `0 <= @c.n < 8` is rejected.
