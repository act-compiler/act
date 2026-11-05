# Grammar

This page is S2 and S3 of the spec.
The ANTLR4 grammars `taidl/antlr4/IDLV2.g4` (semantics) and `taidl/antlr4/IDLV2Expr.g4` (expressions) are normative.
The rules below are copied from them; where the two disagree, the `.g4` file holds.

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
| `OPERATION` | one of the operation words of S8.1 | |
| symbols | `= ( ) { } [ ] : ; ,` | |
| `INT` | `[0-9]+` | unsigned; leading zeros allowed |
| `NUMBER` | `-?` then digits with optional `.digits` and exponent, or `inf` or `nan` | a bare unsigned integer is `INT`, not `NUMBER` |
| `EXPRESSION` | a backtick, any characters but a backtick, a backtick | its inside is parsed by S3.2 |
| `IDENTIFIER` | optional `%`, then `[a-zA-Z_]`, then `[a-zA-Z0-9_.]*` | may contain dots |
| `ERROR_CHAR` | any other single character | no rule accepts it, so the text is rejected |

### S2.2 Expression tokens

| token | form | note |
| --- | --- | --- |
| `AND`, `OR` | `and`, `or` | keywords |
| `ADDRESSING` | `@a.` then a name | a reference to an addressing attribute |
| `COMPUTATIONAL` | `@c.` then a name | a reference to a computational attribute |
| `STATE` | `@s.` then a name | a reference to a state variable |
| `ASSIGN` | `=` | only in a state update |
| comparisons | `== != < <= > >=` | |
| arithmetic | `+ - * / %` and `( )` | |
| `INT` | `0` or `[1-9][0-9]*` | unsigned; no leading zero |
| `WORD` | a name | no rule accepts it, so the text is rejected |
| `ERROR_CHAR` | any other single character | rejected |

- A name in an expression is `[a-zA-Z_][a-zA-Z0-9_]*`: unlike a semantics `IDENTIFIER`, it has no dot.

## S3 Grammar

### S3.1 Semantics grammar

The entry rule is `module`.
The rule `instruction` is one line of a semantics (S7.1), not an instruction of S5.

```antlr
module          : ENTRY LBRACE instruction+ RBRACE EOF ;

instruction     : ROOT? IDENTIFIER EQUAL result_type OPERATION LPAREN operands? RPAREN attributes? SEMICOLON? ;

result_type     : TYPE LBRACKET shape? RBRACKET ;

operands        : operand (COMMA operand)* ;
operand         : value ;

attributes      : COMMA attribute (COMMA attribute)* ;
attribute       : name EQUAL attribute_value ;

attribute_value : brace_list | slice_list | value ;
brace_list      : LBRACE (value (COMMA value)*)? RBRACE ;
slice_list      : LBRACE LBRACKET value COLON value (COLON value)? RBRACKET (COMMA LBRACKET value COLON value (COLON value)? RBRACKET)* RBRACE ;

shape           : dim (COMMA dim)* ;
dim             : INT | EXPRESSION ;

value           : INT | NUMBER | IDENTIFIER | EXPRESSION ;

name            : IDENTIFIER | OPERATION ;
```

### S3.2 Expression grammar

The front end parses each text by the entry rule of its place:

| place | entry rule |
| --- | --- |
| a start or a size of a region | `expression_text` |
| a backtick expression in a semantics | `expression_text` |
| a constraint | `constraint_text` |
| a state update | `update_text` |

```antlr
// One entry per place: a value, a constraint, an update
expression_text : expression EOF ;
constraint_text : condition EOF ;
update_text     : update EOF ;

// A state variable takes a value
update          : STATE ASSIGN expression ;

// A value: an integer, by rising precedence
expression      : additive ;
additive        : multiplicative ((PLUS | MINUS) multiplicative)* ;
multiplicative  : unary ((STAR | SLASH | PERCENT) unary)* ;
unary           : MINUS unary | atom ;
atom            : INT | reference | LPAREN expression RPAREN ;
reference       : ADDRESSING | COMPUTATIONAL | STATE ;

// A condition: true or false, by rising precedence
condition       : conjunction (OR conjunction)* ;
conjunction     : comparison (AND comparison)* ;
comparison      : additive compare additive | LPAREN condition RPAREN ;
```

- `*`, `/`, and `%` bind tighter than `+` and `-`; each binary operator groups from the left.
- `and` binds tighter than `or`.
- A comparison takes exactly two operands, so `0 <= @c.n < 8` is rejected.
