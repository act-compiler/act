grammar IDLV2;

// === Parser Rules ===

module       : ENTRY IDENTIFIER LBRACE instruction+ RBRACE ;

instruction  : (ROOT)? IDENTIFIER EQUAL result_type OPERATION LPAREN operands? RPAREN attributes? SEMICOLON? ;

result_type       : TYPE LBRACKET shape RBRACKET ;

operands     : operand (COMMA operand)* ;
operand      : value ;

attributes   : COMMA attribute (COMMA attribute)* ;
attribute    : IDENTIFIER EQUAL attributeValue ;

attributeValue : braceList | sliceList | value ;
braceList    : LBRACE (value (COMMA value)*)? RBRACE ;
sliceList    : LBRACE ( LBRACKET value COLON value (COLON value)? RBRACKET (COMMA LBRACKET value COLON value (COLON value)? RBRACKET)*)? RBRACE ;

shape : (value) (COMMA (value))*;

value : INT | IDENTIFIER | EXPRESSION;

// === Lexer Rules ===

// Keywords
ENTRY    : 'ENTRY' ;
ROOT     : 'ROOT' ;

// Types
// Probably can just use IDENTIFIER
TYPE     : 's8' | 's32' | 'u8' | 'u32' | 'f32' | 'f16' | 'bf16' | 'pred' ;

// Operations (from template.py function_templates)
OPERATION : 'reshape' | 'convert' | 'copy' | 'exp' | 'concatenate' | 'bitcast_convert'
          | 'transpose' | 'slice' | 'dot' | 'constant' | 'broadcast' | 'maximum'
          | 'minimum' | 'select_lt' | 'select_eq_var' | 'xor' | 'add'
          | 'dynamic_update_slice' | 'subtract' | 'multiply' | 'divide' | 'reduce'
          | 'parameter' | 'reduce_add' | 'exponential'
          | 'power' | 'logistic' | 'tanh' | 'log-plus-one' | 'abs'
          | 'compare' | 'select' | 'negate' | 'and' | 'or' | 'shift-left'
          | 'shift-right-logical' | 'shift-right-arithmetic' | 'iota';

// Symbols
EQUAL    : '=' ;
LPAREN   : '(' ;
RPAREN   : ')' ;
LBRACE   : '{' ;
RBRACE   : '}' ;
LBRACKET : '[' ;
RBRACKET : ']' ;
COLON    : ':' ;
SEMICOLON: ';' ;
COMMA    : ',' ;

// Tokens
INT        : [0-9]+ ;
EXPRESSION : '`' (~'`')* '`' ;
IDENTIFIER : '%'?[a-zA-Z_@][a-zA-Z0-9_.@]* ;
WS         : [ \t\r\n]+ -> skip ;
