# Abstract syntax

The abstract syntax of a TAIDL ISA specification, as the front end builds it.
Each production names the front-end field or grammar rule it abstracts.

## Declarations

```
Acc   ::= acc(name, M*, S*, I*)                 accelerator.py:10-14
M     ::= model(m, A, U, t)                     data_model.py:33-48
          A = [a_1..a_n] addressed dims, U = [u_1..u_k] unit shape, t in T
          d0 = model(d0, [], [], u8)            accelerator.py:31
S     ::= state(s, v0)                          data_model.py:28-30, v0 in Z
I     ::= instr(i, C*, P*, K*, W*, In*, Out*, B?, ip*, marks)
          C computational attrs, P addressing attrs  instruction.py:70-71
          K constraints, W updates              instruction.py:77-80
          In, Out regions                       instruction.py:42-43
          B semantics block                     instruction.py:44
          ip in-place flags, one per input      instruction.py:51 (a marker; expands per typing.md)
```

## Regions and expressions

```
R     ::= region(m, [e_1..e_n], [f_1..f_n])     instruction.py:17-18, n = |A_m| or 1 for d0
e, f  ::= k | @c.x | @a.x | @s.x | -e | e + e | e - e | e * e | e / e | e % e
                                                IDLV2Expr.g4:113-118
cond  ::= e cmp e | cond and cond | cond or cond | (cond)   IDLV2Expr.g4:121-124
cmp   ::= == | != | < | <= | > | >=
upd   ::= @s.x = e                              IDLV2Expr.g4:110
k     ::= decimal literal, -2^63 <= k < 2^63     instruction.py:400-408
```

## Semantics block

```
B     ::= ENTRY { L+ }                          IDLV2.g4:6
L     ::= [ROOT] %v = t[d_1..d_r] op(o*) attrs   IDLV2.g4:8
d     ::= k | `e`                               IDLV2.g4:23
o     ::= %v | k | lit                          IDLV2.g4:25, operations.py:191-217
attrs ::= (name = av)*                          IDLV2.g4:15-20
av    ::= scalar | {v*} | {[v:v(:v)?]+}          IDLV2.g4:18-20
op    ::= one of the 78 words                   IDLV2.g4:45-74, operations.py:41-165
t     ::= pred | s2..s64 | u2..u64 | f8e4m3fn | f8e5m2 | f16 | bf16 | f32 | f64
                                                data_model.py:4-24
```

## Conventions

- `%In<k>` is the value of input region k and is defined by `parameter(k)` alone (instruction.py:290-307).
- `%Out<k>` is the value written to output region k; `ROOT` marks the highest k (instruction.py:308-318).
- A region's value has type `t_m[f_1..f_n, u_1..u_k]`, or `u8[f_1]` for d0 (semantics.md:43).
- An instruction with no B has no regions and only updates state (semantics.md:16).
