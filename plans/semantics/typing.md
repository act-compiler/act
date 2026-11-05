# Static semantics: well-formedness and typing

Judgments for declarations, expressions, regions, and the semantics block.
Per-operation shape and dtype rules are in `typing-ops.md`.
The front end today checks the rules marked FE; the rules marked NEW are not checked anywhere in the front end.

## Environments

```
D     = {m -> model(A, U, t)}         data models
Sg    = {s -> i64}                    state variables
G_i   = (C_i, P_i)                    an instruction's attribute names
V     = {%v -> t[d*]}                 values of one semantics block
```

## Names and declarations (FE)

```
WF-NAME     name is an ASCII identifier, not a Python keyword, not taken      names.py:172-179
WF-MODEL    every a in A and u in U is an int >= 1, t in T                    accelerator.py:37-43
WF-STATE    no instruction declared yet, v0 is an int                          accelerator.py:49-57
WF-STATE64  -2^63 <= v0 < 2^63                                        NEW      declarations.md:37
WF-ATTRS    C_i and P_i pairwise distinct                                       instruction.py:72-74
WF-ORDER    every model and state precedes the first instruction        NEW      conformance.md:61
```

## Expressions

```
G_i; Sg |- k : int                                   (-2^63 <= k < 2^63)     instruction.py:403-408
G_i; Sg |- @c.x : int        if x in C_i                                     instruction.py:340-358
G_i; Sg |- @s.x : int        if x in dom(Sg)
G_i; Sg |- @a.x : int        if x in P_i and place in {start, constraint}    expressions.md:187
G_i; Sg |- e1 : int   G_i; Sg |- e2 : int
----------------------------------------- (op in + - * / %)
G_i; Sg |- e1 op e2 : int
G_i; Sg |- e1 op e2 : int   requires e2 is not the literal 0 for / and %     instruction.py:410-415
G_i; Sg |- e1 : int  G_i; Sg |- e2 : int  ==>  G_i; Sg |- e1 cmp e2 : bool
G_i; Sg |- c1 : bool  G_i; Sg |- c2 : bool  ==>  c1 and c2 : bool, c1 or c2 : bool
G_i; Sg |- @s.x = e : upd    if x in dom(Sg) and G_i; Sg |- e : int          instruction.py:339-344
```

- The place rule for `@a.x` is NEW: the front end accepts `@a.` in a size, a dim, and an update (expressions.md:191).

## Regions

```
D(m) = model(A, U, t)   |A| = n (1 for d0)   each e_j, f_j : int
---------------------------------------------------------------- WF-REGION (FE) instruction.py:115-129
D; G_i; Sg |- region(m, [e_1..e_n], [f_1..f_n]) : t[f_1..f_n, U]
```

## Semantics block

```
B = ENTRY { L_1 .. L_p }
WF-ROOT     exactly one L_j carries ROOT, and it defines the highest %Out<k>     instruction.py:276-318 (FE)
WF-NAME%    every defined name starts with %                                     instruction.py:268-275 (FE)
WF-SSA      no name defined twice; every operand defined by an earlier line      operations.py:171-217 (FE)
WF-IN       %In<k> defined by parameter(k) and by nothing else                   instruction.py:290-307 (FE)
WF-IO       {%In<k>} = indices of In, {%Out<k>} = indices of Out                 instruction.py:142-161 (FE)
WF-ARITY    operand count and attribute set of L_j fit OPERATIONS[op]            operations.py:191-239 (FE)
WF-LIT64    every INT or NUMBER literal in a dim or attribute fits i64            instruction.py:419-437 (FE)
```

Typing a block threads V through the lines in order:

```
V_0 = {}
V_{j-1} |- L_j : (%v_j : t_j[d*_j])   ==>   V_j = V_{j-1} + {%v_j -> t_j[d*_j]}
```

A line is typed by the rule of its operation in `typing-ops.md`, with these shared premises:

```
T-LINE      every operand %o is in V_{j-1}; the result type written on the line equals the
            type the operation rule derives                                      NEW  conformance.md:59
T-PARAM     %In<k> = t[d*] parameter(k)   requires  t = t_m of In_k and d* ~ shape(In_k)
            where d* ~ s* means equal, or equal element count (presumed reshape)   semantics.md:46 (NEW in FE)
T-OUT       %Out<k> : t[d*]   requires  t = t_m of Out_k and d* ~ shape(Out_k)                        (NEW in FE)
```

- Element counts under `~` compare symbolically: equal after evaluating every dim and size for every attribute assignment allowed by the constraints.
- A front end that cannot decide `~` symbolically checks it when every dim and size is a literal, and a generator checks it at issue (act-oracle `check_tile`, utils.py:733-746).

## Constraints and updates

```
WF-CONSTR   each K_j : bool under G_i; Sg                     instruction.py:325-329 (FE), with @a. allowed
WF-UPDATE   each W_j : upd under G_i; Sg                      instruction.py:80 (FE)
WF-INPLACE  |ip| = |In|; no input both in place and placement-marked          instruction.py:163-172 (FE)
WF-INPLACE= ip_k  ==>  In_k and Out_0 name the same model and the same start texts   NEW  (see holes.md H1)
WF-MARK     a placement mark needs |Out| = 1 and inputs in Out_0's model       instruction.py:190-208 (FE)
```

## Instruction

```
every rule above holds, and (Out = [] and B absent) or (B present)             NEW  semantic_generator.py:14-19
-------------------------------------------------------------------------------- WF-INSTR
D; Sg |- instr(i, ...) ok
```
