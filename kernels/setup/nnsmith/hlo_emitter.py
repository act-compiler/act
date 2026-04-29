"""
HLO emitter for NNSmith fuzz kernels.

Emits XLA-HLO IR from a post-processed GIRGraph.
All tensors are s8 with shapes (16,16), (1,16), or (1,1).
"""

from gir_graph import TILE, TYPE

FULL = (TILE, TILE)
COL_VEC = (TILE, 1)
SCALAR = (1, 1)


def fmt(shape):
    """Format shape as HLO type string."""
    if shape == SCALAR:
        return f"{TYPE}[]"
    return f"{TYPE}[{shape[0]},{shape[1]}]"


def generate_hlo(g, output_path):
    """
    Emit XLA-HLO IR from a GIRGraph and write to output_path.
    Returns True on success.
    """
    inst = {}  # HLO name -> HLO line (insertion-ordered)
    param_idx = 0

    for nid in g.order:
        kind = g.node_kind[nid]
        name = g.hlo_name(nid)
        shape = fmt(g.shapes[nid])
        prods = [g.resolved_name(p) for p in g.producers[nid]]

        if kind in ("parameter", "broadcast_parameter"):
            inst[name] = f"{name} = {shape} parameter({param_idx})"
            param_idx += 1

        elif kind == "constant":
            val = g.scalar_values[nid]
            inst[name] = f"{name} = {shape} constant({val})"

        elif kind in ("minimum", "maximum"):
            inst[name] = f"{name} = {shape} {kind}({prods[0]}, {prods[1]})"

        elif kind in ("add", "subtract"):
            inst[name] = f"{name} = {shape} {kind}({prods[0]}, {prods[1]})"

        elif kind == "clamp":
            lo, hi = g.clip_bounds[nid]
            lo_n, hi_n = f"clamp_lo_{nid}", f"clamp_hi_{nid}"
            lo_bc, hi_bc = f"clamp_lo_bc_{nid}", f"clamp_hi_bc_{nid}"
            inst[lo_n] = f"{lo_n} = {fmt(SCALAR)} constant({lo})"
            inst[lo_bc] = f"{lo_bc} = {fmt(FULL)} broadcast({lo_n}), dimensions={{}}"
            inst[hi_n] = f"{hi_n} = {fmt(SCALAR)} constant({hi})"
            inst[hi_bc] = f"{hi_bc} = {fmt(FULL)} broadcast({hi_n}), dimensions={{}}"
            inst[name] = f"{name} = {shape} clamp({lo_bc}, {prods[0]}, {hi_bc})"

        elif kind == "dot":
            inst[name] = f"{name} = {shape} dot({prods[0]}, {prods[1]}), lhs_contracting_dims={{1}}, rhs_contracting_dims={{0}}"

        elif kind == "reverse":
            inst[name] = f"{name} = {shape} reverse({prods[0]}), dimensions={{0,1}}"

        elif len(prods) == 1:
            inst[name] = f"{name} = {shape} {kind}({prods[0]})"

        elif len(prods) >= 2:
            inst[name] = f"{name} = {shape} {kind}({prods[0]}, {prods[1]})"

        # Emit broadcast if this node has a non-full shape
        if nid in g.broadcasts:
            bc = g.broadcasts[nid]
            s = g.shapes[nid]
            if s == SCALAR:
                inst[bc] = f"{bc} = {fmt(FULL)} broadcast({name}), dimensions={{}}"
            elif s == COL_VEC:
                inst[bc] = f"{bc} = {fmt(FULL)} broadcast({name}), dimensions={{0,1}}"

    # Combine multiple leaf outputs with add
    leaf_names = [g.hlo_name(n) for n in g.leaves()
                  if g.hlo_name(n) in inst and g.node_kind[n] != "constant"]

    root = None
    if len(leaf_names) > 1:
        for i in range(len(leaf_names) - 1):
            cn = f"concat_{i}"
            prev = leaf_names[i] if i == 0 else f"concat_{i-1}"
            inst[cn] = f"{cn} = {fmt(FULL)} add({prev}, {leaf_names[i+1]})"
        root = f"concat_{len(leaf_names) - 2}"
    elif leaf_names:
        root = leaf_names[0]
    else:
        root = list(inst.keys())[-1] if inst else None

    if root is None:
        return False

    # Build header
    param_shapes = [fmt(g.shapes[nid]) for nid in g.order
                    if g.node_kind[nid] in ("parameter", "broadcast_parameter")]
    output_type = fmt(COL_VEC) if g.reduce_output else fmt(FULL)

    lines = []
    lines.append(
        f"HloModule xla_computation_unknown, entry_computation_layout={{({','.join(param_shapes)})->{output_type}}}")
    lines.append("")
    lines.append("reduce_add {")
    lines.append(f"  lhs = {fmt(SCALAR)} parameter(0)")
    lines.append(f"  rhs = {fmt(SCALAR)} parameter(1)")
    lines.append(f"  ROOT sum = {fmt(SCALAR)} add(lhs, rhs)")
    lines.append("}")
    lines.append("")
    lines.append("ENTRY main {")
    lines.append(f"  reduce_init = {fmt(SCALAR)} constant(0)")

    for inst_line in inst.values():
        lines.append(f"  {inst_line}")

    if g.reduce_output:
        lines.append(
            f"  {root}_reduced = {TYPE}[{TILE}] reduce({root}, reduce_init), dimensions={{0}}, to_apply=reduce_add")
        lines.append(
            f"  ROOT {root}_reshaped = {fmt(COL_VEC)} reshape({root}_reduced)")
    else:
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith(f"{root} ="):
                lines[i] = "  ROOT " + lines[i].strip()
                break

    lines.append("}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    return True
