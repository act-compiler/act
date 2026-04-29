"""
GIR graph loader for NNSmith fuzz kernel generation.

Loads NNSmith GIR pickle, builds a computation graph, and post-processes it
for HLO emission targeting Gemmini s8[16,16] tiles.
"""

import os
import pickle
import random
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from nnsmith.materialize.tensorflow.dialect import TFMatMul
from nnsmith.abstract.op import Input, Constant, Add, Sub, Min, Max, Neg, Abs, Clip

TILE = 16
TYPE = "s8"

# All NNSmith ops we expect (anything else is an error)
EXPECTED_OPS = (Input, Constant, TFMatMul, Add, Sub, Min, Max, Neg, Abs, Clip)


class GIRGraph:
    """
    NNSmith GIR computation graph with post-processed annotations for HLO emission.

    Attributes:
        producers: dict[int, list[int]]    -- node -> ordered producer ids (from GIR args)
        consumers: dict[int, list[int]]    -- node -> consumer ids (from GIR users)
        order: list[int]                   -- topological order (producers before consumers)
        node_kind: dict[int, str]          -- node -> XLA-HLO op kind:
            "parameter"           : s8[16,16] input
            "broadcast_parameter" : s8[16,1] input + broadcast
            "constant"     : s8[] constant + broadcast
            "dot"                 : matrix multiply
            "minimum"             : minimum with scalar operand
            "maximum"             : maximum with scalar operand
            "clamp"               : clamp with scalar bounds
            "reverse"             : reverse (mapped from NNSmith Abs)
            "negate"              : unary negate
            "add"                 : binary add (or replaced min/max)
            "subtract"            : binary subtract (or replaced min/max)
        scalar_values: dict[int, int]      -- constant value for constant nodes
        clip_bounds: dict[int, (int,int)]  -- (lo, hi) for clamp nodes
        shapes: dict[int, tuple]           -- per-node tensor shape
        broadcasts: dict[int, str]         -- node id -> broadcast HLO name
        reduce_output: bool                -- whether to reduce final output to 1D
    """

    def __init__(self, gir_path, seed=None):
        if seed is not None:
            random.seed(seed)

        self.id_map, self.producers, self.consumers = self.build_graph(gir_path)
        self.order = self.top_sort()

        self.node_kind = {}
        self.scalar_values = {}
        self.clip_bounds = {}
        self.shapes = {}
        self.broadcasts = {}

        self.infer_ops()
        self.infer_shapes()
        self.reduce_output = random.random() < 0.5

    def build_graph(self, gir_path):
        """Load GIR pickle and build adjacency lists."""
        with open(gir_path, 'rb') as f:
            gir = pickle.load(f)

        id_map = {}
        for inst in gir.insts:
            op = inst.iexpr.op
            if not isinstance(op, EXPECTED_OPS):
                raise ValueError(f"Unexpected op: {type(op).__name__} (node {inst.identifier})")
            id_map[inst.identifier] = inst

        # Forward edges: node -> consumers
        consumers = defaultdict(list)
        for inst in gir.insts:
            for user in inst.users[0]:
                consumers[inst.identifier].append(user.identifier)

        # Reverse edges from GIR args (preserves duplicates like min(x, x))
        arg_to_id = {f"v{inst.identifier}_0": inst.identifier for inst in gir.insts}
        producers = defaultdict(list)
        for inst in gir.insts:
            for arg in inst.iexpr.args:
                if str(arg) in arg_to_id:
                    producers[inst.identifier].append(arg_to_id[str(arg)])

        return id_map, producers, consumers

    def top_sort(self):
        """Deterministic topological sort via consumers. Returns producers-first order.

        Ties are broken by node id (sorted) to ensure reproducibility.
        """
        all_nodes = set(self.consumers.keys()) | {n for ns in self.consumers.values() for n in ns}
        in_degree = {n: 0 for n in all_nodes}
        for node in self.consumers:
            for child in self.consumers[node]:
                in_degree[child] += 1

        queue = sorted([n for n in in_degree if in_degree[n] == 0])
        sorted_nodes = []
        while queue:
            current = queue.pop(0)
            sorted_nodes.append(current)
            newly_ready = []
            for child in self.consumers.get(current, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    newly_ready.append(child)
            queue = sorted(queue + newly_ready)

        if len(sorted_nodes) != len(all_nodes):
            raise ValueError("Graph contains a cycle")
        return sorted_nodes

    def infer_ops(self):
        """Infer op kind for each node.

        1. Each Input -> "parameter" or "broadcast_parameter" (random)
        2. Each Constant -> "constant"
        3. For min/max:
           - if a producer is already "constant" -> keep as minimum/maximum
           - else if a producer is "parameter"/"broadcast_parameter" -> upgrade to "constant"
           - else neither producer is Input/Constant -> replace with add/subtract
        4. Other ops -> map to XLA-HLO name
        5. Assert no node is inferred twice
        """
        def set_kind(node_id, kind):
            if node_id in self.node_kind:
                raise ValueError(
                    f"Op conflict for node {node_id}: already {self.node_kind[node_id]}, trying {kind}")
            self.node_kind[node_id] = kind

        # assign all Input and Constant nodes
        for node_id in self.order:
            op = self.id_map[node_id].iexpr.op

            if isinstance(op, Input):
                if random.random() < 0.5:
                    set_kind(node_id, "broadcast_parameter")
                else:
                    set_kind(node_id, "parameter")

            elif isinstance(op, Constant):
                set_kind(node_id, "constant")
                self.scalar_values[node_id] = random.randint(-32, 32)

        # assign all other ops
        for node_id in self.order:
            if node_id in self.node_kind:  # skip input/constant nodes already assigned
                continue
            op = self.id_map[node_id].iexpr.op
            prods = self.producers[node_id]

            if isinstance(op, (Min, Max)):
                # Find a producer that is or can become a constant
                scalar_prod = None
                for p in prods:
                    if self.node_kind[p] == "constant":
                        scalar_prod = p
                        break
                if scalar_prod is None:
                    for p in prods:
                        if self.node_kind[p] in ("parameter", "broadcast_parameter"):
                            # Upgrade parameter -> constant (direct assign, not set_kind)
                            self.node_kind[p] = "constant"
                            self.scalar_values[p] = random.randint(-32, 32)
                            scalar_prod = p
                            break
                if scalar_prod is not None:
                    set_kind(node_id, "minimum" if isinstance(op, Min) else "maximum")
                else:
                    set_kind(node_id, random.choice(["add", "subtract"]))

            elif isinstance(op, Clip):
                set_kind(node_id, "clamp")
                self.clip_bounds[node_id] = (random.randint(-32, -1), random.randint(1, 32))

            elif isinstance(op, TFMatMul):
                set_kind(node_id, "dot")

            elif isinstance(op, Abs):
                set_kind(node_id, "reverse")

            elif isinstance(op, Neg):
                set_kind(node_id, "negate")

            elif isinstance(op, Add):
                set_kind(node_id, "add")

            elif isinstance(op, Sub):
                set_kind(node_id, "subtract")

            else:
                raise ValueError(f"Unhandled op: {type(op).__name__} (node {node_id})")

    def infer_shapes(self):
        """Assign shapes to all nodes based on node_kind set by infer_ops.

        - "constant": (1, 1)
        - "broadcast_parameter": (TILE, 1)
        - "parameter": (TILE, TILE)
        - All other ops: inherit from producers, default (TILE, TILE)
        - Nodes with shape (1, 1) or (TILE, 1) get broadcasts
        """
        def set_shape(node, shape):
            if node in self.shapes and self.shapes[node] != shape:
                raise ValueError(
                    f"Shape conflict for node {node}: already {self.shapes[node]}, trying {shape}")
            self.shapes[node] = shape

        # Assign shapes from node_kind (all set by infer_ops)
        for node_id in self.order:
            kind = self.node_kind[node_id]
            if kind == "constant":
                set_shape(node_id, (1, 1))
            elif kind == "broadcast_parameter":
                set_shape(node_id, (TILE, 1))
            elif kind == "parameter":
                set_shape(node_id, (TILE, TILE))

        # All non-leaf ops get (TILE, TILE) since their inputs are broadcast
        for node_id in self.order:
            if node_id not in self.shapes:
                set_shape(node_id, (TILE, TILE))

        # Assign broadcasts for all non-(TILE,TILE) nodes
        for node_id in self.order:
            if self.shapes[node_id] in ((1, 1), (TILE, 1)):
                self.broadcasts[node_id] = f"broadcast_{node_id}"

    def hlo_name(self, node_id):
        """HLO instruction name for a node."""
        kind = self.node_kind[node_id]
        if kind in ("parameter", "broadcast_parameter"):
            return f"input_{node_id}"
        elif kind == "constant":
            return f"constant_{node_id}"
        else:
            return f"{kind}_{node_id}"

    def resolved_name(self, node_id):
        """Name to use as operand: broadcast name if needed, else hlo_name."""
        if node_id in self.broadcasts:
            return self.broadcasts[node_id]
        return self.hlo_name(node_id)

    def leaves(self):
        """Nodes with no consumers (graph outputs)."""
        return [n for n in self.order if len(self.consumers.get(n, [])) == 0]
