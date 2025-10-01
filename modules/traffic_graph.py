from dataclasses import dataclass
from enum import Enum

from collections import deque


class NeighbourScope(Enum):
    """
    Enumeration for the different scopes of neighboring intersections.

    Attributes:
        ALL (str): Include all neighboring intersections.
        UPSTREAM (str): Include only upstream neighboring intersections.
        DOWNSTREAM (str): Include only downstream neighboring intersections.
    """

    ALL = "ALL"
    UPSTREAM = "UPSTREAM"
    DOWNSTREAM = "DOWNSTREAM"


@dataclass(frozen=True)
class NeighbourInfo:
    """
    Information about a neighboring intersection.

    Attributes:
        node_id (str): The traffic light system ID of the neighboring intersection.
        hop (int): The number of hops to reach the neighboring intersection.
        discount (float): The discount factor based on the number of hops.
    """

    node_id: str
    hop: int
    discount: float


@dataclass
class TrafficGraphConfig:
    """
    Configuration for the traffic graph used in graph-based models.

    Attributes:
        neighbour_scope (NeighbourScope): Scope of neighboring intersections to include. Options are ALL, UPSTREAM, or DOWNSTREAM.
        hops (int): Number of hops to consider when building the graph.
        fixed_discount (float): Fixed discount factor per hop if the model does not use Graph Attention Networks (GAT).
        allow_self_loops (bool): Whether to allow self-loops in the graph.
        nodes (list[str]): List of traffic light system IDs representing the nodes in the graph.
        edges (list[tuple[str, str]] | None): List of directed edges between nodes. Each edge is a tuple of (from_node, to_node). If None, intersections will be completely independent.
    """

    neighbour_scope: NeighbourScope
    hops: int
    fixed_discount: float
    allow_self_loops: bool
    nodes: list[str]
    edges: list[tuple[str, str]] | None


class TrafficGraph:

    def __init__(self, config: TrafficGraphConfig):

        self._validate_config(config)

        self._neighbour_scope: NeighbourScope = config.neighbour_scope
        self._hops: int = config.hops
        self._fixed_discount: float = config.fixed_discount
        self._nodes: list[str] = config.nodes
        self._edges: list[tuple[str, str]] | None = config.edges

        # Build adjacency list for quick lookup of neighbors
        self._build_adjacency_lists()

        self._neighbour_table_cache = None

    # ---- Validation Methods ---- #

    @staticmethod
    def _validate_config(config: TrafficGraphConfig):
        """Validate the TrafficGraphConfig parameters."""
        if not isinstance(config.neighbour_scope, NeighbourScope):
            raise ValueError(
                "TrafficGraphConfig Error: neighbour_scope must be NeighbourScope"
            )

        if config.hops < 0:
            raise ValueError(
                "TrafficGraphConfig Error: Hops must be an integer greater than or equal to zero."
            )

        if not (0.0 < config.fixed_discount <= 1.0):
            raise ValueError(
                "TrafficGraphConfig Error: Fixed_discount must be in the range (0.0, 1.0]"
            )

        if not config.nodes or len(config.nodes) == 0:
            raise ValueError(
                "TrafficGraphConfig Error: Nodes must be a non-empty list of traffic light system IDs"
            )

        if len(config.nodes) != len(set(config.nodes)):
            raise ValueError(
                "TrafficGraphConfig Error: Nodes must be unique traffic light system IDs"
            )

        if config.edges is not None:

            seen: list[tuple[str, str]] = []
            for edge in config.edges:

                if edge in seen:
                    raise ValueError(
                        "TrafficGraphConfig Error: Edges must be unique, but duplicate found"
                    )

                if len(edge) != 2:
                    raise ValueError(
                        "TrafficGraphConfig Error: Each edge must be a tuple of (from_node, to_node)"
                    )

                if edge[0] not in config.nodes or edge[1] not in config.nodes:
                    raise ValueError(
                        "TrafficGraphConfig Error: Edges must reference valid nodes in the nodes list"
                    )

                if not config.allow_self_loops and edge[0] == edge[1]:
                    raise ValueError(
                        "TrafficGraphConfig Error: Self-loops are not allowed but found in edges list"
                    )

                seen.append(edge)

    # ---- Utility Methods ---- #

    def _build_adjacency_lists(self):
        """Build adjacency lists for quick neighbor lookup."""
        self._adjacency_list_out = {node: set() for node in self._nodes}
        self._adjacency_list_in = {node: set() for node in self._nodes}
        self._adjacency_list_all = {node: set() for node in self._nodes}

        if self._edges is not None:
            for from_node, to_node in self._edges:
                self._adjacency_list_out[from_node].add(to_node)
                self._adjacency_list_in[to_node].add(from_node)

        self._adjacency_list_all = {
            node: self._adjacency_list_out[node].union(self._adjacency_list_in[node])
            for node in self._nodes
        }

    # ---- Setters ---- #

    def set_scope(self, scope: NeighbourScope) -> None:
        """Set the neighbour scope for the graph."""
        if not isinstance(scope, NeighbourScope):
            raise ValueError(
                "TrafficGraph Error: Scope must be an instance of NeighbourScope Enum"
            )
        self._neighbour_scope = scope

        self._neighbour_table_cache = None  # Invalidate cache on scope change

    def set_fixed_discount(self, discount: float) -> None:
        """Set the fixed discount factor for the graph."""
        if not (0.0 < discount <= 1.0):
            raise ValueError(
                "TrafficGraph Error: Fixed_discount must be in the range (0.0, 1.0]"
            )
        self._fixed_discount = discount

        self._neighbour_table_cache = None  # Invalidate cache on discount change

    def set_hops(self, hops: int) -> None:
        """Set the number of hops for the graph."""
        if hops < 0:
            raise ValueError(
                "TrafficGraphConfig Error: Hops must be an integer greater than or equal to zero."
            )
        self._hops = hops

        self._neighbour_table_cache = None  # Invalidate cache on hops change

    # ---- Getters ---- #

    def get_neighbours(self, node_id: str) -> list[NeighbourInfo]:
        """
        Get neighbours of `node_id` under current scope and hops, keeping shortest hops.

        Returns:
            list[NeighbourInfo]
            where NeighbourInfo = (node_id, hop, discount)
        """
        if node_id not in self._nodes:
            raise ValueError(f"TrafficGraph Error: Node ID '{node_id}' not found")

        # Independent intersections or zero-hop => no neighbours
        if self._edges is None or self._hops == 0:
            return []

        # BFS with shortest-hop tracking
        seen: dict[str, int] = {}  # neighbour -> hop
        q: deque[tuple[str, int]] = deque([(node_id, 0)])

        while q:
            u, d = q.popleft()
            if d >= self._hops:
                continue

            if self._neighbour_scope == NeighbourScope.ALL:
                next_nodes = self._adjacency_list_all[u]
            elif self._neighbour_scope == NeighbourScope.UPSTREAM:
                next_nodes = self._adjacency_list_in[u]
            elif self._neighbour_scope == NeighbourScope.DOWNSTREAM:
                next_nodes = self._adjacency_list_out[u]
            else:
                raise ValueError("TrafficGraph Error: Invalid neighbour scope")

            nd = d + 1
            for v in next_nodes:
                if v == node_id:  # skip self even if self-loop exists
                    continue
                # record only if unseen or found with shorter hop
                if v not in seen or nd < seen[v]:
                    seen[v] = nd
                    q.append((v, nd))

        # Deterministic ordering by neighbour id

        neighbours = [
            NeighbourInfo(v, seen[v], self._fixed_discount ** seen[v])
            for v in sorted(seen.keys())
        ]

        return neighbours

    def get_neighbour_table(self) -> dict[str, list[NeighbourInfo]]:
        """Precompute MARL-ready neighbours for all nodes under current settings."""

        if self._neighbour_table_cache is not None:
            return self._neighbour_table_cache

        neighbour_table = {u: self.get_neighbours(u) for u in self._nodes}

        self._neighbour_table_cache = neighbour_table

        return neighbour_table
