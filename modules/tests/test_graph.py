# modules/tests/test_traffic_graph.py
import math
import pytest

from modules.traffic_graph import (
    TrafficGraph,
    TrafficGraphConfig,
    NeighbourScope,
    NeighbourInfo,
)


# ------------------------- Fixtures -------------------------

@pytest.fixture
def nodes():
    # tlN1..tlN21
    return [f"tlN{i}" for i in range(1, 22)]


@pytest.fixture
def edges():
    # From your YAML (directed)
    return [
        ("tlN1","tlN2"), ("tlN2","tlN1"),
        ("tlN2","tlN10"), ("tlN10","tlN2"),
        ("tlN8","tlN3"), ("tlN3","tlN8"),
        ("tlN8","tlN7"), ("tlN7","tlN6"), ("tlN6","tlN7"),
        ("tlN10","tlN8"),
        ("tlN8","tlN9"), ("tlN9","tlN8"),
        ("tlN9","tlN11"),
        ("tlN11","tlN13"), ("tlN13","tlN11"),
        ("tlN11","tlN10"), ("tlN10","tlN11"),
        ("tlN7","tlN9"),
        ("tlN3","tlN4"), ("tlN4","tlN3"),
        ("tlN4","tlN6"), ("tlN6","tlN4"),
        ("tlN5","tlN4"), ("tlN4","tlN5"),
        ("tlN5","tlN6"), ("tlN6","tlN5"),
        ("tlN13","tlN12"), ("tlN12","tlN13"),
        ("tlN14","tlN13"), ("tlN13","tlN14"),
        ("tlN9","tlN14"), ("tlN14","tlN9"),
        ("tlN14","tlN17"), ("tlN17","tlN14"),
        ("tlN17","tlN16"), ("tlN16","tlN17"),
        ("tlN16","tlN15"), ("tlN15","tlN16"),
        ("tlN15","tlN7"), ("tlN7","tlN15"),
        ("tlN17","tlN18"), ("tlN18","tlN17"),
        ("tlN18","tlN19"), ("tlN19","tlN18"),
        ("tlN19","tlN20"), ("tlN20","tlN19"),
        ("tlN18","tlN21"), ("tlN21","tlN18"),
        ("tlN21","tlN15"), ("tlN15","tlN21"),
    ]


@pytest.fixture
def base_config(nodes, edges):
    return TrafficGraphConfig(
        neighbour_scope=NeighbourScope.ALL,
        hops=1,
        fixed_discount=0.9,
        allow_self_loops=False,
        nodes=nodes,
        edges=edges,
    )


@pytest.fixture
def graph(base_config):
    return TrafficGraph(base_config)


# ------------------------- Validation -------------------------

def test_validation_unknown_nodes(nodes):
    cfg = TrafficGraphConfig(
        neighbour_scope=NeighbourScope.ALL,
        hops=1,
        fixed_discount=0.9,
        allow_self_loops=False,
        nodes=nodes,
        edges=[("tlN1", "tlNXXX")],  # unknown
    )
    with pytest.raises(ValueError, match="Edges must reference valid nodes"):
        TrafficGraph(cfg)


def test_validation_duplicate_nodes(nodes, edges):
    bad_nodes = nodes + [nodes[0]]
    cfg = TrafficGraphConfig(
        neighbour_scope=NeighbourScope.ALL,
        hops=1,
        fixed_discount=0.9,
        allow_self_loops=False,
        nodes=bad_nodes,
        edges=edges,
    )
    with pytest.raises(ValueError, match="Nodes must be unique"):
        TrafficGraph(cfg)


def test_validation_duplicate_edges(nodes):
    cfg = TrafficGraphConfig(
        neighbour_scope=NeighbourScope.ALL,
        hops=1,
        fixed_discount=0.9,
        allow_self_loops=False,
        nodes=nodes,
        edges=[("tlN1","tlN2"), ("tlN1","tlN2")],
    )
    with pytest.raises(ValueError, match="Duplicate edges"):
        TrafficGraph(cfg)


def test_validation_self_loops_disallowed(nodes):
    cfg = TrafficGraphConfig(
        neighbour_scope=NeighbourScope.ALL,
        hops=1,
        fixed_discount=0.9,
        allow_self_loops=False,
        nodes=nodes,
        edges=[("tlN1","tlN1")],
    )
    with pytest.raises(ValueError, match="Self-loops are not allowed"):
        TrafficGraph(cfg)


def test_validation_self_loops_allowed(nodes):
    cfg = TrafficGraphConfig(
        neighbour_scope=NeighbourScope.ALL,
        hops=1,
        fixed_discount=0.9,
        allow_self_loops=True,
        nodes=nodes,
        edges=[("tlN1","tlN1")],
    )
    g = TrafficGraph(cfg)
    # Even if allowed, get_neighbours should skip returning self
    assert all(n.node_id != "tlN1" for n in g.get_neighbours("tlN1"))


def test_validation_hops_and_discount_bounds(nodes):
    with pytest.raises(ValueError):
        TrafficGraph(TrafficGraphConfig(
            neighbour_scope=NeighbourScope.ALL,
            hops=-1, fixed_discount=0.9,
            allow_self_loops=False, nodes=nodes, edges=[]
        ))
    with pytest.raises(ValueError):
        TrafficGraph(TrafficGraphConfig(
            neighbour_scope=NeighbourScope.ALL,
            hops=1, fixed_discount=0.0,
            allow_self_loops=False, nodes=nodes, edges=[]
        ))


# ------------------------- Basic behaviour -------------------------

def test_zero_hop_returns_empty(graph):
    graph.set_hops(0)
    assert graph.get_neighbours("tlN8") == []


def test_none_edges_independent(nodes):
    g = TrafficGraph(TrafficGraphConfig(
        neighbour_scope=NeighbourScope.ALL,
        hops=2,
        fixed_discount=0.9,
        allow_self_loops=False,
        nodes=nodes,
        edges=None,
    ))
    assert g.get_neighbours("tlN1") == []
    assert g.get_neighbour_table() == {n: [] for n in nodes}


def test_downstream_1hop(graph):
    graph.set_scope(NeighbourScope.DOWNSTREAM)
    graph.set_hops(1)
    nbrs = graph.get_neighbours("tlN8")
    got = {(n.node_id, n.hop, n.discount) for n in nbrs}
    exp = {
        ("tlN3", 1, 0.9**1),
        ("tlN7", 1, 0.9**1),
        ("tlN9", 1, 0.9**1),
    }
    assert got == exp


def test_upstream_1hop(graph):
    graph.set_scope(NeighbourScope.UPSTREAM)
    graph.set_hops(1)
    nbrs = graph.get_neighbours("tlN9")
    got_ids = sorted(n.node_id for n in nbrs)
    # upstream into tlN9: tlN8, tlN14, tlN7
    assert got_ids == ["tlN14", "tlN7", "tlN8"]
    assert all(n.hop == 1 and math.isclose(n.discount, 0.9) for n in nbrs)


def test_all_1hop(graph):
    graph.set_scope(NeighbourScope.ALL)
    graph.set_hops(1)
    nbrs = graph.get_neighbours("tlN9")
    got = sorted(n.node_id for n in nbrs)
    # union of upstream {8,14,7} and downstream {11,14} -> {7,8,11,14}
    assert got == ["tlN11", "tlN14", "tlN7", "tlN8"]


def test_downstream_2hop_shortest_paths(graph):
    graph.set_scope(NeighbourScope.DOWNSTREAM)
    graph.set_hops(2)
    nbrs = graph.get_neighbours("tlN8")
    # 1-hop: 3,7,9
    # 2-hop from 3 -> 4 ; from 7 -> 6,15 ; from 9 -> 11,14
    ids_by_hop = {n.node_id: n.hop for n in nbrs}
    for nid in ["tlN3","tlN7","tlN9"]:
        assert ids_by_hop[nid] == 1
    for nid in ["tlN4","tlN6","tlN15","tlN11","tlN14"]:
        assert ids_by_hop[nid] == 2
    # discounts per hop
    for n in nbrs:
        assert math.isclose(n.discount, 0.9 ** n.hop)


def test_shortest_hop_if_multiple_paths(graph):
    # Ensure shortest-hop kept. Example: tlN14 can be reached from tlN8 via tlN9 in 2 hops,
    # and other longer paths might exist; we must record hop=2.
    graph.set_scope(NeighbourScope.DOWNSTREAM)
    graph.set_hops(3)
    nbrs = {n.node_id: n.hop for n in graph.get_neighbours("tlN8")}
    assert nbrs["tlN14"] == 2


def test_sorted_output_determinism(graph):
    graph.set_scope(NeighbourScope.DOWNSTREAM)
    graph.set_hops(1)
    nbrs = graph.get_neighbours("tlN7")
    ids = [n.node_id for n in nbrs]
    assert ids == sorted(ids)


# ------------------------- Cache behaviour -------------------------

def test_neighbour_table_cache_identity(graph):
    graph.set_scope(NeighbourScope.ALL)
    graph.set_hops(1)
    t1 = graph.get_neighbour_table()
    t2 = graph.get_neighbour_table()
    # same object returned due to cache
    assert t1 is t2

def test_cache_invalidation_on_scope_change(graph):
    graph.set_scope(NeighbourScope.ALL)
    t1 = graph.get_neighbour_table()
    graph.set_scope(NeighbourScope.UPSTREAM)
    t2 = graph.get_neighbour_table()
    assert t1 is not t2

def test_cache_invalidation_on_hops_change(graph):
    graph.set_hops(1)
    t1 = graph.get_neighbour_table()
    graph.set_hops(2)
    t2 = graph.get_neighbour_table()
    assert t1 is not t2

def test_cache_invalidation_on_discount_change(graph):
    t1 = graph.get_neighbour_table()
    graph.set_fixed_discount(0.8)
    t2 = graph.get_neighbour_table()
    assert t1 is not t2


# ------------------------- Robustness & edge cases -------------------------

def test_empty_neighbours_for_isolated_node(nodes, edges):
    # Create an isolated node tlN99
    nodes2 = nodes + ["tlN99"]
    cfg = TrafficGraphConfig(
        neighbour_scope=NeighbourScope.ALL,
        hops=2,
        fixed_discount=0.9,
        allow_self_loops=False,
        nodes=nodes2,
        edges=edges,  # no edges touching tlN99
    )
    g = TrafficGraph(cfg)
    assert g.get_neighbours("tlN99") == []

def test_setters_validate_and_apply(graph):
    with pytest.raises(ValueError):
        graph.set_fixed_discount(1.5)
    with pytest.raises(ValueError):
        graph.set_hops(-2)
    # valid updates
    graph.set_scope(NeighbourScope.DOWNSTREAM)
    graph.set_hops(3)
    graph.set_fixed_discount(0.75)
    nbrs = graph.get_neighbours("tlN8")
    assert all(math.isclose(n.discount, 0.75 ** n.hop) for n in nbrs)
