"""Tests for the directed graph foundation used by the decoder."""

import pytest

from src.decoder.graph import DirectedGraph


def test_connected_path_preserves_direction_and_arc_ids() -> None:
    graph = DirectedGraph([10, 20, 30], [(10, 20), (20, 30)])

    assert graph.outgoing_arcs(10) == (0,)
    assert graph.outgoing_arcs(20) == (1,)
    assert graph.outgoing_arcs(30) == ()
    assert graph.is_connected(10, 20)
    assert graph.is_connected(20, 30)
    assert not graph.is_connected(10, 30)
    assert graph.arc_endpoints(0) == (10, 20)
    assert graph.arc_endpoints(1) == (20, 30)


def test_branching_node_returns_each_original_outgoing_arc() -> None:
    graph = DirectedGraph(
        ["junction", "north", "south", "west"],
        [("junction", "north"), ("west", "junction"), ("junction", "south")],
    )

    assert graph.outgoing_arcs("junction") == (0, 2)
    assert graph.is_connected("junction", "north")
    assert graph.is_connected("junction", "south")
    assert not graph.is_connected("junction", "west")
    assert graph.arc_endpoints(2) == ("junction", "south")


def test_reverse_direction_is_not_implied() -> None:
    graph = DirectedGraph([1, 2], [(1, 2)])

    assert graph.is_connected(1, 2)
    assert not graph.is_connected(2, 1)
    assert graph.outgoing_arcs(2) == ()


def test_disconnected_nodes_are_not_directly_connected() -> None:
    graph = DirectedGraph([1, 2, 3], [(1, 2)])

    assert not graph.is_connected(1, 3)
    assert not graph.is_connected(3, 1)
    assert graph.outgoing_arcs(3) == ()


def test_parallel_arcs_and_self_loops_keep_distinct_row_ids() -> None:
    graph = DirectedGraph(["a", "b"], [("a", "b"), ("a", "b"), ("b", "b")])

    assert graph.outgoing_arcs("a") == (0, 1)
    assert graph.outgoing_arcs("b") == (2,)
    assert graph.arc_endpoints(0) == ("a", "b")
    assert graph.arc_endpoints(1) == ("a", "b")
    assert graph.arc_endpoints(2) == ("b", "b")
    assert graph.is_connected("b", "b")


def test_edgeless_graph_retains_isolated_nodes() -> None:
    graph = DirectedGraph(["left", "right"], [])

    assert graph.outgoing_arcs("left") == ()
    assert graph.outgoing_arcs("right") == ()
    assert not graph.is_connected("left", "right")


def test_empty_graph_is_supported() -> None:
    graph = DirectedGraph([], [])

    with pytest.raises(KeyError, match="unknown node ID"):
        graph.outgoing_arcs(1)
    with pytest.raises(IndexError, match="unknown arc ID"):
        graph.arc_endpoints(0)


@pytest.mark.parametrize("arc_rows", [[(1, 3)], [(3, 1)]])
def test_unknown_arc_endpoint_is_rejected(arc_rows: list[tuple[int, int]]) -> None:
    with pytest.raises(ValueError, match="unknown node ID"):
        DirectedGraph([1, 2], arc_rows)


def test_duplicate_node_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate node ID"):
        DirectedGraph([1, 1], [])


@pytest.mark.parametrize("row", [(), (1,), (1, 2, 3), "12"])
def test_malformed_arc_rows_are_rejected(row: object) -> None:
    with pytest.raises(ValueError, match="exactly two node IDs"):
        DirectedGraph([1, 2, 3], [row])  # type: ignore[list-item]


def test_queries_reject_unknown_nodes_and_arc_ids() -> None:
    graph = DirectedGraph([1, 2], [(1, 2)])

    with pytest.raises(KeyError, match="unknown node ID"):
        graph.is_connected(1, 99)
    with pytest.raises(KeyError, match="unknown node ID"):
        graph.outgoing_arcs(99)
    with pytest.raises(IndexError, match="unknown arc ID"):
        graph.arc_endpoints(1)
    with pytest.raises(IndexError, match="unknown arc ID"):
        graph.arc_endpoints(-1)
    with pytest.raises(TypeError, match="integer row index"):
        graph.arc_endpoints("0")  # type: ignore[arg-type]


def test_unhashable_node_ids_are_rejected() -> None:
    with pytest.raises(TypeError, match="node ID must be hashable"):
        DirectedGraph([[1]], [])  # type: ignore[list-item]


def test_unhashable_arc_endpoints_are_rejected() -> None:
    with pytest.raises(TypeError, match="unhashable node ID"):
        DirectedGraph([1, 2], [([1], 2)])  # type: ignore[list-item]

