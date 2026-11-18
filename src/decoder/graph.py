"""Directed graph primitives for map-matching decoders.

Arc IDs correspond to row positions in the parsed ``.arcs`` input. Pass the
rows in their original order and do not filter or reorder them before graph
construction.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable

NodeId = Hashable
ArcId = int
ArcEndpoints = tuple[NodeId, NodeId]


class DirectedGraph:
    """A directed graph indexed by parsed node IDs and original arc row IDs.

    Args:
        node_ids: Unique, hashable IDs for every node in the graph.
        arc_rows: Directed ``(source_node_id, target_node_id)`` pairs in
            original ``.arcs`` row order. The zero-based row position is the
            arc ID.

    Raises:
        TypeError: If inputs are not iterable or contain unhashable IDs.
        ValueError: If node IDs are duplicated, an arc row is malformed, or
            an arc references a node absent from ``node_ids``.
    """

    def __init__(
        self,
        node_ids: Iterable[NodeId],
        arc_rows: Iterable[Iterable[NodeId]],
    ) -> None:
        self._node_ids: set[NodeId] = set()
        outgoing: dict[NodeId, list[ArcId]] = {}
        neighbors: dict[NodeId, set[NodeId]] = {}

        try:
            parsed_node_ids = iter(node_ids)
        except TypeError as exc:
            raise TypeError("node_ids must be iterable") from exc

        for node_id in parsed_node_ids:
            try:
                hash(node_id)
            except TypeError as exc:
                raise TypeError(f"node ID must be hashable: {node_id!r}") from exc
            if node_id in self._node_ids:
                raise ValueError(f"duplicate node ID: {node_id!r}")
            self._node_ids.add(node_id)
            outgoing[node_id] = []
            neighbors[node_id] = set()

        self._arc_endpoints: list[ArcEndpoints] = []
        try:
            parsed_arc_rows = iter(arc_rows)
        except TypeError as exc:
            raise TypeError("arc_rows must be iterable") from exc

        for arc_id, row in enumerate(parsed_arc_rows):
            if isinstance(row, (str, bytes)):
                raise ValueError(f"arc row {arc_id} must contain exactly two node IDs")
            try:
                source, target = row
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"arc row {arc_id} must contain exactly two node IDs"
                ) from exc

            for endpoint in (source, target):
                try:
                    hash(endpoint)
                except TypeError as exc:
                    raise TypeError(
                        f"arc row {arc_id} contains an unhashable node ID: {endpoint!r}"
                    ) from exc
                if endpoint not in self._node_ids:
                    raise ValueError(
                        f"arc row {arc_id} references unknown node ID {endpoint!r}"
                    )

            self._arc_endpoints.append((source, target))
            outgoing[source].append(arc_id)
            neighbors[source].add(target)

        # Freeze adjacency values so callers cannot mutate the graph through
        # outgoing_arcs(). The node and neighbor indexes remain private.
        self._outgoing_arcs: dict[NodeId, tuple[ArcId, ...]] = {
            node_id: tuple(arc_ids) for node_id, arc_ids in outgoing.items()
        }
        self._outgoing_neighbors = neighbors

    def outgoing_arcs(self, node_id: NodeId) -> tuple[ArcId, ...]:
        """Return original arc IDs leaving ``node_id`` in input row order.

        Raises:
            KeyError: If ``node_id`` is not in this graph.
        """
        try:
            return self._outgoing_arcs[node_id]
        except KeyError:
            raise KeyError(f"unknown node ID: {node_id!r}") from None

    def is_connected(self, source: NodeId, target: NodeId) -> bool:
        """Return whether a directed arc exists directly from source to target.

        This checks one graph edge only; it does not test reachability through
        intermediate nodes.

        Raises:
            KeyError: If either node ID is not in this graph.
        """
        if source not in self._node_ids:
            raise KeyError(f"unknown node ID: {source!r}")
        if target not in self._node_ids:
            raise KeyError(f"unknown node ID: {target!r}")
        return target in self._outgoing_neighbors[source]

    def arc_endpoints(self, arc_id: ArcId) -> ArcEndpoints:
        """Return the source and target node IDs for an original arc row ID.

        Raises:
            TypeError: If ``arc_id`` is not an integer.
            IndexError: If ``arc_id`` is outside the original row range.
        """
        if not isinstance(arc_id, int) or isinstance(arc_id, bool):
            raise TypeError("arc_id must be an integer row index")
        if arc_id < 0 or arc_id >= len(self._arc_endpoints):
            raise IndexError(f"unknown arc ID: {arc_id}")
        try:
            return self._arc_endpoints[arc_id]
        except IndexError:
            raise IndexError(f"unknown arc ID: {arc_id}") from None

