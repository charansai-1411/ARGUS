"""Mini-batch neighbour sampling around the transfers being scored."""

from torch_geometric.loader import LinkNeighborLoader

from argus.graph.build import TRANSFER


def make_loader(graph, edge_ids, gc: dict, shuffle: bool, batch_size: int) -> LinkNeighborLoader:
    """Loader yielding a sampled subgraph around each batch of transfers `edge_ids`.

    With gc["temporal"] (default on), a scored transfer only sees transfers with
    time <= its own, so there is no look-ahead even on the full test-period graph.
    PyG makes temporal sampling disjoint: one subgraph per scored transfer.
    """
    nn_cfg = {et: gc["num_neighbors"][et[1]] for et in graph.edge_types}
    temporal = {}
    if gc.get("temporal", True):
        temporal = dict(time_attr="time", edge_label_time=graph[TRANSFER].time[edge_ids],
                        temporal_strategy=gc.get("temporal_strategy", "last"))
    return LinkNeighborLoader(
        graph,
        num_neighbors=nn_cfg,
        edge_label_index=(TRANSFER, graph[TRANSFER].edge_index[:, edge_ids]),
        edge_label=edge_ids,            # carry global edge ids through the sampler
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=gc.get("num_workers", 0),
        **temporal,
    )
