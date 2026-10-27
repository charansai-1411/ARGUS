"""Graph construction and GNN forward/backward on the synthetic fixture (skipped without PyG)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from argus.data.ibm import attach_rings, load_accounts, load_patterns, load_transactions  # noqa: E402
from argus.graph.build import REV_TRANSFER, SPLITS, TRANSFER, build_hetero_graph, phase_graph  # noqa: E402
from argus.graph.model import CONVS, EdgeGNN  # noqa: E402


@pytest.fixture
def graph(raw_dir):
    tx = attach_rings(load_transactions(raw_dir, "T"), load_patterns(raw_dir, "T"))
    return build_hetero_graph(tx, load_accounts(raw_dir, "T"))


def test_graph_structure(graph):
    assert set(graph.node_types) == {"account", "entity", "bank"}
    n = graph[TRANSFER].num_edges
    assert n == 404 and graph[TRANSFER].edge_attr.shape[0] == n
    assert torch.equal(graph[REV_TRANSFER].edge_index, graph[TRANSFER].edge_index.flip(0))
    assert graph[TRANSFER].y.sum() == 4 and (graph[TRANSFER].ring_id == 0).sum() == 4
    assert graph["entity"].num_nodes < graph["account"].num_nodes  # accounts share owners
    assert torch.isfinite(graph[TRANSFER].edge_attr).all()


def test_phase_graph_hides_future(graph):
    split = graph[TRANSFER].split
    for phase, v in SPLITS.items():
        g = phase_graph(graph, phase)
        assert g[TRANSFER].num_edges == int((split <= v).sum())
        assert g[REV_TRANSFER].num_edges == g[TRANSFER].num_edges
    assert graph[TRANSFER].num_edges == 404  # original untouched


@pytest.mark.parametrize("temporal", [True, False])
def test_temporal_sampling_has_no_lookahead(graph, temporal):
    from argus.graph.sampling import make_loader
    gc = {"num_neighbors": {et[1]: [20, 20] for et in graph.edge_types}, "temporal": temporal}
    ids = torch.arange(graph[TRANSFER].num_edges)
    later = 0
    for b in make_loader(graph, ids, gc, False, 64):
        seed_t = graph[TRANSFER].time[b[TRANSFER].edge_label]
        for et in (TRANSFER, REV_TRANSFER):
            dst = b[et].edge_index[1]
            seed = b["account"].batch[dst] if temporal else None
            if temporal:
                later += int((b[et].time > seed_t[seed]).sum())
            else:
                later += int((b[et].time > seed_t.min()).sum())
    if temporal:
        assert later == 0
    else:
        assert later > 0  # sanity: without temporal sampling the future is visible


@pytest.mark.parametrize("conv", CONVS)
def test_model_forward_backward(graph, conv):
    model = EdgeGNN(graph.metadata(), {nt: graph[nt].x.size(1) for nt in graph.node_types},
                    graph[TRANSFER].edge_attr.size(1), hidden=16, layers=2, conv=conv)
    ei = graph[TRANSFER].edge_index[:, :32]
    logits = model(graph.x_dict, graph.edge_index_dict,
                   {et: graph[et].edge_attr for et in (TRANSFER, REV_TRANSFER)},
                   ei, graph[TRANSFER].edge_attr[:32])
    assert logits.shape == (32,)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, graph[TRANSFER].y[:32]).backward()
