# Graph Type Ablation: Mobility vs Spatial KNN

This note summarizes a topology-only ablation for the mobility GNN. The baseline
uses dynamic OD mobility adjacency. The ablation keeps the mobility GNN enabled
but replaces the graph with a static municipality centroid KNN graph using
`model.max_neighbors` as `k`.

Run:

```bash
uv run python dataviz/adjacency_graph_comparison.py \
  --config configs/train_epifor_real_local.yaml \
  --output-dir outputs/reports/adjacency_graph_comparison \
  --max-hops 2 \
  --neighbor-ceiling 20 \
  --top-ego-maps 6 \
  --max-overlay-edges 2500
```

The comparison used the midpoint real-data date, `2020-10-04`, from the available
range `2020-03-01` to `2021-05-09`. Mobility edges used positive-flow topology
(`mobility > 0`). Receptive fields were limited to two hops with at most 20 newly
added ego nodes per hop, matching the local config's `model.max_neighbors=20`
and `model.gnn_depth=2`.

## Graph-Level Summary

| Graph | Edges | Density | Mean degree | Median degree |
|---|---:|---:|---:|---:|
| Mobility directed | 86,349 | 0.1466 | 112.4 | 114.0 |
| Mobility undirected | 46,665 | 0.1584 | 121.5 | 120.0 |
| Spatial KNN | 8,919 | 0.0303 | 23.2 | 23.0 |
| Undirected overlap | 8,882 | 0.0302 | - | - |

The spatial KNN graph is much sparser globally. Nearly every KNN edge is present
in the undirected mobility skeleton, but mobility contains many extra long-range
or nonlocal edges, giving an edge Jaccard of `0.190`.

## Receptive Field Summary

With the capped 2-hop ego expansion, graph capacity is held almost constant:
mobility ego size is exactly `41` for every municipality, while spatial KNN
averages `40.95`. The difference is composition. Mean ego overlap is `22.4`
nodes, leaving about `18-19` swapped nodes per municipality. Mean ego Jaccard is
`0.387`; median is `0.367`.

The most divergent ego neighborhoods are:

| Region | Ego Jaccard |
|---|---:|
| Ulldemolins | 0.108 |
| Gualba | 0.123 |
| Jafre | 0.123 |
| Sant Guim de Freixenet | 0.123 |
| Riba-roja d'Ebre | 0.123 |
| Vila-rodona | 0.123 |

## Interpretation

This ablation is well matched for model capacity: the GNN, depth, and capped ego
budget stay fixed. It primarily swaps which municipalities provide context. The
KNN graph tests whether local geographic proximity is sufficient, while mobility
keeps a denser set of empirical OD links that can include nonlocal couplings.

Figures and CSVs are in `outputs/reports/adjacency_graph_comparison/`.
