# Box 8. Visualize the top-ranked inferred GRN
# This version keeps the original spring-layout style, but slightly separates TF nodes.

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

edge_file = "./output/PBMC3k/Top1000_inferredGRN_PBMC3k.csv"
out_file = "./output/PBMC3k/top300_ranked_inferred_GRN.pdf"

top_n_edges = 300
random_seed = 42

edges = pd.read_csv(edge_file)

# Adapt column names if necessary
rename_map = {}
for col in edges.columns:
    low = col.lower()
    if low in ["gene1", "source", "tf", "transcription_factor"]:
        rename_map[col] = "TF"
    elif low in ["gene2", "target", "target_gene"]:
        rename_map[col] = "target"
    elif low in ["score", "edgeweight", "edge_weight", "weight"]:
        rename_map[col] = "score"

edges = edges.rename(columns=rename_map)

required_cols = {"TF", "target", "score"}
if not required_cols.issubset(edges.columns):
    raise ValueError(
        f"The edge file must contain columns {required_cols}. "
        f"Current columns are: {list(edges.columns)}"
    )

# Rank by absolute regulatory score
edges["abs_score"] = edges["score"].abs()
edges = edges.sort_values("abs_score", ascending=False).head(top_n_edges)

# Build directed graph
G = nx.DiGraph()
for _, row in edges.iterrows():
    G.add_edge(row["TF"], row["target"], weight=row["abs_score"])

tf_nodes = sorted(edges["TF"].unique())
tf_nodes = [n for n in tf_nodes if n in G.nodes]
target_nodes = [n for n in G.nodes if n not in tf_nodes]

# Original force-directed layout
pos = nx.spring_layout(
    G,
    seed=random_seed,
    k=0.7,              # increase slightly if the whole network is still too compact
    iterations=300,
    weight="weight"
)

# ---------------------------------------------------------------------
# Slightly separate TF nodes while preserving the original spring layout
# ---------------------------------------------------------------------
def separate_tf_nodes(pos, tf_nodes, min_dist=0.16, n_iter=200, anchor_strength=0.15):
    rng = np.random.default_rng(42)
    original_pos = {n: np.array(pos[n], dtype=float).copy() for n in tf_nodes}

    for _ in range(n_iter):
        disp = {n: np.zeros(2, dtype=float) for n in tf_nodes}

        for i, u in enumerate(tf_nodes):
            for v in tf_nodes[i + 1:]:
                delta = np.array(pos[u]) - np.array(pos[v])
                dist = np.linalg.norm(delta)

                if dist < 1e-6:
                    angle = rng.uniform(0, 2 * np.pi)
                    delta = np.array([np.cos(angle), np.sin(angle)])
                    dist = 1e-6

                if dist < min_dist:
                    direction = delta / dist
                    shift = 0.5 * (min_dist - dist) * direction
                    disp[u] += shift
                    disp[v] -= shift

        for n in tf_nodes:
            pos[n] = (
                np.array(pos[n])
                + disp[n]
                - anchor_strength * (np.array(pos[n]) - original_pos[n])
            )

    return pos

pos = separate_tf_nodes(
    pos,
    tf_nodes=tf_nodes,
    min_dist=0.2,        # increase to 0.20–0.25 if TFs are still too crowded
    n_iter=250,
    anchor_strength=0.07  # increase if the layout changes too much
)

# Node size: larger for high-degree nodes
degree_dict = dict(G.degree())
node_sizes = {
    n: 250 + 40 * degree_dict.get(n, 1)
    for n in G.nodes
}

# Edge width according to score
max_weight = max(nx.get_edge_attributes(G, "weight").values())
edge_widths = [
    0.4 + 2.0 * G[u][v]["weight"] / max_weight
    for u, v in G.edges()
]

plt.figure(figsize=(12, 7))

nx.draw_networkx_edges(
    G,
    pos,
    arrows=True,
    arrowstyle="-|>",
    arrowsize=5,
    width=edge_widths,
    edge_color="#4D4D4D",
    alpha=0.4,
    connectionstyle="arc3,rad=0.04"
)

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=target_nodes,
    node_shape="o",
    node_size=[node_sizes[n] for n in target_nodes],
    node_color="#D9ECF5",
    edgecolors="#BDBDBD",
    linewidths=0.5,
    alpha=0.95
)

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=tf_nodes,
    node_shape="o",
    node_size=[node_sizes[n] for n in tf_nodes],
    node_color="#D9ECF5",
    edgecolors="#BDBDBD",
    linewidths=0.5,
    alpha=0.95
)

# Label TF nodes only
nx.draw_networkx_labels(
    G,
    pos,
    labels={tf: tf for tf in tf_nodes},
    font_size=8,
    font_color="#000000"
)

plt.axis("off")
plt.tight_layout()

os.makedirs(os.path.dirname(out_file), exist_ok=True)
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close()
