import networkx as nx
import matplotlib.pyplot as plt
import itertools
import ujson


v2_lp_data = {}
for filename in [
    r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\arbitrum_lps_sushiswapv2.json",
]:
    with open(filename) as file:
        for pool in ujson.load(file)[0:20]:
            v2_lp_data[pool.get("pool_address")] = {
                key: value
                for key, value in pool.items()
                if key not in ["pool_id"]
            }
print(f"Found {len(v2_lp_data)} V2 pools")

v3_lp_data = {}
for filename in [
    "arbitrum_lps_sushiswapv3.json",
]:
    with open(filename) as file:
        pool_data = ujson.load(file)
    pool_data.pop()  # drop the last entry, which is metadata
    for pool in pool_data[0:20]:
        v3_lp_data[pool.get("pool_address")] = pool
print(f"Found {len(v3_lp_data)} V3 pools")

curve_v1_lp_data = {}
for filename in [
    "arbitrum_lps_curvev1_registry.json"
]:
    with open(filename) as file:
        pool_data = ujson.load(file)
    for pool in pool_data[0:20]:
        curve_v1_lp_data[pool["pool_address"]] = pool
print(f"Found {len(curve_v1_lp_data)} Curve V1 pools")

all_curve_v1_pools = set(curve_v1_lp_data.keys())
all_uni_v2_pools = set(v2_lp_data.keys())
all_uni_v3_pools = set(v3_lp_data.keys())

# build the graph with tokens as nodes, adding an edge
# between any two tokens held by a liquidity pool
G = nx.MultiGraph()
for pool in v2_lp_data.values():
    G.add_edge(
        pool["token0"],
        pool["token1"],
        lp_address=pool["pool_address"],
        pool_type="UniswapV2",
    )

for pool in v3_lp_data.values():
    G.add_edge(
        pool["token0"],
        pool["token1"],
        lp_address=pool["pool_address"],
        pool_type="UniswapV3",
    )

for pool in curve_v1_lp_data.values():
    if pool.get("underlying_coin_addresses"):
        # Add all the underlying tokens (basepool tokens + non-LP token in metapool)
        for token_address_pair in itertools.combinations(
            pool["underlying_coin_addresses"], 2
        ):
            G.add_edge(
                *token_address_pair,
                lp_address=pool["pool_address"],
                pool_type="CurveV1",
            )

    for token_address_pair in itertools.combinations(
        pool["coin_addresses"], 2
    ):
        G.add_edge(
            *token_address_pair,
            lp_address=pool["pool_address"],
            pool_type="CurveV1",
        )


print("Finished processing all 1inch tokens and connecting them to pools.")

# Layout für die Visualisierung
pos = nx.spring_layout(G, k=0.2)

# Größe der Abbildung erhöhen
plt.figure(figsize=(16, 16))

# Knoten zeichnen
nx.draw_networkx_nodes(G, pos, node_size=90)

# Kanten zeichnen
nx.draw_networkx_edges(G, pos)


# Grafik anzeigen
plt.show()
