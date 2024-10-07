import ujson
import web3
import networkx as nx
import itertools
import time

start_timer = time.perf_counter()

BLACKLISTED_TOKENS = []
AGGREGATOR_ADDRESS = "0x111111125421cA6dc452d289314280a0f8842A65"
# Funktion zur Erstellung der Path ID als lesbare Zeichenkette
def create_path_id(tokens):
    # Generiere den Pfad als Zeichenkette
    path_id = f"{tokens[0]}/{tokens[1]} -> {tokens[1]}/{tokens[2]} -> {tokens[2]}/{tokens[0]}"
    return path_id

# Laden der Pool-Daten
def load_pools(filenames):
    pool_data = {}
    for filename in filenames:
        with open(filename) as file:
            for pool in ujson.load(file):
                pool_data[pool.get("pool_address")] = {key: value for key, value in pool.items() if key not in ["pool_id"]}
    return pool_data

# Pool-Daten laden
uni_v2_lp_data = load_pools([
    r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_uniswapv2.json"
])
print(f"Found {len(uni_v2_lp_data)} Uniswap V2 pools")

sushi_v2_lp_data = load_pools([
    r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_sushiswapv2.json"
])
print(f"Found {len(sushi_v2_lp_data)} Sushiswap V2 pools")

camelot_v2_lp_data = load_pools([
    r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_camelotv2.json"
])
print(f"Found {len(camelot_v2_lp_data)} Camelot V2 pools")

sushi_v3_lp_data = load_pools([
    r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_sushiswapv3.json"
])
print(f"Found {len(sushi_v3_lp_data)} Sushiswap V3 pools")

v3_lp_data = load_pools([
    r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_uniswapv3.json"
])
print(f"Found {len(v3_lp_data)} Uniswap V3 pools")

curve_v1_lp_data = load_pools([
    r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_curvev1_registry.json"
])
print(f"Found {len(curve_v1_lp_data)} Curve V1 pools")

# Load 1inch tokens data
with open(r"C:\Users\PC\Projects\dex_arb\arbitrum\1inch_code\1inch_tokens.json") as file:
    oneinch_token_data = {token["token"]: token for token in ujson.load(file)}

# Set von allen Pooladressen erstellen
all_curve_v1_pools = set(curve_v1_lp_data.keys())
all_sushi_v3_pools = set(sushi_v3_lp_data.keys())
all_sushi_v2_pools = set(sushi_v2_lp_data.keys())
all_camelot_v2_pools = set(camelot_v2_lp_data.keys())
all_uni_v3_pools = set(v3_lp_data.keys())
all_uni_v2_pools = set(uni_v2_lp_data.keys())
all_1inch_tokens = set(oneinch_token_data.keys())

# Erstellen des Graphen mit den Token-Knoten und den Kanten der Liquidity Pools
G = nx.MultiGraph()

# Hinzufügen von Kanten für die verschiedenen DEXes
def add_edges(pool_data, pool_type):
    for pool in pool_data.values():
        G.add_edge(pool["token0"], pool["token1"], lp_address=pool["pool_address"], pool_type=pool_type)

add_edges(uni_v2_lp_data, "UniswapV2")
add_edges(sushi_v2_lp_data, "SushiswapV2")
add_edges(camelot_v2_lp_data, "CamelotV2")
add_edges(v3_lp_data, "UniswapV3")
add_edges(sushi_v3_lp_data, "SushiswapV3")

# Spezielles Hinzufügen von Kanten für Curve-Pools
for pool in curve_v1_lp_data.values():
    if pool.get("underlying_coin_addresses"):
        for token_address_pair in itertools.combinations(pool["underlying_coin_addresses"], 2):
            G.add_edge(*token_address_pair, lp_address=pool["pool_address"], pool_type="CurveV1")
    for token_address_pair in itertools.combinations(pool["coin_addresses"], 2):
        G.add_edge(*token_address_pair, lp_address=pool["pool_address"], pool_type="CurveV1")

# Entfernen von blacklisted Tokens aus dem Graphen
G.remove_nodes_from(BLACKLISTED_TOKENS)
print(f"G ready: {len(G.nodes)} nodes, {len(G.edges)} edges")

# Verarbeitung der Pfade
triangle_arb_paths = {"type1": {}, "type2": {}}

# Pfadtyp 1: 1inch an einer Stelle, Curve in der Mitte
for token in oneinch_token_data.values():
    token_address = token.get("token")
    try:
        if token_address in G:
            all_tokens_with_pair_pool = list(G.neighbors(token_address))
        else:
            print(f"The node {token_address} is not in the graph.")
            continue
    except Exception as e:
        print(f"No neighbors found for {token_address} with reason {e}")
        continue

    filtered_tokens = [t for t in all_tokens_with_pair_pool if G.degree(t) > 1]

    for token_a, token_b in itertools.combinations(filtered_tokens, 2):
        if not G.get_edge_data(token_a, token_b):
            continue

        inside_pools = [
            edge["lp_address"]
            for edge in G.get_edge_data(token_a, token_b).values()
            if edge["pool_type"] == "CurveV1"
        ]
        
        for pool in inside_pools.copy():
            pool_data = curve_v1_lp_data[pool]
            if curve_v1_lp_data[pool].get(
                "underlying_coin_addresses"
            ) is not None and all(
                [
                    token_a in pool_data["underlying_coin_addresses"],
                    token_b in pool_data["underlying_coin_addresses"],
                ]
            ):
                inside_pools.pop(inside_pools.index(pool))

        if not inside_pools:
            continue

        # Möglichkeit 1: 1inch am Anfang, Curve in der Mitte, Uniswap/Sushiswap/Camelot am Ende
        outside_pools_tokenA = [
            edge["lp_address"]
            for edge in G.get_edge_data(token_a, token_address).values()
            if edge["pool_type"] in ["UniswapV2", "UniswapV3", "SushiswapV3","SushiswapV2", "CamelotV2", "1inch_Aggregator"]
        ]

        # Möglichkeit 2: Uniswap/Sushiswap/Camelot am Anfang, Curve in der Mitte, 1inch am Ende
        outside_pools_tokenB = [AGGREGATOR_ADDRESS]

        # Generiere die Pfade für Möglichkeit 1
        # find all triangular arbitrage paths of form:
        # WETH/tokenA -> tokenA/tokenB -> tokenB/WETH
        for pool_addresses in itertools.product(outside_pools_tokenA, inside_pools, outside_pools_tokenB):
                path_type = "type1"
                pool_data = {}
                tokens = [token_a, token_b, token_address]
                if None in tokens:
                    print(f"Error: None found in tokens {tokens} for path addresses {pool_addresses}")
                    raise ValueError("Token value is None")

                # Erfasse die Poolinformationen
                for pool_address in pool_addresses:
                    if pool_address in all_sushi_v2_pools:
                        pool_info = sushi_v2_lp_data.get(pool_address)
                    elif pool_address in all_uni_v2_pools:
                        pool_info = uni_v2_lp_data.get(pool_address)
                    elif pool_address in all_uni_v3_pools:
                        pool_info = v3_lp_data.get(pool_address)
                    elif pool_address in all_sushi_v3_pools:
                        pool_info = sushi_v3_lp_data.get(pool_address)
                    elif pool_address in all_camelot_v2_pools:
                        pool_info = camelot_v2_lp_data.get(pool_address)
                    elif pool_address in all_curve_v1_pools:
                        pool_info = curve_v1_lp_data.get(pool_address)
                    elif pool_address == AGGREGATOR_ADDRESS:
                        pool_info = {"@1inch_Aggregator": token_address} 
                    else:
                        raise Exception(f"No pool data found for {pool_address}")
                    pool_data[pool_address] = pool_info
                        
                # Erstelle die Path ID
                triangle_arb_paths[path_type][id] = {
                "id": (
                    id := web3.Web3.keccak(
                        hexstr="".join(
                            [
                                pool_address[2:]
                                for pool_address in pool_addresses
                            ]
                        )
                    ).hex()
                ),
                "path": pool_addresses,
                "pools": pool_data,
                "token_flow": create_path_id(tokens),
            }

        # Generiere die Pfade für Möglichkeit 2
        for pool_addresses in itertools.product(outside_pools_tokenB, inside_pools, outside_pools_tokenA):
            path_type = "type1"
            pool_data = {}

            tokens = [token_a, token_b, token_address]
            if None in tokens:
                print(f"Error: None found in tokens {tokens} for path addresses {pool_addresses}")
                raise ValueError("Token value is None")

            # Erfasse die Poolinformationen
            for pool_address in pool_addresses:
                if pool_address in all_sushi_v2_pools:
                    pool_info = sushi_v2_lp_data.get(pool_address)
                elif pool_address in all_uni_v2_pools:
                    pool_info = uni_v2_lp_data.get(pool_address)
                elif pool_address in all_uni_v3_pools:
                    pool_info = v3_lp_data.get(pool_address)
                elif pool_address in all_sushi_v3_pools:
                    pool_info = sushi_v3_lp_data.get(pool_address)
                elif pool_address in all_camelot_v2_pools:
                    pool_info = camelot_v2_lp_data.get(pool_address)
                elif pool_address in all_curve_v1_pools:
                    pool_info = curve_v1_lp_data.get(pool_address)
                elif pool_address in all_curve_v1_pools:
                    pool_info = curve_v1_lp_data.get(pool_address)
                elif pool_address == AGGREGATOR_ADDRESS:
                    pool_info = {"@1inch_Aggregator": token_address} 
                else:
                    raise Exception(f"No pool data found for {pool_address}")
                pool_data[pool_address] = pool_info

            # Erstelle die Path ID
            triangle_arb_paths[path_type][id] = {
            "id": (
                id := web3.Web3.keccak(
                    hexstr="".join(
                        [
                            pool_address[2:]
                            for pool_address in pool_addresses
                        ]
                    )
                ).hex()
            ),
            "path":  pool_addresses,
            "pools": pool_data,
            "token_flow": create_path_id(tokens),
        }

# Pfadtyp 2: Ohne 1inch, nur Uniswap/Sushiswap/Camelot und Curve
for token in oneinch_token_data.values():
    token_address = token.get("token")
    try:
        if token_address in G:
            all_tokens_with_pair_pool = list(G.neighbors(token_address))
        else:
            print(f"The node {token_address} is not in the graph.")
            continue
    except Exception as e:
        print(f"No neighbors found for {token_address} with reason {e}")
        continue
    filtered_tokens = [t for t in all_tokens_with_pair_pool if G.degree(t) > 1]

    for token_a, token_b in itertools.combinations(filtered_tokens, 2):
        if not G.get_edge_data(token_a, token_b):
            continue

        inside_pools = [
            edge["lp_address"]
            for edge in G.get_edge_data(token_a, token_b).values()
            if edge["pool_type"] == "CurveV1"
        ]

        if not inside_pools:
            continue

        outside_pools_tokenA = [
            edge["lp_address"]
            for edge in G.get_edge_data(token_a, token_address).values()
            if edge["pool_type"] in ["UniswapV2", "UniswapV3","SushiswapV2", "SushiswapV3", "CamelotV2"]
        ]

        outside_pools_tokenB = [
            edge["lp_address"]
            for edge in G.get_edge_data(token_b, token_address).values()
            if edge["pool_type"] in ["UniswapV2", "UniswapV3","SushiswapV2", "SushiswapV3", "CamelotV2"]
        ]

        # Erzeuge die Pfade für Typ 2
        for pool_addresses in itertools.product(outside_pools_tokenA, inside_pools, outside_pools_tokenB):
            path_type = "type2"
            pool_data = {}
            tokens = [token_a, token_b, token_address]

            if None in tokens:
                print(f"Error: None found in tokens {tokens} for path addresses {pool_addresses}")
                raise ValueError("Token value is None")

            # Erfasse die Poolinformationen
            for pool_address in pool_addresses:
                if pool_address in all_sushi_v2_pools:
                    pool_info = sushi_v2_lp_data.get(pool_address)
                elif pool_address in all_uni_v2_pools:
                    pool_info = uni_v2_lp_data.get(pool_address)
                elif pool_address in all_uni_v3_pools:
                    pool_info = v3_lp_data.get(pool_address)
                elif pool_address in all_sushi_v3_pools:
                    pool_info = sushi_v3_lp_data.get(pool_address)
                elif pool_address in all_camelot_v2_pools:
                    pool_info = camelot_v2_lp_data.get(pool_address)
                elif pool_address in all_curve_v1_pools:
                    pool_info = curve_v1_lp_data.get(pool_address)
                else:
                    continue
                pool_data[pool_address] = pool_info

            # Erstelle die Path ID
            triangle_arb_paths[path_type][id] = {
            "id": (
                id := web3.Web3.keccak(
                    hexstr="".join(
                        [
                            pool_address[2:]
                            for pool_address in pool_addresses
                        ]
                    )
                ).hex()
            ),
            "path": pool_addresses,
            "pools": pool_data,
            "token_flow": create_path_id(tokens),
        }

print(f"Found {len(triangle_arb_paths['type1'])} type 1 paths and {len(triangle_arb_paths['type2'])} type 2 paths in {time.perf_counter() - start_timer:.1f}s")

# Separate the paths by type
type1_paths = triangle_arb_paths['type1']
type2_paths = triangle_arb_paths['type2']

# Save type1 paths to a JSON file
with open(r"C:\Users\PC\Projects\dex_arb\arbitrum\arbitrage_paths\arbitrum_arbs_3pool_type1.json", "w") as file_type1:
    ujson.dump(type1_paths, file_type1, indent=2)

# Save type2 paths to a JSON file
with open(r"C:\Users\PC\Projects\dex_arb\arbitrum\arbitrage_paths\arbitrum_arbs_3pool_type2.json", "w") as file_type2:
    ujson.dump(type2_paths, file_type2, indent=2)

print(f"Type 1 paths saved to arbitrum_arbs_3pool_type1.json")
print(f"Type 2 paths saved to arbitrum_arbs_3pool_type2.json")
