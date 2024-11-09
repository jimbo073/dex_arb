import ujson
import web3
import networkx as nx
import itertools
import time

# Timer to measure the execution time
start_timer = time.perf_counter()

# List of blacklisted tokens
BLACKLISTED_TOKENS = []

# Function to create a readable path ID from token addresses
def create_path_id(tokens):
    path_id = f"{tokens[0]}/{tokens[1]} -> {tokens[1]}/{tokens[2]} -> {tokens[2]}/{tokens[0]}"
    return path_id

# Load pool data from JSON files
def load_pools(filenames):
    pool_data = {}
    for filename in filenames:
        with open(filename) as file:
            for pool in ujson.load(file):
                pool_data[pool.get("pool_address")] = {key: value for key, value in pool.items() if key not in ["pool_id"]}
    return pool_data

# Load pool data for various DEXes (Uniswap, Sushiswap, Camelot, Curve)
uni_v2_lp_data = load_pools([r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_uniswapv2.json"])
sushi_v2_lp_data = load_pools([r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_sushiswapv2.json"])
camelot_v2_lp_data = load_pools([r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_camelotv2.json"])
sushi_v3_lp_data = load_pools([r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_sushiswapv3.json"])
v3_lp_data = load_pools([r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_uniswapv3.json"])
curve_v1_lp_data = load_pools([r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_curvev1_registry.json"])

# Create sets of unique token addresses from each DEX
tokens_uniswap_v2 = set(token for pool in uni_v2_lp_data.values() for token in [pool["token0"], pool["token1"]])
tokens_sushiswap_v2 = set(token for pool in sushi_v2_lp_data.values() for token in [pool["token0"], pool["token1"]])
tokens_camelot_v2 = set(token for pool in camelot_v2_lp_data.values() for token in [pool["token0"], pool["token1"]])
tokens_uniswap_v3 = set(token for pool in v3_lp_data.values() for token in [pool["token0"], pool["token1"]])
tokens_sushiswap_v3 = set(token for pool in sushi_v3_lp_data.values() for token in [pool["token0"], pool["token1"]])

# For Curve pools, handle both `underlying_coin_addresses` and `coin_addresses`
tokens_curve_v1 = set()
for pool in curve_v1_lp_data.values():
    if pool.get("underlying_coin_addresses"):
        tokens_curve_v1.update(pool["underlying_coin_addresses"])
    if pool.get("coin_addresses"):
        tokens_curve_v1.update(pool["coin_addresses"])

# Combine all unique tokens into one set
all_tokens = tokens_uniswap_v2.union(
    tokens_sushiswap_v2,
    tokens_camelot_v2,
    tokens_uniswap_v3,
    tokens_sushiswap_v3,
    tokens_curve_v1
)

# Convert the set to a list for use in the PROFIT_TOKENS list
PROFIT_TOKENS = [
    # "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9",
    # "0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8",
    # "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
    "0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f",
    "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
    # "0xFa7F8980b0f1E64A2062791cc3b0871572f1F7f0",
    # "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1",
    # "0x912CE59144191C1204E64559FE8253a0e49E6548",
    # "0x6314C31A7a1652cE482cffe247E9CB7c3f4BB9aF",
    # "0x11cDb42B0EB46D95f990BeDD4695A6e3fA034978",
    # "0xd4d42F0b6DEF4CE0383636770eF773390d85c61A",
    # "0xFEa7a6a0B346362BF88A9e4A88416B77a57D6c2A",
    ] # list(all_tokens)

print(f"Total unique tokens available across all DEXes: {len(PROFIT_TOKENS)}")

# Create a set of all pool addresses
all_curve_v1_pools = set(curve_v1_lp_data.keys())
all_sushi_v3_pools = set(sushi_v3_lp_data.keys())
all_sushi_v2_pools = set(sushi_v2_lp_data.keys())
all_camelot_v2_pools = set(camelot_v2_lp_data.keys())
all_uni_v3_pools = set(v3_lp_data.keys())
all_uni_v2_pools = set(uni_v2_lp_data.keys())

# Create a graph with token nodes and liquidity pool edges
G = nx.MultiGraph()

# Add edges for the various DEXes
def add_edges(pool_data, pool_type):
    for pool in pool_data.values():
        G.add_edge(pool["token0"], pool["token1"], lp_address=pool["pool_address"], pool_type=pool_type)

# Add edges for DEX pools
add_edges(uni_v2_lp_data, "UniswapV2")
add_edges(sushi_v2_lp_data, "SushiswapV2")
add_edges(camelot_v2_lp_data, "CamelotV2")
add_edges(v3_lp_data, "UniswapV3")
add_edges(sushi_v3_lp_data, "SushiswapV3")

# Add edges for Curve pools (underlying coins and coin addresses)
for pool in curve_v1_lp_data.values():
    if pool.get("underlying_coin_addresses"):
        for token_address_pair in itertools.combinations(pool["underlying_coin_addresses"], 2):
            G.add_edge(*token_address_pair, lp_address=pool["pool_address"], pool_type="CurveV1")
    if pool.get("coin_addresses"):
        for token_address_pair in itertools.combinations(pool["coin_addresses"], 2):
            G.add_edge(*token_address_pair, lp_address=pool["pool_address"], pool_type="CurveV1")

# Remove blacklisted tokens from the graph
G.remove_nodes_from(BLACKLISTED_TOKENS)
print(f"Graph created: {len(G.nodes)} tokens, {len(G.edges)} edges")

# Process the paths
triangle_arb_paths = {}

# Generate arbitrage paths without 1inch
for token in PROFIT_TOKENS:
    token_address = token
    if token_address not in G:
        print(f"Token {token_address} not found in the graph.")
        continue

    # Counter for the number of paths per token
    path_count = 0

    # Find all tokens that share a pool with the profit token
    all_tokens_with_pair_pool = list(G.neighbors(token_address))
    filtered_tokens = [t for t in all_tokens_with_pair_pool if G.degree(t) > 1]

    # Create arbitrage paths without the 1inch aggregator
    for token_a, token_b in itertools.combinations(filtered_tokens, 2):
        if not G.get_edge_data(token_a, token_b):
            continue

        # Inside pools (Curve pools in the middle of the arbitrage path)
        inside_pools = [
            edge["lp_address"]
            for edge in G.get_edge_data(token_a, token_b).values()
            if edge["pool_type"] == "CurveV1"
        ]

        # Outside pools for Token A (e.g., Uniswap, Sushiswap, Camelot)
        outside_pools_tokenA = [
            edge["lp_address"]
            for edge in G.get_edge_data(token_a, token_address).values()
            if edge["pool_type"] in ["UniswapV2", "UniswapV3", "SushiswapV2", "SushiswapV3", "CamelotV2"]
        ]

        # Outside pools for Token B (last swap to profit token)
        last_swap_pools = [
            edge["lp_address"]
            for edge in G.get_edge_data(token_b, token_address).values()
            if edge["pool_type"] in ["UniswapV2", "UniswapV3", "SushiswapV2", "SushiswapV3", "CamelotV2"]
        ]

        # Only process if last_swap_pools contains at least 3 token addresses
        if len(last_swap_pools) < 3:
            continue

        # Combine outside_pools_tokenA and inside_pools for the arbitrage paths
        for pool_addresses in itertools.product(outside_pools_tokenA, inside_pools):
            pool_data = {}
            tokens = [token_a, token_b, token_address]

            # Record pool information
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

            # Create the path ID and save the list of all last swap pools
                triangle_arb_paths[id] = {
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
                "path": list(pool_addresses), #+ [last_swap_pools],  # The list of last swap pools is added here
                "pools": pool_data,
                "token_flow": create_path_id(tokens),
                "last_swap_pools": last_swap_pools  # List of last pools for swapping Token C to profit token
            }

# Print the total number of arbitrage paths found after processing all tokens
print(f"Total arbitrage paths found: {len(triangle_arb_paths)}")

# Save the results
with open("arbitrum_arbs_3pool_type.json", "w") as file:
    ujson.dump(triangle_arb_paths, file, indent=2)

print(f"°Paths saved. Finished in {time.perf_counter() - start_timer:.1f}s")
