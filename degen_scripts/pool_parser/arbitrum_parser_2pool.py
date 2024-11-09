import json
import web3
import networkx as nx
import itertools
import matplotlib.pyplot as plt
import sys

WETH_ADDRESS = "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1"
AGGREGATOR = "0x111111125421cA6dc452d289314280a0f8842A65"
w3 = web3.Web3()

BLACKLISTED_TOKENS = [
#     "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",  # USDC
#     "0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8"
#     "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9",  # USDT
 ]


sushi_v2_lp_data = {}
for filename in [
    r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\arbitrum_lps_sushiswapv2.json",
]:
    with open(filename) as file:
        for pool in json.load(file):
            sushi_v2_lp_data[pool.get("pool_address")] = {
                key: value
                for key, value in pool.items()
                if key not in ["pool_id"]
            }
print(f"Found {len(sushi_v2_lp_data)} V2 pools")

camelot_v2_lp_data = {}
for filename in [
    r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\arbitrum_lps_camelotv2.json",
]:
    with open(filename) as file:
        for pool in json.load(file):
            camelot_v2_lp_data[pool.get("pool_address")] = {
                key: value
                for key, value in pool.items()
                if key not in ["pool_id"]
            }
print(f"Found {len(camelot_v2_lp_data)} V2 pools")

v3_lp_data = {}
for filename in [
    r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\arbitrum_lps_uniswapv3.json",
    r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\arbitrum_lps_sushiswapv3.json"
]:
    with open(filename) as file:
        for pool in json.load(file):
            v3_lp_data[pool.get("pool_address")] = {
                key: value
                for key, value in pool.items()
                if key not in ["block_number"]
            }
print(f"Found {len(v3_lp_data)} V3 pools")

oneinch_token_data = {}
for filename in [
    r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\1inch_tokens.json"
]:
    with open(filename) as file:
        for token in json.load(file):
            oneinch_token_data[token.get("token")] = {
                key: value
                for key, value in token.items()
            }
print(f"Found {len(oneinch_token_data)} 1inch coins")

all_v2_pools = set(sushi_v2_lp_data.keys())
all_v3_pools = set(v3_lp_data.keys())

all_tokens = set(
    [lp.get("token0") for lp in sushi_v2_lp_data.values()]
    + [lp.get("token1") for lp in sushi_v2_lp_data.values()]
    + [lp.get("token0") for lp in v3_lp_data.values()]
    + [lp.get("token1") for lp in v3_lp_data.values()]
)

# build the graph with tokens as nodes, adding an edge
# between any two tokens held by a liquidity pool
G = nx.MultiGraph()

# Process 1inch tokens and connect them to matching pools
for coin in oneinch_token_data.values():
    token = coin.get("token")
    pool_type = coin.get("pool_type")
    # Iterate through all pools to find matches
    for pool in sushi_v2_lp_data.values():
        if token in (pool.get("token0"), pool.get("token1")) and (pool.get("token0") in oneinch_token_data.values() and pool.get("token1") in oneinch_token_data.values()):
            # Add edges between the 1inch token and the pool tokens
            if pool.get("token0") != token:
                G.add_edge(
                    token,
                    pool.get("token0"),
                    lp_address=pool.get("pool_address"),
                    pool_type=pool.get("type")
                )
            if pool.get("token1") != token:
                G.add_edge(
                    token,
                    pool.get("token1"),
                    lp_address=pool.get("pool_address"),
                    pool_type=pool.get("type")
                )
print("Finished processing all 1inch tokens and connecting them to pools.")
        
# Process 1inch tokens and connect them to matching pools
for coin in oneinch_token_data.values():
    token = coin.get("token")
    pool_type = coin.get("pool_type")
    # Iterate through all pools to find matches
    for pool in camelot_v2_lp_data.values():
        if token in (pool.get("token0"), pool.get("token1")) and (pool.get("token0") in oneinch_token_data.values() and pool.get("token1") in oneinch_token_data.values()):
            # Add edges between the 1inch token and the pool tokens
            if pool.get("token0") != token:
                G.add_edge(
                    token,
                    pool.get("token0"),
                    lp_address=pool.get("pool_address"),
                    pool_type=pool.get("type")
                )
            if pool.get("token1") != token:
                G.add_edge(
                    token,
                    pool.get("token1"),
                    lp_address=pool.get("pool_address"),
                    pool_type=pool.get("type")
                )
print("Finished processing all 1inch tokens and connecting them to pools.")

# Process 1inch tokens and connect them to matching pools
for coin in oneinch_token_data.values():
    token = coin.get("token")
    pool_type = coin.get("pool_type")
    # Iterate through all pools to find matches
    for pool in v3_lp_data.values():
        if token in (pool.get("token0"), pool.get("token1")) and (pool.get("token0") in oneinch_token_data.values() and pool.get("token1") in oneinch_token_data.values()):
            # Add edges between the 1inch token and the pool tokens
            if pool.get("token0") != token:
                G.add_edge(
                    token,
                    pool.get("token0"),
                    lp_address=pool.get("pool_address"),
                    pool_type=pool.get("type")
                )
            if pool.get("token1") != token:
                G.add_edge(
                    token,
                    pool.get("token1"),
                    lp_address=pool.get("pool_address"),
                    pool_type=pool.get("type")
                )
print("Finished processing all 1inch tokens and connecting them to pools.")
        
# delete nodes for blacklisted tokens
G.remove_nodes_from(BLACKLISTED_TOKENS)

print(f"G ready: {len(G.nodes)} nodes, {len(G.edges)} edges")
two_pool_arb_paths = {}

# Iterate over each token in oneinch_token_data
for coin in oneinch_token_data.values():
    token_address = coin.get("token")
    # Find all tokens with a pair to the current token
    if token_address in G.nodes:
        all_tokens_with_pair = list(G.neighbors(token_address))
        print(f"Token {token_address} has {len(all_tokens_with_pair)} pool pairs:")
        print(f"*** Finding two-pool arbitrage paths for {token_address} ***")

        # Iterate through all pairs of pools for the token
        for other_token in all_tokens_with_pair:
            pools = G.get_edge_data(other_token, token_address).values()

            for pool_ in pools:
                # Determine the type of pool and retrieve its data
                if pool_.get("pool_type") == "SushiswapV2":
                    pool__dict = sushi_v2_lp_data.get(pool_.get("lp_address"))      
                elif pool_.get("pool_type") == "CamelotV2":
                    pool__dict = camelot_v2_lp_data.get(pool_.get("lp_address"))
                elif pool_.get("pool_type") == "UniswapV3" or pool_.get("pool_type") == "SushiswapV3":
                    pool__dict = v3_lp_data.get(pool_.get("lp_address"))
                elif pool_.get("pool_type") ==  "1inch_Aggregator":
                    pool__dict = oneinch_token_data.get(pool_.get("lp_address"))
                else:
                    raise Exception(f"Could not identify pool {pool_}")

                two_pool_arb_paths[id] = {
                    "id": (
                        id := web3.Web3.keccak(
                            hexstr="".join(
                                [
                                    pool_[2:]
                                    for pool_ in pools
                                ]
                            )
                        ).hex()
                    ),
                    "path": pools,
                    "pools": [AGGREGATOR,pool__dict]
                }

print(f"Found {len(two_pool_arb_paths)} unique two-pool arbitrage paths")

# Save the results to a JSON file
print("• Saving arb paths to JSON")
with open("arbitrum_arbs_2pool.json", "w") as file:
    json.dump(two_pool_arb_paths, file, indent=2)
