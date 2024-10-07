import json
import web3
import networkx as nx

# Web3 initialization
w3 = web3.Web3()

# Blacklisted tokens
BLACKLISTED_TOKENS = [
    # Add tokens if needed
]

# Load pools data from files
def load_pools(file_path):
    with open(file_path) as file:
        return {
            pool.get("pool_address"): {
                key: value for key, value in pool.items() if key not in ["pool_id", "block_number"]
            } for pool in json.load(file)
        }
        
uni_v2_lp_data = load_pools(r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_uniswapv2.json")
sushi_v2_lp_data = load_pools(r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_sushiswapv2.json")
camelot_v2_lp_data = load_pools(r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_camelotv2.json")
v3_lp_data = load_pools(r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_uniswapv3.json")

# Load 1inch tokens data
with open(r"C:\Users\PC\Projects\dex_arb\arbitrum\1inch_code\1inch_tokens.json") as file:
    oneinch_token_data = {token["token"]: token for token in json.load(file)}

# Initialize graph
G = nx.MultiGraph()

# Function to add edges between tokens and pools
def add_edges_to_graph(pools):
    for pool in pools.values():
        token0 = pool.get("token0")
        token1 = pool.get("token1")

        # Check if both tokens are in the 1inch token list
        if token0 in oneinch_token_data and token1 in oneinch_token_data:
            # Add edges between the tokens and the pool
            G.add_edge(token0, token1, lp_address=pool.get("pool_address"), pool_type=pool.get("type"))

# Process all pool data and connect tokens in the graph
add_edges_to_graph(uni_v2_lp_data)
add_edges_to_graph(sushi_v2_lp_data)
add_edges_to_graph(camelot_v2_lp_data)
add_edges_to_graph(v3_lp_data)

# Remove blacklisted nodes
G.remove_nodes_from(BLACKLISTED_TOKENS)

print(f"G ready: {len(G.nodes)} nodes, {len(G.edges)} edges")

# Find and print arbitrage paths
two_pool_arb_paths = {}
for token_address in oneinch_token_data.keys():
    if token_address in G.nodes:
        neighbors = list(G.neighbors(token_address))
        print(f"Token {token_address} has {len(neighbors)} pool pairs:")
        
        for neighbor in neighbors:
            pools = G.get_edge_data(token_address, neighbor).values()
            for pool_ in pools:
                pool_id = pool_.get("lp_address")
                pool_type = pool_.get("pool_type")
                
                # Assuming you have further logic here to handle the pool details and find arbitrage paths
                two_pool_arb_paths[pool_id] = {
                    "path": [token_address, neighbor],
                    "type": pool_type,
                    "details": pool_
                }

print(f"Found {len(two_pool_arb_paths)} unique two-pool arbitrage paths")
# Save the results to a JSON file
print("• Saving arb paths to JSON")
with open(r"C:\Users\PC\Projects\dex_arb\arbitrum\arbitrage_paths\arbitrum_arbs_2pool.json", "w") as file:
    json.dump(two_pool_arb_paths, file, indent=2)