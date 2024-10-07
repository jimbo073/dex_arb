import ujson
import web3
import networkx as nx
import itertools
import time

WETH_ADDRESS = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"

start_timer = time.perf_counter()


BLACKLISTED_TOKENS = [
    # "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
    # "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
]

v2_lp_data = {}
for filename in [
    "ethereum_lps_sushiswapv2.json",
    "ethereum_lps_uniswapv2.json",
]:
    with open(filename) as file:
        for pool in ujson.load(file):
            v2_lp_data[pool.get("pool_address")] = {
                key: value
                for key, value in pool.items()
                if key not in ["pool_id"]
            }
print(f"Found {len(v2_lp_data)} V2 pools")

v3_lp_data = {}
for filename in [
    "ethereum_lps_sushiswapv3.json",
    "ethereum_lps_uniswapv3.json",
]:
    with open(filename) as file:
        pool_data = ujson.load(file)
    pool_data.pop()  # drop the last entry, which is metadata
    for pool in pool_data:
        v3_lp_data[pool.get("pool_address")] = pool
print(f"Found {len(v3_lp_data)} V3 pools")

curve_v1_lp_data = {}
for filename in [
    "ethereum_lps_curvev1_factory.json",
    "ethereum_lps_curvev1_registry.json",
]:
    with open(filename) as file:
        pool_data = ujson.load(file)
    for pool in pool_data:
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


# delete nodes for blacklisted tokens
G.remove_nodes_from(BLACKLISTED_TOKENS)

print(f"G ready: {len(G.nodes)} nodes, {len(G.edges)} edges")

all_tokens_with_weth_pool = list(G.neighbors(WETH_ADDRESS))
print(f"Found {len(all_tokens_with_weth_pool)} tokens with a WETH pair")

print("*** Finding triangular arbitrage paths ***")
triangle_arb_paths = {}

# only consider tokens with degree (number of pools holding the token) > 1
filtered_tokens = [
    token for token in all_tokens_with_weth_pool if G.degree(token) > 1
]
print(f"Processing {len(filtered_tokens)} tokens with degree > 1")


# loop through all possible starting/ending tokens that have a WETH pair
for token_a, token_b in itertools.combinations(filtered_tokens, 2):
    # find tokenA/tokenB pools, skip if a tokenA/tokenB pool is not found
    if not G.get_edge_data(token_a, token_b):
        continue

    inside_pools = [
        edge["lp_address"]
        for edge in G.get_edge_data(token_a, token_b).values()
        if edge["pool_type"] in ["CurveV1"]
    ]

    # Drop metapools where both token_a and token_b are held by the underlying basepool, and the metapool is not involved,
    # because it's more efficient to use the basepool directly
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

    # TODO: support Curve pools in position 0 or 2

    # find tokenA/WETH pools (Uniswap only)
    outside_pools_tokenA = [
        edge["lp_address"]
        for edge in G.get_edge_data(token_a, WETH_ADDRESS).values()
        if edge["pool_type"] in ["UniswapV2", "UniswapV3"]
    ]

    # find tokenB/WETH pools (Uniswap only)
    outside_pools_tokenB = [
        edge["lp_address"]
        for edge in G.get_edge_data(token_b, WETH_ADDRESS).values()
        if edge["pool_type"] in ["UniswapV2", "UniswapV3"]
    ]

    # find all triangular arbitrage paths of form:
    # WETH/tokenA -> tokenA/tokenB -> tokenB/WETH
    for pool_addresses in itertools.product(
        outside_pools_tokenA, inside_pools, outside_pools_tokenB
    ):
        pool_data = {}
        for pool_address in pool_addresses:
            if pool_address in all_uni_v2_pools:
                pool_info = {pool_address: v2_lp_data[pool_address]}
            elif pool_address in all_uni_v3_pools:
                pool_info = {pool_address: v3_lp_data[pool_address]}
            elif pool_address in all_curve_v1_pools:
                pool_info = {
                    pool_address: curve_v1_lp_data[pool_address]
                }
            else:
                raise Exception
            pool_data.update(pool_info)

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
            "path": pool_addresses,
            "pools": pool_data,
        }

    # find all triangular arbitrage paths of form:
    # WETH/tokenB -> tokenB/tokenA -> tokenA/WETH
    for pool_addresses in itertools.product(
        outside_pools_tokenB, inside_pools, outside_pools_tokenA
    ):
        pool_data = {}
        for pool_address in pool_addresses:
            if pool_address in all_uni_v2_pools:
                pool_info = {pool_address: v2_lp_data[pool_address]}
            elif pool_address in all_uni_v3_pools:
                pool_info = {pool_address: v3_lp_data[pool_address]}
            elif pool_address in all_curve_v1_pools:
                pool_info = {
                    pool_address: curve_v1_lp_data[pool_address]
                }
            else:
                raise Exception
            pool_data.update(pool_info)

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
            "path": pool_addresses,
            "pools": pool_data,
        }

print(
    f"Found {len(triangle_arb_paths)} triangle arb paths in {time.perf_counter() - start_timer:.1f}s"
)

print("• Saving arb paths to JSON")
with open("arbitrum_arbs_3pool_curve.json", "w") as file:
    ujson.dump(triangle_arb_paths, file, indent=2)