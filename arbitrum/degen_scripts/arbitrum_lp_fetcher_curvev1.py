import sys
from typing import Dict
import ujson
import web3
from degenbot.constants import ZERO_ADDRESS
from degenbot.curve.abi import (
    CURVE_V1_STABLE_SWAP_FACTORY_ABI
)

# CONFIG: Dict[str, str] = dotenv.dotenv_values("mainnet_archive.env")
NODE_IPC_PATH =  "http://localhost:8547"

# Maximum blocks to process with get_logs
BLOCK_SPAN = 5_000

node_w3 = web3.Web3(web3.HTTPProvider(NODE_IPC_PATH))
if node_w3.is_connected() is False:
    sys.exit("Could not connect!")


registries = [
    {
        "name": "Curve V1: Stable Swap Factory",
        "filename": "arbitrum_lps_curvev1_registry.json",
        "factory_address": "0xb17b674D9c5CB2e441F8e196a2f048A81355d031",
        "pool_type": "CurveV1",
        "abi": CURVE_V1_STABLE_SWAP_FACTORY_ABI,
    },
    # {
    #     "name": "Curve V1: Factory",
    #     "filename": "ethereum_lps_curvev1_factory.json",
    #     "factory_address": "0x127db66E7F0b16470Bec194d0f496F9Fa065d0A9",
    #     "pool_type": "CurveV1",
    #     "abi": CURVE_V1_FACTORY_ABI,
    # },
]

current_block = node_w3.eth.block_number

for (
    name,
    registry_address,
    filename,
    pool_type,
    registry_abi,
) in [
    (
        registry["name"],
        registry["factory_address"],
        registry["filename"],
        registry["pool_type"],
        registry["abi"],
    )
    for registry in registries
]:
    print(name)

    registry_contract = node_w3.eth.contract(
        address=registry_address,
        abi=registry_abi,
    )

    try:
        with open(filename) as file:
            lp_data = ujson.load(file)
    except FileNotFoundError:
        lp_data = []

    if lp_data:
        previous_pool_count = len(lp_data)
        print(
            f"Found previously-fetched data: {previous_pool_count} pools"
        )
    else:
        previous_pool_count = 0

    current_pool_count: int = (
        registry_contract.functions.pool_count().call()
    )

    for pool_id in range(previous_pool_count, current_pool_count):
        pool_address = registry_contract.functions.pool_list(
            pool_id
        ).call()

        pool_coin_addresses = registry_contract.functions.get_coins(
            pool_address
        ).call()

        pool_is_meta = registry_contract.functions.is_meta(
            pool_address
        ).call()
        if pool_is_meta:
            pool_underlying_coin_addresses = (
                registry_contract.functions.get_underlying_coins(
                    pool_address
                ).call()
            )

        lp_data.append(
            {
                "pool_address": pool_address,
                "pool_id": pool_id,
                "type": pool_type,
                "coin_addresses": [
                    coin_address
                    for coin_address in pool_coin_addresses
                    if coin_address != ZERO_ADDRESS
                ],
            }
        )

        if pool_is_meta:
            lp_data[-1].update(
                {
                    "underlying_coin_addresses": [
                        coin_address
                        for coin_address in pool_underlying_coin_addresses
                        if coin_address != ZERO_ADDRESS
                    ],
                }
            )

    with open(filename, "w") as file:
        ujson.dump(lp_data, file, indent=2)

    print(
        f"Saved {len(lp_data)} pools ({len(lp_data) - previous_pool_count} new)"
    )