import sys
import json
import time
import web3
import web3.contract


NODE_RPC_URI = "http://localhost:8547"

# starting block span to process with getLogs
BLOCK_SPAN = 5_000


FACTORY_CONTRACT_ABI = json.loads(
    """
[{"inputs":[],"stateMutability":"nonpayable","type":"constructor"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"uint24","name":"fee","type":"uint24"},{"indexed":true,"internalType":"int24","name":"tickSpacing","type":"int24"}],"name":"FeeAmountEnabled","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"oldOwner","type":"address"},{"indexed":true,"internalType":"address","name":"newOwner","type":"address"}],"name":"OwnerChanged","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"token0","type":"address"},{"indexed":true,"internalType":"address","name":"token1","type":"address"},{"indexed":true,"internalType":"uint24","name":"fee","type":"uint24"},{"indexed":false,"internalType":"int24","name":"tickSpacing","type":"int24"},{"indexed":false,"internalType":"address","name":"pool","type":"address"}],"name":"PoolCreated","type":"event"},{"inputs":[{"internalType":"address","name":"tokenA","type":"address"},{"internalType":"address","name":"tokenB","type":"address"},{"internalType":"uint24","name":"fee","type":"uint24"}],"name":"createPool","outputs":[{"internalType":"address","name":"pool","type":"address"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint24","name":"fee","type":"uint24"},{"internalType":"int24","name":"tickSpacing","type":"int24"}],"name":"enableFeeAmount","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint24","name":"","type":"uint24"}],"name":"feeAmountTickSpacing","outputs":[{"internalType":"int24","name":"","type":"int24"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"","type":"address"},{"internalType":"address","name":"","type":"address"},{"internalType":"uint24","name":"","type":"uint24"}],"name":"getPool","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"owner","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"parameters","outputs":[{"internalType":"address","name":"factory","type":"address"},{"internalType":"address","name":"token0","type":"address"},{"internalType":"address","name":"token1","type":"address"},{"internalType":"uint24","name":"fee","type":"uint24"},{"internalType":"int24","name":"tickSpacing","type":"int24"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"_owner","type":"address"}],"name":"setOwner","outputs":[],"stateMutability":"nonpayable","type":"function"}]
"""
)

w3 = web3.Web3(web3.HTTPProvider(NODE_RPC_URI))

if w3.is_connected() is False:
    sys.exit("Could not connect!")


exchanges = [
    {
        "name": "Sushiswap (V3)",
        "filename": "arbitrum_lps_sushiswapv3.json",
        "factory_address": "0x1af415a1EbA07a4986a52B6f2e7dE7003D82231e",
        "factory_deployment_block": 75998697,
    },
]

newest_block = w3.eth.block_number
print(f"Current block height: {newest_block}")


for name, factory_address, filename, deployment_block in [
    (
        exchange["name"],
        exchange["factory_address"],
        exchange["filename"],
        exchange["factory_deployment_block"],
    )
    for exchange in exchanges
]:
    print(f"DEX: {name}")

    factory_contract: web3.contract.Contract = w3.eth.contract(
        address=factory_address, abi=FACTORY_CONTRACT_ABI
    )
    try:
        with open(filename) as file:
            lp_data = json.load(file)
    except FileNotFoundError:
        lp_data = []

    if lp_data:
        previous_pool_count = len(lp_data)
        print(
            f"Found previously-fetched data: {previous_pool_count} pools"
        )
        previous_block = lp_data[-1].get("block_number")
        print(f"Found pool data up to block {previous_block}")
    else:
        previous_pool_count = 0
        previous_block = deployment_block

    failure = False
    start_block = previous_block + 1

    while True:
        if failure:
            BLOCK_SPAN = int(0.9 * BLOCK_SPAN)
            # reduce the working span by 10%
        else:
            # increase the working span by .1%
            BLOCK_SPAN = int(1.001 * BLOCK_SPAN)

        end_block = min(newest_block, start_block + BLOCK_SPAN)

        try:
            pool_created_events = (
                factory_contract.events.PoolCreated().get_logs(
                    fromBlock=start_block, toBlock=end_block
                )
            )
        except ValueError:
            failure = True
            time.sleep(1)
            continue
        else:
            print(
                f"Fetched PoolCreated events, block range [{start_block},{end_block}]"
            )
            # set the next start block
            start_block = end_block + 1
            failure = False

            # print(pool_created_events)
            for event in pool_created_events:
                lp_data.append(
                    {
                        "pool_address": event.args.get("pool"),
                        "fee": event.args.get("fee"),
                        "token0": event.args.get("token0"),
                        "token1": event.args.get("token1"),
                        "block_number": event.get("blockNumber"),
                        "type": "SushiswapV3",
                    }
                )

        if end_block == newest_block:
            break

    with open(filename, "w") as file:
        json.dump(lp_data, file, indent=2)

    print(f"Saved {len(lp_data) - previous_pool_count} new pools")
