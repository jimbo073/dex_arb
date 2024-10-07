import sys
import json
import time
import web3
import web3.contract

NODE_RPC_URI = "http://localhost:8547"

# starting # of blocks to request with getLogs
BLOCK_SPAN = 5_000

FACTORY_CONTRACT_ABI = json.loads(
    """
[{"inputs":[{"internalType":"address","name":"_feeToSetter","type":"address"}],"stateMutability":"nonpayable","type":"constructor"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"token0","type":"address"},{"indexed":true,"internalType":"address","name":"token1","type":"address"},{"indexed":false,"internalType":"address","name":"pair","type":"address"},{"indexed":false,"internalType":"uint256","name":"","type":"uint256"}],"name":"PairCreated","type":"event"},{"inputs":[{"internalType":"uint256","name":"","type":"uint256"}],"name":"allPairs","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"allPairsLength","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"tokenA","type":"address"},{"internalType":"address","name":"tokenB","type":"address"}],"name":"createPair","outputs":[{"internalType":"address","name":"pair","type":"address"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"feeTo","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"feeToSetter","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"","type":"address"},{"internalType":"address","name":"","type":"address"}],"name":"getPair","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"migrator","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"pairCodeHash","outputs":[{"internalType":"bytes32","name":"","type":"bytes32"}],"stateMutability":"pure","type":"function"},{"inputs":[{"internalType":"address","name":"_feeTo","type":"address"}],"name":"setFeeTo","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"_feeToSetter","type":"address"}],"name":"setFeeToSetter","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"_migrator","type":"address"}],"name":"setMigrator","outputs":[],"stateMutability":"nonpayable","type":"function"}]
"""
)

w3 = web3.Web3(web3.HTTPProvider(NODE_RPC_URI))
if w3.is_connected() is False:
    sys.exit("Could not connect!")

exchanges = [
    {
        "name": "Uniswap (V2)",
        "filename": "arbitrum_lps_uniswapv2.json",
        "factory_address": "0xf1D7CC64Fb4452F05c498126312eBE29f30Fbcf9",
        "factory_deployment_block": 150442611,
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
            # reduce the working span by 10%
            BLOCK_SPAN = int(0.9 * BLOCK_SPAN)

        else:
            # increase the working span by .1%
            BLOCK_SPAN = int(1.001 * BLOCK_SPAN)

        end_block = min(newest_block, start_block + BLOCK_SPAN)

        try:
            pool_created_events = (
                factory_contract.events.PairCreated().get_logs(
                    fromBlock=start_block, toBlock=end_block
                )
            )
        except ValueError:
            failure = True
            time.sleep(1)
            continue
        else:
            print(
                f"Fetched PairCreated events, block range [{start_block},{end_block}]"
            )
            # set the next start block
            start_block = end_block + 1
            failure = False
            
            try:
                for event in pool_created_events:
                    lp_data.append(
                        {
                            "pool_address": event.args.get("pair"),
                            "token0": event.args.get("token0"),
                            "token1": event.args.get("token1"),
                            "block_number": event.get("blockNumber"),
                            "pool_id": event.args.get(""),
                            "type": "UniswapV2",
                        }
                    )
                    
            except AttributeError as e:
                print(e)
                print(event)
                print(lp_data)
                time.sleep(1)
                continue

        if end_block == newest_block:
            break

    with open(filename, "w") as file:
        json.dump(lp_data, file, indent=2)

    print(f"Saved {len(lp_data) - previous_pool_count} new pools")
