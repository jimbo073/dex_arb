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
[{"inputs":[{"internalType":"address","name":"feeTo_","type":"address"}],"payable":false,"stateMutability":"nonpayable","type":"constructor"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"prevOwner","type":"address"},{"indexed":true,"internalType":"address","name":"newOwner","type":"address"}],"name":"FeePercentOwnershipTransferred","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"prevFeeTo","type":"address"},{"indexed":true,"internalType":"address","name":"newFeeTo","type":"address"}],"name":"FeeToTransferred","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"uint256","name":"prevOwnerFeeShare","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"ownerFeeShare","type":"uint256"}],"name":"OwnerFeeShareUpdated","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"prevOwner","type":"address"},{"indexed":true,"internalType":"address","name":"newOwner","type":"address"}],"name":"OwnershipTransferred","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"token0","type":"address"},{"indexed":true,"internalType":"address","name":"token1","type":"address"},{"indexed":false,"internalType":"address","name":"pair","type":"address"},{"indexed":false,"internalType":"uint256","name":"length","type":"uint256"}],"name":"PairCreated","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"address","name":"referrer","type":"address"},{"indexed":false,"internalType":"uint256","name":"prevReferrerFeeShare","type":"uint256"},{"indexed":false,"internalType":"uint256","name":"referrerFeeShare","type":"uint256"}],"name":"ReferrerFeeShareUpdated","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"prevOwner","type":"address"},{"indexed":true,"internalType":"address","name":"newOwner","type":"address"}],"name":"SetStableOwnershipTransferred","type":"event"},{"constant":true,"inputs":[],"name":"OWNER_FEE_SHARE_MAX","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"REFERER_FEE_SHARE_MAX","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[{"internalType":"uint256","name":"","type":"uint256"}],"name":"allPairs","outputs":[{"internalType":"address","name":"","type":"address"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"allPairsLength","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":false,"inputs":[{"internalType":"address","name":"tokenA","type":"address"},{"internalType":"address","name":"tokenB","type":"address"}],"name":"createPair","outputs":[{"internalType":"address","name":"pair","type":"address"}],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":true,"inputs":[],"name":"feeInfo","outputs":[{"internalType":"uint256","name":"_ownerFeeShare","type":"uint256"},{"internalType":"address","name":"_feeTo","type":"address"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"feePercentOwner","outputs":[{"internalType":"address","name":"","type":"address"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"feeTo","outputs":[{"internalType":"address","name":"","type":"address"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[{"internalType":"address","name":"","type":"address"},{"internalType":"address","name":"","type":"address"}],"name":"getPair","outputs":[{"internalType":"address","name":"","type":"address"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"owner","outputs":[{"internalType":"address","name":"","type":"address"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"ownerFeeShare","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[{"internalType":"address","name":"","type":"address"}],"name":"referrersFeeShare","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":false,"inputs":[{"internalType":"address","name":"_feePercentOwner","type":"address"}],"name":"setFeePercentOwner","outputs":[],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":false,"inputs":[{"internalType":"address","name":"_feeTo","type":"address"}],"name":"setFeeTo","outputs":[],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":false,"inputs":[{"internalType":"address","name":"_owner","type":"address"}],"name":"setOwner","outputs":[],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":false,"inputs":[{"internalType":"uint256","name":"newOwnerFeeShare","type":"uint256"}],"name":"setOwnerFeeShare","outputs":[],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":false,"inputs":[{"internalType":"address","name":"referrer","type":"address"},{"internalType":"uint256","name":"referrerFeeShare","type":"uint256"}],"name":"setReferrerFeeShare","outputs":[],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":false,"inputs":[{"internalType":"address","name":"_setStableOwner","type":"address"}],"name":"setSetStableOwner","outputs":[],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":true,"inputs":[],"name":"setStableOwner","outputs":[{"internalType":"address","name":"","type":"address"}],"payable":false,"stateMutability":"view","type":"function"}]
"""
)

w3 = web3.Web3(web3.HTTPProvider(NODE_RPC_URI))
if w3.is_connected() is False:
    sys.exit("Could not connect!")

exchanges = [
    {
        "name": "Camelot",
        "filename": "arbitrum_lps_camelotv2.json",
        "factory_address": "0x6EcCab422D763aC031210895C81787E87B43A652",
        "factory_deployment_block": 35061163,
    },
]

newest_block = w3.eth.block_number

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

            for event in pool_created_events:
                lp_data.append(
                    {
                        "pool_address": event.args.get("pair"),
                        "token0": event.args.get("token0"),
                        "token1": event.args.get("token1"),
                        "block_number": event.get("blockNumber"),
                        "pool_id": event.args.get("length"),
                        "type": "CamelotV2",
                    }
                )

        if end_block == newest_block:
            break

    with open(filename, "w") as file:
        json.dump(lp_data, file, indent=2)

    print(f"Saved {len(lp_data) - previous_pool_count} new pools")
