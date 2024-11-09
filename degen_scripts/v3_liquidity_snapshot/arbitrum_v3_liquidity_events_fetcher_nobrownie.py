from socket import timeout
import ujson
from threading import Lock
from typing import Dict
import web3
import sys

from web3._utils.events import get_event_data
from web3._utils.filters import construct_event_filter_params

import degenbot
import degenbot.uniswap.abi
from degenbot.uniswap import (
    UniswapV3BitmapAtWord,
    UniswapV3LiquidityAtTick,
    UniswapV3PoolExternalUpdate,
    UniswapV3PoolState,
)


NODE_URI = "http://localhost:8547"
w3 = web3.Web3(web3.HTTPProvider(NODE_URI,request_kwargs={'timeout': 60}))
if w3.is_connected() is False:
    sys.exit("Could not connect!")

SNAPSHOT_FILENAME = r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\v3_liquidity_snapshot\arbitrum_v3_liquidity_snapshot.json"
UNISWAPV3_START_BLOCK = 165
MAX_BLOCK_SPAN = 100_000  # Nitro node enforces a max query range of 100,000. Adjust downward to suit your node.


TICKSPACING_BY_FEE: Dict = {
    100: 1,
    500: 10,
    3000: 60,
    10000: 200,
}


class MockV3LiquidityPool(degenbot.V3LiquidityPool):
    def __init__(self):
        pass


current_block = w3.eth.block_number
print("Starting pool primer")

liquidity_snapshot: Dict[str, Dict] = {}

lp_data: Dict[str, Dict] = {}

for path in [
    r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_sushiswapv3.json",
    r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_uniswapv3.json",
]:
    try:
        with open(path, "r") as file:
            lps = ujson.load(file)
        for lp in lps:
            lp_data[lp["pool_address"]] = lp
    except Exception as e:
        print(e)

try:
    with open(SNAPSHOT_FILENAME, "r") as file:
        json_liquidity_snapshot = ujson.load(file)
except Exception:
    snapshot_last_block = None
else:
    snapshot_last_block = json_liquidity_snapshot.pop("snapshot_block")
    print(
        f"Loaded LP snapshot: {len(json_liquidity_snapshot)} pools @ block {snapshot_last_block}"
    )

    assert (
        snapshot_last_block < current_block
    ), f"Aborting, snapshot block ({snapshot_last_block}) is newer than current chain height ({current_block})"

    # Transform the JSON-encoded info from the snapshot to the dataclass
    # used by V3LiquidityPool
    for pool_address, snapshot in json_liquidity_snapshot.items():
        liquidity_snapshot[pool_address] = {
            "tick_bitmap": {
                int(k): UniswapV3BitmapAtWord(**v)
                for k, v in snapshot["tick_bitmap"].items()
            },
            "tick_data": {
                int(k): UniswapV3LiquidityAtTick(**v)
                for k, v in snapshot["tick_data"].items()
            },
        }

pool_contract = w3.eth.contract(
    abi=degenbot.uniswap.abi.UNISWAP_V3_POOL_ABI
)

liquidity_events = {}

for method_name in ("get_logs", "getLogs"):
    try:
        get_logs_method = getattr(w3.eth, method_name)
    except AttributeError:
        pass
    else:
        break

print(f"Using {method_name} for log fetching")

for event in [pool_contract.events.Mint, pool_contract.events.Burn]:
    print(f"Processing {event.event_name} events")

    start_block = (
        max(UNISWAPV3_START_BLOCK, snapshot_last_block + 1)
        if snapshot_last_block is not None
        else UNISWAPV3_START_BLOCK
    )
    block_span = MAX_BLOCK_SPAN
    done = False

    event_abi = event._get_event_abi()

    while not done:
        end_block = min(current_block, start_block + block_span - 1)
        # print(f"{start_block=}")
        # print(f"{end_block=}")

        _, event_filter_params = construct_event_filter_params(
            event_abi=event_abi,
            abi_codec=w3.codec,
            fromBlock=start_block,
            toBlock=end_block,
        )

        try:
            event_logs = get_logs_method(event_filter_params)
        except Exception as e:
            print(f"{type(e)}: {e}")
            block_span = int(0.75 * block_span)
            continue

        for event in event_logs:
            decoded_event = get_event_data(w3.codec, event_abi, event)

            pool_address = decoded_event["address"]
            block = decoded_event["blockNumber"]
            tx_index = decoded_event["transactionIndex"]
            liquidity = decoded_event["args"]["amount"] * (
                -1 if decoded_event["event"] == "Burn" else 1
            )
            tick_lower = decoded_event["args"]["tickLower"]
            tick_upper = decoded_event["args"]["tickUpper"]

            # skip zero liquidity events
            if liquidity == 0:
                continue

            try:
                liquidity_events[pool_address]
            except KeyError:
                liquidity_events[pool_address] = []

            liquidity_events[pool_address].append(
                (
                    block,
                    tx_index,
                    (
                        liquidity,
                        tick_lower,
                        tick_upper,
                    ),
                )
            )

        print(f"Fetched events: block span [{start_block},{end_block}]")

        if end_block == current_block:
            done = True
        else:
            start_block = end_block + 1
            block_span = min(int(1.05 * block_span), MAX_BLOCK_SPAN)

lp_helper = MockV3LiquidityPool()
lp_helper._subscribers = set()
lp_helper._sparse_bitmap = False
lp_helper._update_log = list()
lp_helper._state_lock = Lock()
lp_helper.state = UniswapV3PoolState(
    pool=lp_helper,
    liquidity=0,
    sqrt_price_x96=0,
    tick=0,
    tick_bitmap={},
    tick_data={},
)

for pool_address in liquidity_events.keys():
    # Ignore all pool addresses not held in the LP data dict. A strange
    # pool (0x820e891b14149e98b48b39ee2667157Ef750539b) was triggering an
    # early termination because it had liquidity events, but was not
    # associated with the known factories.
    if not lp_data.get(pool_address):
        continue

    try:
        previous_snapshot_tick_data = liquidity_snapshot[pool_address][
            "tick_data"
        ]
    except KeyError:
        previous_snapshot_tick_data = {}

    try:
        previous_snapshot_tick_bitmap = liquidity_snapshot[
            pool_address
        ]["tick_bitmap"]
    except KeyError:
        previous_snapshot_tick_bitmap = {}

    lp_helper.address = "0x0000000000000000000000000000000000000000"
    lp_helper.liquidity = 1 << 256
    lp_helper.tick_data = previous_snapshot_tick_data
    lp_helper.tick_bitmap = previous_snapshot_tick_bitmap
    lp_helper._update_block = (
        snapshot_last_block or UNISWAPV3_START_BLOCK
    )
    lp_helper.liquidity_update_block = (
        snapshot_last_block or UNISWAPV3_START_BLOCK
    )
    lp_helper.tick = 0
    lp_helper._fee = lp_data[pool_address]["fee"]
    lp_helper._tick_spacing = TICKSPACING_BY_FEE[lp_helper._fee]
    lp_helper.sqrt_price_x96 = 0
    lp_helper._pool_state_archive = dict()

    sorted_liquidity_events = sorted(
        liquidity_events[pool_address],
        key=lambda event: (event[0], event[1]),
    )

    for liquidity_event in sorted_liquidity_events:
        (
            event_block,
            _,
            (liquidity_delta, tick_lower, tick_upper),
        ) = liquidity_event

        # Push the liquidity events into the mock helper
        lp_helper.external_update(
            update=UniswapV3PoolExternalUpdate(
                block_number=event_block,
                liquidity_change=(
                    liquidity_delta,
                    tick_lower,
                    tick_upper,
                ),
            ),
        )

    # After all events have been pushed, update the liquidity snapshot with
    # the full liquidity data from the helper
    try:
        liquidity_snapshot[pool_address]
    except KeyError:
        liquidity_snapshot[pool_address] = {
            "tick_bitmap": {},
            "tick_data": {},
        }

    liquidity_snapshot[pool_address]["tick_bitmap"].update(
        lp_helper.tick_bitmap
    )
    liquidity_snapshot[pool_address]["tick_data"].update(
        lp_helper.tick_data
    )

for pool_address in liquidity_snapshot:
    # Convert all liquidity data to JSON format so it can be exported
    liquidity_snapshot[pool_address] = {
        "tick_data": {
            key: value.to_dict()
            for key, value in liquidity_snapshot[pool_address][
                "tick_data"
            ].items()
        },
        "tick_bitmap": {
            key: value.to_dict()
            for key, value in liquidity_snapshot[pool_address][
                "tick_bitmap"
            ].items()
            if value.bitmap  # skip empty bitmaps
        },
    }

liquidity_snapshot["snapshot_block"] = current_block

with open(SNAPSHOT_FILENAME, "w") as file:
    ujson.dump(
        liquidity_snapshot,
        file,
        indent=2,
        sort_keys=True,
    )
    print("Wrote LP snapshot")
