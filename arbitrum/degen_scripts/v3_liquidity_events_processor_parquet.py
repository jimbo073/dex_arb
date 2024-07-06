import sys
from threading import Lock
from typing import Dict

import degenbot
import pyarrow
import pyarrow.compute
import pyarrow.parquet
import ujson
from degenbot import (
    UniswapV3BitmapAtWord,
    UniswapV3LiquidityAtTick,
    UniswapV3PoolExternalUpdate,
    UniswapV3PoolState,
)
from eth_utils import to_checksum_address
from hexbytes import HexBytes

MINT_LIQUIDITY_EVENTS_PATH = "./uniswap_v3_mint_events"
BURN_LIQUIDITY_EVENTS_PATH = "./uniswap_v3_burn_events/"
SNAPSHOT_FILENAME = "ethereum_v3_liquidity_snapshot.json"

TICKSPACING_BY_FEE: Dict = {
    100: 1,
    500: 10,
    3000: 60,
    10000: 200,
}

EARLIEST_BLOCK = 12_369_621


class MockV3LiquidityPool(degenbot.V3LiquidityPool):
    def __init__(self):
        pass


print("Processing V3 Liquidity Events...")

liquidity_snapshot: Dict[str, Dict] = {}
lp_data: Dict[str, Dict] = {}

for path in [
    "ethereum_lps_sushiswapv3.json",
    "ethereum_lps_uniswapv3.json",
]:
    try:
        with open(path, "r") as file:
            lps: list = ujson.load(file)
        lps.pop()  # strip metadata entry
        for lp in lps:
            lp_data[lp["pool_address"]] = lp
    except Exception as exc:
        print(f"{type(exc)}: {exc} on row {lp}")


try:
    with open(SNAPSHOT_FILENAME, "r") as file:
        liquidity_snapshot: dict = ujson.load(file)
except Exception:
    snapshot_last_block = 0
else:
    snapshot_last_block = liquidity_snapshot.pop("snapshot_block")
    print(
        f"Loaded LP snapshot: {len(liquidity_snapshot)} pools @ block {snapshot_last_block}"
    )

    # Transform the keys to int (JSON stores strings)
    for pool_address, snapshot in liquidity_snapshot.items():
        liquidity_snapshot[pool_address] = {
            "tick_bitmap": {int(k): v for k, v in snapshot["tick_bitmap"].items()},
            "tick_data": {int(k): v for k, v in snapshot["tick_data"].items()},
        }


mint_liquidity_events = pyarrow.parquet.read_table(
    source=MINT_LIQUIDITY_EVENTS_PATH,
    filters=(
        pyarrow.compute.field("block_number")
        > pyarrow.scalar(snapshot_last_block, type=pyarrow.uint32())
    ),
).combine_chunks()

burn_liquidity_events = pyarrow.parquet.read_table(
    source=BURN_LIQUIDITY_EVENTS_PATH,
    filters=(
        pyarrow.compute.field("block_number")
        > pyarrow.scalar(snapshot_last_block, type=pyarrow.uint32())
    ),
).combine_chunks()

if not mint_liquidity_events and not burn_liquidity_events:
    sys.exit("No new events")

pool_addresses = set(
    pyarrow.compute.unique(mint_liquidity_events["address"]).to_pylist()
    + pyarrow.compute.unique(burn_liquidity_events["address"]).to_pylist()
)

for pool_address in pool_addresses:
    _pool_address_bytes = HexBytes(pool_address)
    _pool_address_hex = to_checksum_address(_pool_address_bytes.hex())

    if _pool_address_hex not in lp_data:
        continue

    # Filter for the liquidity events for this pool only
    mint_events = mint_liquidity_events.filter(
        pyarrow.compute.field("address")
        == pyarrow.scalar(_pool_address_bytes, type=pyarrow.large_binary())
    )
    burn_events = burn_liquidity_events.filter(
        pyarrow.compute.field("address")
        == pyarrow.scalar(_pool_address_bytes, type=pyarrow.large_binary())
    )

    try:
        previous_snapshot_tick_data = liquidity_snapshot[_pool_address_hex]["tick_data"]
    except KeyError:
        previous_snapshot_tick_data = {}

    try:
        previous_snapshot_tick_bitmap = liquidity_snapshot[_pool_address_hex][
            "tick_bitmap"
        ]
    except KeyError:
        previous_snapshot_tick_bitmap = {}

    lp_helper = MockV3LiquidityPool()
    lp_helper.address = _pool_address_hex
    lp_helper.name = ""
    lp_helper._pool_state_archive = dict()
    lp_helper._update_block = 0
    lp_helper._fee = lp_data[_pool_address_hex]["fee"]
    lp_helper._tick_spacing = TICKSPACING_BY_FEE[lp_helper._fee]
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
    lp_helper.tick_bitmap = {
        int(k): UniswapV3BitmapAtWord(**v)
        for k, v in previous_snapshot_tick_bitmap.items()
    }
    lp_helper.tick_data = {
        int(k): UniswapV3LiquidityAtTick(**v)
        for k, v in previous_snapshot_tick_data.items()
    }
    lp_helper.liquidity = 1 << 256

    for event_index in range(len(mint_events)):
        mint_event = mint_events.take([event_index])
        mint_event_block = mint_event["block_number"][0].as_py()
        mint_event_tick_lower = mint_event["event__tickLower"][0].as_py()
        mint_event_tick_upper = mint_event["event__tickUpper"][0].as_py()
        mint_event_amount = int.from_bytes(
            mint_event["event__amount_binary"][0].as_py()
        )

        try:
            # push values into the helper
            lp_helper.external_update(
                update=UniswapV3PoolExternalUpdate(
                    block_number=mint_event_block,
                    liquidity_change=(
                        mint_event_amount,
                        mint_event_tick_lower,
                        mint_event_tick_upper,
                    ),
                ),
            )
        except Exception as exc:
            print(f"EXCEPTION: {type(exc)}: {exc}")
            print(f"{mint_event_block=}")
            print(f"{mint_event_amount=}")
            print(f"{mint_event_tick_lower=}")
            print(f"{mint_event_tick_upper=}")
            print(f"{lp_helper.tick_bitmap=}")
            print(f"{lp_helper.tick_data=}")
            sys.exit()

    # For simplicity, burns are applied after mints. Reset the update block so old burn events aren't rejected
    lp_helper._update_block = 0

    for event_index in range(len(burn_events)):
        burn_event = burn_events.take([event_index])
        burn_event_block = burn_event["block_number"][0].as_py()
        burn_event_tick_lower = burn_event["event__tickLower"][0].as_py()
        burn_event_tick_upper = burn_event["event__tickUpper"][0].as_py()
        burn_event_amount = int.from_bytes(
            burn_event["event__amount_binary"][0].as_py()
        )

        try:
            # push values into the helper
            lp_helper.external_update(
                update=UniswapV3PoolExternalUpdate(
                    block_number=burn_event_block,
                    liquidity_change=(
                        -burn_event_amount,  # must be negative to indicate BURN
                        burn_event_tick_lower,
                        burn_event_tick_upper,
                    ),
                ),
            )
        except Exception as exc:
            print(f"EXCEPTION: {type(exc)}: {exc}")
            print(f"{burn_event_block=}")
            print(f"{burn_event_amount=}")
            print(f"{burn_event_tick_lower=}")
            print(f"{burn_event_tick_upper=}")
            print(f"{lp_helper.tick_bitmap=}")
            print(f"{lp_helper.tick_data=}")
            sys.exit()

    # After all events have been pushed, update the liquidity snapshot with
    # the liquidity data from the helper
    try:
        liquidity_snapshot[_pool_address_hex]
    except KeyError:
        liquidity_snapshot[_pool_address_hex] = {
            "tick_bitmap": {},
            "tick_data": {},
        }
    finally:
        liquidity_snapshot[_pool_address_hex]["tick_bitmap"] = {
            int(k): v.to_dict() for k, v in lp_helper.tick_bitmap.items()
        }
        liquidity_snapshot[_pool_address_hex]["tick_data"] = {
            int(k): v.to_dict() for k, v in lp_helper.tick_data.items()
        }


last_mint_event_block = (
    pyarrow.compute.max(mint_liquidity_events["block_number"]).as_py()
    or snapshot_last_block  # there may be no events, so use the snapshot as a failsafe value
)
last_burn_event_block = (
    pyarrow.compute.max(burn_liquidity_events["block_number"]).as_py()
    or snapshot_last_block  # there may be no events, so use the snapshot as a failsafe value
)

liquidity_snapshot["snapshot_block"] = max(
    last_mint_event_block,
    last_burn_event_block,
)

with open(SNAPSHOT_FILENAME, "w") as file:
    ujson.dump(
        liquidity_snapshot,
        file,
        indent=2,
        sort_keys=True,
    )
    print(f'Wrote LP snapshot at block {liquidity_snapshot["snapshot_block"]}')
