import asyncio
import functools
import ujson as json
from dex_arb.arbitrum._1inch_aggregator_classes import OneInchAggregator
import logging
import multiprocessing
import os
import signal
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import web3
import degenbot
import eth_abi
import eth_account
import web3
import websockets.client
from web3._utils.events import get_event_data
from web3._utils.filters import construct_event_filter_params
import degenbot as bot
from degenbot import LiquidityPool, V3LiquidityPool, UniswapLpCycle
from degenbot.curve.curve_stableswap_liquidity_pool import CurveStableswapPool
from degenbot.uniswap.abi import UNISWAP_V3_POOL_ABI

# Globale Variablen
NODE_WEBSOCKET_URI = "ws://localhost:8548"
NODE_HTTP_URI = "http://localhost:8547"

ARBISCAN_API_KEY = "EDITME" # TODO ...
AGGREGATOR_ADDRESS = "0x111111125421cA6dc452d289314280a0f8842A65"
ARB_CONTRACT_ADDRESS = "EDITME" # TODO ...
WETH_ADDRESS = "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1"

MIN_PROFIT_ETH = int(0.000001 * 10**18)

DRY_RUN = True

VERBOSE_ACTIVATION = True
VERBOSE_BLOCKS = True
VERBOSE_EVENTS = True
VERBOSE_GAS_ESTIMATES = True
VERBOSE_PROCESSING = True
VERBOSE_SIMULATION = True
VERBOSE_TIMING = True
VERBOSE_UPDATES = True
VERBOSE_WATCHDOG = True

# require min. number of simulations before evaluating the cutoff threshold
SIMULATION_CUTOFF_MIN_ATTEMPTS = 10
# arbs that fail simulations greater than this percentage will be added to a blacklist
SIMULATION_CUTOFF_FAIL_THRESHOLD = 0.99

AVERAGE_BLOCK_TIME = 0.25
LATE_BLOCK_THRESHOLD = 10

# if True, triangle arbs will only be processed if the middle leg is updated.
REDUCE_TRIANGLE_ARBS = True


# Platzhalter für globale Datenstrukturen
last_base_fee = 1 * 10**9  # overridden on first received block
next_base_fee = 1 * 10**9  # overridden on first received block
newest_block_timestamp = int(time.time())  # overridden on first received block
status_events = False
status_live = False
status_new_blocks = False
status_pool_sync_in_progress = False
status_paused = True
first_new_block = 0
first_event_block = 0
degenbot_lp_helpers: Dict[str, Union[LiquidityPool, V3LiquidityPool]] = {}
inactive_arbs: Dict[str, UniswapLpCycle] = {}
active_arbs: Dict[str, UniswapLpCycle] = {}
arb_simulations: Dict[str, Dict] = dict()
active_arbs_to_check: Set[UniswapLpCycle] = set()
arbs_to_activate: Set[UniswapLpCycle] = set()

# --- Multiprocessing Setup ---
def handle_task_exception(loop, context):
    """
    By default, do nothing
    """

def _pool_worker_init():
    """
    Ignore SIGINT signals. Used for subprocesses spawned by `ProcessPoolExecutor` via the `initializer=` argument.

    Otherwise SIGINT is translated to KeyboardInterrupt, which is unhandled and will lead to messy tracebacks when thrown into a subprocess.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

async def main():
    global process_pool
    global all_tasks
    all_tasks = set()

    asyncio.get_running_loop().set_exception_handler(
        handle_task_exception
    )

    signals = (
        # signal.SIGHUP,
        # signal.SIGTERM,
        # signal.SIGINT,
        # signal.SIGBREAK, # For Windows users, will catch CTRL+C
    )

    for sig in signals:
        asyncio.get_event_loop().add_signal_handler(
            sig,
            shutdown,
        )

    with ProcessPoolExecutor(
        max_workers=16,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_pool_worker_init,
    ) as process_pool:
        for coro in [
            activate_arbs(),
            load_arbs(),
            process_onchain_arbs(),
            status(),
            sweep(),
            track_balance(),
            watchdog(),
            watch_events(),
            watch_new_blocks(),
        ]:
            task = asyncio.create_task(coro)
            task.add_done_callback(all_tasks.discard)
            all_tasks.add(task)

        try:
            await asyncio.gather(*all_tasks)
        except asyncio.exceptions.CancelledError:
            print("Shutting down all running tasks...")
        except Exception as e:
            print("(main) caught cancel")


# --- Platzhalter für Funktionendefinitionen ---

async def activate_arbs():
    """
    Aktiviert und verwaltet die Arbitrage-Möglichkeiten.
    Dies beinhaltet das Berechnen der besten Pfade und die Prüfung von Gas-Nutzung.
    """
    pass

async def load_arbs(): #TODO: implement the classes left for router and router arbitrage... ---------------------------------------
    """
    Lädt alle benötigten Arbitrage-Pfade und Pool-Daten und bereitet sie für die Berechnungen vor.
    """
    print("Starting arb loading function")

    global degenbot_lp_helpers
    global active_arbs
    global inactive_arbs
    global arb_simulations
    global status_live

    # liquidity_pool_and_token_addresses will filter out any blacklisted addresses, so helpers should draw from this as the "official" source of truth
    liquidity_pool_data = {}
    for filename in [
        r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_camelotv2.json",
        r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_curvev1_registry.json",
        r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_sushiswapv2.json",
        r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_uniswapv2.json",
        r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_sushiswapv3.json",
        r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_uniswapv3.json",
    ]:
        with open(filename) as file:
            for pool in json.load(file):
                pool_address = pool.get("pool_address")
                token0_address = pool.get("token0")
                token1_address = pool.get("token1")
                if (
                    token0_address in BLACKLISTED_TOKENS
                    or token1_address in BLACKLISTED_TOKENS
                ):
                    continue
                else:
                    liquidity_pool_data[pool_address] = pool
    print(f"Found {len(liquidity_pool_data)} pools")

    arb_paths = []
    for filename in [
       r"C:\Users\PC\Projects\dex_arb\arbitrum\arbitrage_paths\arbitrum_arbs_3pool_type1.json",
    ]:
        with open(filename) as file:
            for arb_id, arb in json.load(file).items():
                passed_checks = True
                if arb_id in BLACKLISTED_ARBS:
                    passed_checks = False
                for pool_address in arb.get("path"):
                    if pool_address == AGGREGATOR_ADDRESS:
                        continue
                    if not liquidity_pool_data.get(pool_address):
                        passed_checks = False
                if passed_checks:
                    arb_paths.append(arb)
    print(f"Found {len(arb_paths)} arb paths")

    # Identify all unique pool addresses in arb paths
    unique_pool_addresses = {
        pool_address
        for arb in arb_paths
        for pool_address in arb.get("path")
        if liquidity_pool_data.get(pool_address)
    }
    print(f"Found {len(unique_pool_addresses)} unique pools")
    
    # Identify all unique token addresses, checking if the pool is present in pools_and_tokens (pre-checked against the blacklist)
    # note: | is the operator for a set 'union' method
    unique_tokens = (
        # all token0 addresses
        {
            token_address
            for arb in arb_paths
            for pool_address in arb.get("path")
            for pool_dict in arb.get("pools").values()
            if (token_address := pool_dict.get("token0"))
            if token_address not in BLACKLISTED_TOKENS
            if liquidity_pool_data.get(pool_address)
        }
        |
        # all token1 addresses
        {
            token_address
            for arb in arb_paths
            for pool_address in arb.get("path")
            for pool_dict in arb.get("pools").values()
            if (token_address := pool_dict.get("token1"))
            if token_address not in BLACKLISTED_TOKENS
            if liquidity_pool_data.get(pool_address)
        }
        |
        # Tokens von Curve Pools: nutze entweder underlying_coin_addresses, wenn vorhanden, sonst coin_addresses
        {
            token_address
            for arb in arb_paths
            for pool_address in arb.get("path")
            for pool_dict in arb.get("pools").values()
            if pool_dict.get("type") == "CurveV1"
            for token_address in (
                pool_dict.get("coin_addresses")  # Extrahiere coin_addresses
                if pool_dict.get("underlying_coin_addresses") is None  # Wenn keine underlying vorhanden
                else pool_dict.get("coin_addresses") + pool_dict.get("underlying_coin_addresses")  # Extrahiere beides
            )
            if token_address not in BLACKLISTED_TOKENS
            if liquidity_pool_data.get(pool_address)
        }
    )
    print(f"Found {len(unique_tokens)} unique tokens")
    
    start = time.time()

    liquidity_snapshot = {}

    while not first_event_block:
        await asyncio.sleep(AVERAGE_BLOCK_TIME)

    # update the snapshot to the block before our event watcher came online
    snapshot_end_block = first_event_block - 1
    print(f"Updating snapshot to block {snapshot_end_block}")

    try:
        with open("arbitrum_liquidity_snapshot.json", "r") as file:
            json_liquidity_snapshot = json.load(file)
    except:
        snapshot_last_block = None
    else:
        snapshot_last_block = json_liquidity_snapshot["snapshot_block"]
        print(
            f"Loaded LP snapshot: {len(json_liquidity_snapshot)} pools @ block {snapshot_last_block}"
        )

        for pool_address, snapshot in [
            (k, v)
            for k, v in json_liquidity_snapshot.items()
            if k not in ["snapshot_block"]
        ]:
            liquidity_snapshot[pool_address] = {
                "tick_bitmap": {
                    int(k): v for k, v in snapshot["tick_bitmap"].items()
                },
                "tick_data": {
                    int(k): v for k, v in snapshot["tick_data"].items()
                },
            }

        V3LP = w3.eth.contract(abi=UNISWAP_V3_POOL_ABI)

        liquidity_events = {}

        for event in [V3LP.events.Mint, V3LP.events.Burn]:
            print(f"processing {event.event_name} events")

            start_block = snapshot_last_block + 1
            block_span = 10_000
            done = False

            event_abi = event._get_event_abi()

            while not done:
                end_block = min(snapshot_end_block, start_block + block_span)

                _, event_filter_params = construct_event_filter_params(
                    event_abi=event_abi,
                    abi_codec=w3.codec,
                    argument_filters={},
                    fromBlock=start_block,
                    toBlock=end_block,
                )

                try:
                    event_logs = w3.eth.get_logs(event_filter_params)
                except:
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

                print(
                    f"Fetched events: block span [{start_block},{end_block}]"
                )

                if end_block == snapshot_end_block:
                    done = True
                else:
                    start_block = end_block + 1
                    block_span = int(1.05 * block_span)

    for pool_address in unique_pool_addresses:
        await asyncio.sleep(0)

        pool_type = liquidity_pool_data[pool_address]["type"]

        try:
            if pool_type == "UniswapV2":
                pool_helper = bot.LiquidityPool(
                    address=pool_address,
                    # tokens=[token0_obj, token1_obj],
                    update_method="external",
                    silent=True,
                )
                
            elif pool_type == "CamelotV2":
                pool_helper = bot.uniswap.v2_liquidity_pool.CamelotLiquidityPool(
                    address=pool_address,
                    # tokens=[token0_obj, token1_obj],
                    update_method="external",
                    silent=True,
                )
                
            elif pool_type == "UniswapV3":
                try:
                    snapshot_tick_data = liquidity_snapshot[pool_address][
                        "tick_data"
                    ]
                except KeyError:
                    snapshot_tick_data = {}

                try:
                    snapshot_tick_bitmap = liquidity_snapshot[pool_address][
                        "tick_bitmap"
                    ]
                except KeyError:
                    snapshot_tick_bitmap = {}

                pool_helper = bot.V3LiquidityPool(
                    address=pool_address,
                    # tokens=[token0_obj, token1_obj],
                    update_method="external",
                    silent=True,
                    tick_data=snapshot_tick_data,
                    tick_bitmap=snapshot_tick_bitmap,
                )

                # update the helper with all liquidity events since the snapshot
                if liquidity_events.get(pool_address):
                    print(
                        f"processing {len(liquidity_events.get(pool_address))} liq event(s) for {pool_helper}"
                    )

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

                        pool_helper.external_update(
                            updates={
                                "liquidity_change": (
                                    liquidity_delta,
                                    tick_lower,
                                    tick_upper,
                                )
                            },
                            block_number=event_block,
                            fetch_missing=False,
                            force=True,
                        )
                pool_helper.update_block = snapshot_end_block
                
            elif pool_type == "CurveV1":
                try:
                    pool_helper = degenbot.CurveStableswapPool(
                        address=pool_address,
                        silent=True,
                        # state_block=bot_status.first_event - 1,
                    )
                except Exception as exc:
                    print(f"CurveV1 get_pool: {type(exc)}: {exc}")
                    continue

            else:
                raise Exception(
                    f"Could not identify pool type! {pool_type=}"
                )
                
        except Exception as e:
            print(e)
            print(type(e))
        else:
            # add the helper to the dictionary of LP objects, keyed by address
            degenbot_lp_helpers[pool_helper.address] = pool_helper

            if VERBOSE_PROCESSING:
                print(f"Created pool helper: {pool_helper}")

    print(
        f"Built {len(degenbot_lp_helpers)} liquidity pool helpers in {time.time() - start:.2f}s"
    )

    #----------- TODO : ADD LIST AND LIST COMPREHENSION TO RETRIEVE ALL !INCH TOKENS AND LOAD THEM AS ERC20 TOKENS!!!!!!!!!--------
    try:
        with open (r"C:\Users\PC\Projects\dex_arb\arbitrum\1inch_code\1inch_tokens.json", "r") as file:
            one_inch_token_list = json.load(file) # degenbot.Erc20Token(...)
            for token in one_inch_token_list:
                token_obj = degenbot.Erc20Token(token["token"])
    except FileNotFoundError:
            print("1inch_tokens.json not found!")
    try:
        with open("arbitrum_activated_arbs.json", "r") as file:
            _activated_arbs = json.load(file)
    except FileNotFoundError:
        _activated_arbs = []
        
# ----------------------------------TODO : ADD NEW CLASS ABBILITIES TO THE BOT!!!!!!!!!---------------------------
    # build a dict of arb helpers, keyed by arb ID
    inactive_arbs = {
        arb_id: bot.UniswapLpCycle(
            input_token=token_obj,
            swap_pools=swap_pools,
            # max_input=_weth_balance,
            id=arb_id,
        )
        for arb in arb_paths
        # ignore arbs on the blacklist and arbs where pool helpers are not available for ALL hops in the path
        if (arb_id := arb.get("id")) not in BLACKLISTED_ARBS
        if len(
            swap_pools := [
                pool_obj
                for pool_address in arb.get("path")
                if (pool_obj := degenbot_lp_helpers.get(pool_address))
            ]
        )
        == len(arb.get("path"))
    }
    print(f"Built {len(inactive_arbs)} cycle arb helpers")

    for id in _activated_arbs:
        try:
            active_arbs[id] = inactive_arbs.pop(id)
        except KeyError:
            pass

    print(f"Pre-activated {len(active_arbs)} cycle arb helpers")

    arb_simulations = {
        id: {
            "simulations": 0,
            "failures": 0,
        }
        for id in inactive_arbs
    }

    status_live = True
    print("Arb loading complete")

    

async def process_onchain_arbs():
    global active_arbs_to_check

    try:
        while True:
            while status_paused or not active_arbs_to_check:
                await asyncio.sleep(AVERAGE_BLOCK_TIME)

            _arbs_to_check = list(active_arbs_to_check)
            active_arbs_to_check.clear()

            await calculate_arb_in_pool(_arbs_to_check)
    except asyncio.exceptions.CancelledError:
        return

async def status():
    activated_arbs = 0

    while True:
        try:
            await asyncio.sleep(30)

            if activated_arbs != len(active_arbs):
                activated_arbs = len(active_arbs)
                print(
                    f"{activated_arbs} active / {len(inactive_arbs)} inactive arbs"
                )
        except asyncio.exceptions.CancelledError:
            return

async def sweep():
    """
    Sucht nach den profitabelsten Arbitrage-Möglichkeiten und führt sie aus.
    """
    pass

async def calculate_arb_in_pool(arbs: Iterable):
    """
    TBD
    """

    loop = asyncio.get_running_loop()

    global process_pool

    _tasks = [
        loop.run_in_executor(
            executor=process_pool,
            func=arb_helper.calculate_arbitrage_return_best, # TODO calc arbitragereturn best func !!!--------------------------------------------------------
        )
        for arb_helper in arbs
        if arb_helper.auto_update()
    ]

    for task in asyncio.as_completed(_tasks):
        try:
            arb_id, best = await task
        except bot.exceptions.ArbitrageError:
            pass
        except Exception as e:
            print(e)
        else:
            arb_helper = active_arbs[arb_id]
            arb_helper.best = best

async def track_balance():
    """
    Überwacht das Guthaben des Bots und aktualisiert die verfügbaren Mittel für Arbitragen.
    """
    pass

async def watchdog():
    """
    Tasked with monitoring other coroutines, functions, objects, etc. and
    setting bot status variables like `status_paused`

    Other coroutines should monitor the state of `status_paused` and adjust their activity as needed
    """

    global status_paused

    print("Starting status watchdog")

    while True:
        try:
            await asyncio.sleep(AVERAGE_BLOCK_TIME)

            # our node will always be slightly delayed compared to the timestamp of the block,
            # so compare that difference on each pass through the loop
            if (
                late_timer := (time.time() - newest_block_timestamp)
            ) > AVERAGE_BLOCK_TIME + LATE_BLOCK_THRESHOLD:
                # if the expected block is late, set the paused flag to True
                if not status_paused:
                    status_paused = True
                    if VERBOSE_WATCHDOG:
                        print(
                            f"WATCHDOG: paused (block {late_timer:.1f}s late)"
                        )
                        # print(f"{newest_block_timestamp=}")
            elif status_pool_sync_in_progress:
                if not status_paused:
                    status_paused = True
                    if VERBOSE_WATCHDOG:
                        print("WATCHDOG: paused (pool sync in progress)")
            else:
                if status_paused:
                    status_paused = False
                    if VERBOSE_WATCHDOG:
                        print("WATCHDOG: unpaused")

        except asyncio.exceptions.CancelledError:
            return


async def watch_events():
    global active_arbs_to_check
    global arbs_to_activate
    global first_event_block
    global status_events

    received_events = 0
    processed_mints = 0
    processed_swaps = 0
    processed_burns = 0
    processed_syncs = 0

    event_queue = deque()

    print("Starting event watcher loop")

    def process_burn_event(message: dict) -> None:
        event_address = w3.toChecksumAddress(
            message["params"]["result"]["address"]
        )
        event_block = int(
            message["params"]["result"]["blockNumber"],
            16,
        )
        event_data = message["params"]["result"]["data"]

        # ignore events for pools we are not tracking
        try:
            v3_pool_helper = degenbot_lp_helpers[event_address]
        except KeyError:
            return
        else:
            assert isinstance(v3_pool_helper, V3LiquidityPool)

        try:
            event_tick_lower = eth_abi.decode(
                ["int24"],
                bytes.fromhex(message["params"]["result"]["topics"][2][2:]),
            )[0]

            event_tick_upper = eth_abi.decode(
                ["int24"],
                bytes.fromhex(message["params"]["result"]["topics"][3][2:]),
            )[0]

            event_liquidity, _, _ = eth_abi.decode(
                ["uint128", "uint256", "uint256"],
                bytes.fromhex(event_data[2:]),
            )
        except KeyError:
            return
        else:
            if event_liquidity == 0:
                return

            try:
                v3_pool_helper.external_update(
                    updates={
                        "liquidity_change": (
                            -event_liquidity,
                            event_tick_lower,
                            event_tick_upper,
                        )
                    },
                    block_number=event_block,
                )
            except Exception as e:
                print(f"(process_burn_event): {e}")
            else:
                # find all arbs that care about this pool
                active_arbs_to_check.update(
                    set(
                        [
                            arb
                            for arb in active_arbs.values()
                            if len(arb.swap_pools) == 2
                            if v3_pool_helper in arb.swap_pools
                        ]
                    )
                )
                active_arbs_to_check.update(
                    set(
                        [
                            arb
                            for arb in active_arbs.values()
                            if len(arb.swap_pools) == 3
                            if (
                                v3_pool_helper is arb.swap_pools[1]
                                if REDUCE_TRIANGLE_ARBS
                                else v3_pool_helper in arb.swap_pools
                            )
                        ]
                    )
                )
                arbs_to_activate.update(
                    set(
                        [
                            arb
                            for arb in inactive_arbs.values()
                            if len(arb.swap_pools) == 2
                            if v3_pool_helper in arb.swap_pools
                        ]
                    )
                )
                arbs_to_activate.update(
                    set(
                        [
                            arb
                            for arb in inactive_arbs.values()
                            if len(arb.swap_pools) == 3
                            if (
                                v3_pool_helper is arb.swap_pools[1]
                                if REDUCE_TRIANGLE_ARBS
                                else v3_pool_helper in arb.swap_pools
                            )
                        ]
                    )
                )
        finally:
            nonlocal processed_burns
            processed_burns += 1
            if VERBOSE_EVENTS:
                print(f"[EVENT] Processed {processed_burns} burns")

    def process_mint_event(message: dict) -> None:
        event_address = w3.toChecksumAddress(
            message["params"]["result"]["address"]
        )
        event_block = int(
            message["params"]["result"]["blockNumber"],
            16,
        )
        event_data = message["params"]["result"]["data"]

        try:
            v3_pool_helper = degenbot_lp_helpers[event_address]
            event_tick_lower = eth_abi.decode(
                ["int24"],
                bytes.fromhex(message["params"]["result"]["topics"][2][2:]),
            )[0]

            event_tick_upper = eth_abi.decode(
                ["int24"],
                bytes.fromhex(message["params"]["result"]["topics"][3][2:]),
            )[0]

            _, event_liquidity, _, _ = eth_abi.decode(
                ["address", "uint128", "uint256", "uint256"],
                bytes.fromhex(event_data[2:]),
            )
        except KeyError:
            return
        else:
            assert isinstance(v3_pool_helper, V3LiquidityPool)
            if event_liquidity == 0:
                return

            try:
                v3_pool_helper.external_update(
                    updates={
                        "liquidity_change": (
                            event_liquidity,
                            event_tick_lower,
                            event_tick_upper,
                        )
                    },
                    block_number=event_block,
                )
            except Exception as e:
                print(f"(process_mint_event): {e}")
            else:
                # find all arbs that care about this pool
                active_arbs_to_check.update(
                    set(
                        [
                            arb
                            for arb in active_arbs.values()
                            if len(arb.swap_pools) == 2
                            if v3_pool_helper in arb.swap_pools
                        ]
                    )
                )
                active_arbs_to_check.update(
                    set(
                        [
                            arb
                            for arb in active_arbs.values()
                            if len(arb.swap_pools) == 3
                            if (
                                v3_pool_helper is arb.swap_pools[1]
                                if REDUCE_TRIANGLE_ARBS
                                else v3_pool_helper in arb.swap_pools
                            )
                        ]
                    )
                )
                arbs_to_activate.update(
                    set(
                        [
                            arb
                            for arb in inactive_arbs.values()
                            if len(arb.swap_pools) == 2
                            if v3_pool_helper in arb.swap_pools
                        ]
                    )
                )
                arbs_to_activate.update(
                    set(
                        [
                            arb
                            for arb in inactive_arbs.values()
                            if len(arb.swap_pools) == 3
                            if (
                                v3_pool_helper is arb.swap_pools[1]
                                if REDUCE_TRIANGLE_ARBS
                                else v3_pool_helper in arb.swap_pools
                            )
                        ]
                    )
                )
        finally:
            nonlocal processed_mints
            processed_mints += 1
            if VERBOSE_EVENTS:
                print(f"[EVENT] Processed {processed_mints} mints")

    def process_sync_event(message: dict) -> None:
        event_address = w3.toChecksumAddress(
            message["params"]["result"]["address"]
        )
        event_block = int(
            message["params"]["result"]["blockNumber"],
            16,
        )
        event_data = message["params"]["result"]["data"]

        event_reserves = eth_abi.decode(
            ["uint112", "uint112"],
            bytes.fromhex(event_data[2:]),
        )

        try:
            v2_pool_helper = degenbot_lp_helpers[event_address]
        except KeyError:
            return
        else:
            assert isinstance(v2_pool_helper, LiquidityPool)
            reserves0, reserves1 = event_reserves

            try:
                v2_pool_helper.update_reserves(
                    external_token0_reserves=reserves0,
                    external_token1_reserves=reserves1,
                    silent=not VERBOSE_UPDATES,
                    print_ratios=False,
                    print_reserves=False,
                    update_block=event_block,
                )
            except bot.exceptions.ExternalUpdateError:
                pass
            except Exception as e:
                print(f"(process_sync_event): {e}")
            else:
                active_arbs_to_check.update(
                    set(
                        [
                            arb
                            for arb in active_arbs.values()
                            if len(arb.swap_pools) == 2
                            if v2_pool_helper in arb.swap_pools
                        ]
                    )
                )
                active_arbs_to_check.update(
                    set(
                        [
                            arb
                            for arb in active_arbs.values()
                            if len(arb.swap_pools) == 3
                            if (
                                v2_pool_helper is arb.swap_pools[1]
                                if REDUCE_TRIANGLE_ARBS
                                else v2_pool_helper in arb.swap_pools
                            )
                        ]
                    )
                )
                arbs_to_activate.update(
                    set(
                        [
                            arb
                            for arb in inactive_arbs.values()
                            if len(arb.swap_pools) == 2
                            if v2_pool_helper in arb.swap_pools
                        ]
                    )
                )
                arbs_to_activate.update(
                    set(
                        [
                            arb
                            for arb in inactive_arbs.values()
                            if len(arb.swap_pools) == 3
                            if (
                                v2_pool_helper is arb.swap_pools[1]
                                if REDUCE_TRIANGLE_ARBS
                                else v2_pool_helper in arb.swap_pools
                            )
                        ]
                    )
                )
        finally:
            nonlocal processed_syncs
            processed_syncs += 1
            if VERBOSE_EVENTS:
                print(f"[EVENT] Processed {processed_syncs} syncs")

    def process_swap_event(message: dict) -> None:
        event_address = w3.toChecksumAddress(
            message["params"]["result"]["address"]
        )
        event_block = int(
            message["params"]["result"]["blockNumber"],
            16,
        )
        event_data = message["params"]["result"]["data"]

        (
            _,
            _,
            event_sqrt_price_x96,
            event_liquidity,
            event_tick,
        ) = eth_abi.decode(
            [
                "int256",
                "int256",
                "uint160",
                "uint128",
                "int24",
            ],
            bytes.fromhex(event_data[2:]),
        )

        try:
            v3_pool_helper = degenbot_lp_helpers[event_address]
        except KeyError:
            return
        else:
            assert isinstance(v3_pool_helper, V3LiquidityPool)

        try:
            v3_pool_helper.external_update(
                updates={
                    "tick": event_tick,
                    "liquidity": event_liquidity,
                    "sqrt_price_x96": event_sqrt_price_x96,
                },
                block_number=event_block,
            )
        except Exception as e:
            print(f"(process_swap_event): {e}")
        else:
            # find all activated arbs that track this pool
            active_arbs_to_check.update(
                set(
                    [
                        arb
                        for arb in active_arbs.values()
                        if len(arb.swap_pools) == 2
                        if v3_pool_helper in arb.swap_pools
                    ]
                )
            )
            active_arbs_to_check.update(
                set(
                    [
                        arb
                        for arb in active_arbs.values()
                        if len(arb.swap_pools) == 3
                        if (
                            v3_pool_helper is arb.swap_pools[1]
                            if REDUCE_TRIANGLE_ARBS
                            else v3_pool_helper in arb.swap_pools
                        )
                    ]
                )
            )
            arbs_to_activate.update(
                set(
                    [
                        arb
                        for arb in inactive_arbs.values()
                        if len(arb.swap_pools) == 2
                        if v3_pool_helper in arb.swap_pools
                    ]
                )
            )
            arbs_to_activate.update(
                set(
                    [
                        arb
                        for arb in inactive_arbs.values()
                        if len(arb.swap_pools) == 3
                        if (
                            v3_pool_helper is arb.swap_pools[1]
                            if REDUCE_TRIANGLE_ARBS
                            else v3_pool_helper in arb.swap_pools
                        )
                    ]
                )
            )
        finally:
            nonlocal processed_swaps
            processed_swaps += 1
            if VERBOSE_EVENTS:
                print(f"[EVENT] Processed {processed_swaps} swaps")

    def process_new_v2_pool_event(message: dict) -> None:
        event_data = message["params"]["result"]["data"]
        token0_address = message["params"]["result"]["topics"][1]
        token1_address = message["params"]["result"]["topics"][2]

        pool_address, _ = eth_abi.decode(
            [
                "address",
                "uint256",
            ],
            bytes.fromhex(event_data[2:]),
        )

        # print(f"New V2 pool: {pool_address}")
        # TODO: add tokens and pool to arb-builder

    def process_new_v3_pool_event(message: dict) -> None:
        event_data = message["params"]["result"]["data"]
        token0_address = message["params"]["result"]["topics"][1]
        token1_address = message["params"]["result"]["topics"][2]
        fee = message["params"]["result"]["topics"][3]

        _, pool_address = eth_abi.decode(
            [
                "int24",
                "address",
            ],
            bytes.fromhex(event_data[2:]),
        )

        # print(f"New V3 pool: {pool_address}")
        # TODO: add tokens and pool to arb-builder
        
    def process_curve_token_exchange_event(
        message: dict, removed: bool = False
    ):
        curve_pool_helper = CurveStableswapPool(message["result"]["address"])
        if curve_pool_helper is None:
            return
        else:
            assert isinstance(
                curve_pool_helper, degenbot.CurveStableswapPool
            )
        curve_pool_helper.auto_update()
        logger.info(f"Updated Curve pool: {curve_pool_helper}")

    _TOPICS = {
        w3.keccak(
            text="Sync(uint112,uint112)",
        ).hex(): {
            "name": "Uniswap V2: SYNC",
            "process_func": process_sync_event,
        },
        w3.keccak(
            text="Mint(address,address,int24,int24,uint128,uint256,uint256)"
        ).hex(): {
            "name": "Uniswap V3: MINT",
            "process_func": process_mint_event,
        },
        w3.keccak(
            text="Burn(address,int24,int24,uint128,uint256,uint256)"
        ).hex(): {
            "name": "Uniswap V3: BURN",
            "process_func": process_burn_event,
        },
        w3.keccak(
            text="Swap(address,address,int256,int256,uint160,uint128,int24)"
        ).hex(): {
            "name": "Uniswap V3: SWAP",
            "process_func": process_swap_event,
        },
        w3.keccak(text="PairCreated(address,address,address,uint256)").hex(): {
            "name": "Uniswap V2: POOL CREATED",
            "process_func": process_new_v2_pool_event,
        },
        w3.keccak(
            text="PoolCreated(address,address,uint24,int24,address)"
        ).hex(): {
            "name": "Uniswap V3: POOL CREATED",
            "process_func": process_new_v3_pool_event,
        },
        w3.keccak(
            text="TokenExchange(address,int128,uint256,int128,uint256)"
        ): {
            "name": "Curve V1: TOKEN EXCHANGE",
            "process_func": process_curve_token_exchange_event,
        },
        w3.keccak(
            text="TokenExchangeUnderlying(address,int128,uint256,int128,uint256)"
        ): {
            "name": "Curve V1: TOKEN EXCHANGE UNDERLYING",
            "process_func": process_curve_token_exchange_event,
        },
        w3.keccak(
            text="AddLiquidity(address,uint256[4],uint256[4],uint256,uint256)"
        ): {
            "name": "Curve V1: ADD LIQUIDITY (4 coins)",
            "process_func": process_curve_token_exchange_event,
        },
        w3.keccak(
            text="AddLiquidity(address,uint256[3],uint256[3],uint256,uint256)"
        ): {
            "name": "Curve V1: ADD LIQUIDITY (3 coins)",
            "process_func": process_curve_token_exchange_event,
        },
        w3.keccak(
            text="AddLiquidity(address,uint256[2],uint256[2],uint256,uint256)"
        ): {
            "name": "Curve V1: ADD LIQUIDITY (2 coins)",
            "process_func": process_curve_token_exchange_event,
        },
        w3.keccak(
            text="RemoveLiquidity(address,uint256[4],uint256[4],uint256)"
        ): {
            "name": "Curve V1: REMOVE LIQUIDITY (4 coins)",
            "process_func": process_curve_token_exchange_event,
        },
        w3.keccak(
            text="RemoveLiquidity(address,uint256[3],uint256[3],uint256)"
        ): {
            "name": "Curve V1: REMOVE LIQUIDITY (3 coins)",
            "process_func": process_curve_token_exchange_event,
        },
        w3.keccak(
            text="RemoveLiquidity(address,uint256[2],uint256[2],uint256)"
        ): {
            "name": "Curve V1: REMOVE LIQUIDITY (2 coins)",
            "process_func": process_curve_token_exchange_event,
        },
        w3.keccak(
            text="RemoveLiquidityOne(address,uint256,uint256)"
        ): {
            "name": "Curve V1: REMOVE LIQUIDITY (1 coin)",
            "process_func": process_curve_token_exchange_event,
        },
        w3.keccak(
            text="RemoveLiquidityImbalance(address,uint256[4],uint256[4],uint256,uint256)"
        ): {
            "name": "Curve V1: REMOVE LIQUIDITY IMBALANCE (4 coins)",
            "process_func": process_curve_token_exchange_event,
        },
        w3.keccak(
            text="RemoveLiquidityImbalance(address,uint256[3],uint256[3],uint256,uint256)"
        ): {
            "name": "Curve V1: REMOVE LIQUIDITY IMBALANCE (3 coins)",
            "process_func": process_curve_token_exchange_event,
        },
        w3.keccak(
            text="RemoveLiquidityImbalance(address,uint256[2],uint256[2],uint256,uint256)"
        ): {
            "name": "Curve V1: REMOVE LIQUIDITY IMBALANCE (2 coins)",
            "process_func": process_curve_token_exchange_event,
        },
    }

    async for websocket in websockets.client.connect(
        uri=NODE_WEBSOCKET_URI,
        ping_timeout=None,
        max_queue=None,
    ):
        # reset the status and first block every time we start a new websocket connection
        status_events = False
        first_event_block = None

        try:
            await websocket.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "eth_subscribe",
                        "params": ["logs", {}],
                    }
                )
            )
            subscribe_result = json.loads(await websocket.recv())
        except asyncio.exceptions.CancelledError:
            return
        except Exception as e:
            print("event_watcher reconnecting...")
            print(e)
            print(type(e))
            continue
        else:
            print(subscribe_result)
            status_events = True

        while True:
            # process the queue completely if bot is "live"
            # does not yield to the event loop,
            # coroutines will remain suspended until the queue is empty
            if status_live and event_queue:
                message = event_queue.popleft()
                topic0 = message["params"]["result"]["topics"][0]

                # process the message for the associated event
                try:
                    _TOPICS[topic0]["process_func"](message)
                except KeyError:
                    continue
                else:
                    if VERBOSE_PROCESSING:
                        print(
                            f"processed {_TOPICS[topic0]['name']} event - {len(event_queue)} remaining"
                        )

            try:
                message = json.loads(await websocket.recv())
            except asyncio.exceptions.CancelledError:
                return
            except websockets.exceptions.WebSocketException as e:
                print("(watch_events) (WebSocketException)...")
                print(f"Latency: {websocket.latency}")
                sys.exit()
                print("watch_events reconnecting...")
                print(e)
                break
            except Exception as e:
                import traceback

                traceback.print_exc()
                sys.exit()

            if not first_event_block:
                first_event_block = int(
                    message["params"]["result"]["blockNumber"],
                    16,
                )
                print(f"First event block: {first_event_block}")

            received_events += 1
            if VERBOSE_EVENTS and received_events % 1000 == 0:
                print(f"[EVENTS] Received {received_events} total events")

            try:
                topic0 = message["params"]["result"]["topics"][0]
            except IndexError:
                # ignore anonymous events (no topic0)
                continue
            except Exception as e:
                print(f"(event_watcher): {e}")
                print(type(e))
                print(f"message={message}")
                continue
            else:
                event_queue.append(message)
def check_failures(arb_helper):
    global arb_simulations

    arb_id = arb_helper.id

    try:
        arb_simulations[arb_id]
    except KeyError:
        return

    try:
        if (
            arb_simulations[arb_id]["simulations"]
            >= SIMULATION_CUTOFF_MIN_ATTEMPTS
            and (
                arb_simulations[arb_id]["failures"]
                / arb_simulations[arb_id]["simulations"]
            )
            >= SIMULATION_CUTOFF_FAIL_THRESHOLD
        ):
            old_state = arb_helper.pool_states.copy()
            # do a final "forced" update to check for outdated pool state. If `auto_update` returns True,
            # one or more of the pools tracked by the arb helper were updated
            if arb_helper.auto_update(
                override_update_method="polling",
                block_number=newest_block,
            ):
                new_state = arb_helper.pool_states.copy()
                # print(
                #     f"CANCELLED BLACKLIST, ARB {arb_helper} ({arb_id}) WAS OUTDATED"
                # )
                # print(f"\told state: {old_state}")
                # print(f"\tnew state: {new_state}")
                arb_simulations[arb_id]["simulations"] = 0
                arb_simulations[arb_id]["failures"] = 0
            else:
                print(f"BLACKLISTED ARB: {arb_helper}, ID: {arb_id}")
                # print(f"LAST STATE:")
                # print(arb_helper.best)
                inactive_arbs.pop(arb_id)
                arb_simulations.pop(arb_id)
                BLACKLISTED_ARBS.append(arb_id)
                with open("arbitrum_blacklisted_arbs.json", "w") as file:
                    json.dump(BLACKLISTED_ARBS, file, indent=2)
    except Exception as e:
        print(f"check_failures: {e}")
        
        
async def watch_new_blocks():
    """
    Watches the websocket for new blocks, updates the base fee for the last block, scans
    transactions and removes them from the pending tx queue, and prints various messages
    """

    print("Starting block watcher loop")

    global first_new_block
    global newest_block
    global newest_block_timestamp
    global last_base_fee
    global next_base_fee
    global status_new_blocks
    global AVERAGE_BLOCK_TIME

    # a rolling window of the last 100 block deltas, seeded with an initial value
    block_times = deque(
        [time.time() - AVERAGE_BLOCK_TIME],
        maxlen=100,
    )

    async for websocket in websockets.client.connect(
        uri=NODE_WEBSOCKET_URI,
        ping_timeout=None,
        max_queue=None,
    ):
        # reset the first block and status every time we connect or reconnect
        status_new_blocks = False
        first_new_block = 0

        try:
            await websocket.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "eth_subscribe",
                        "params": ["newHeads"],
                    }
                )
            )
            subscribe_result = json.loads(await websocket.recv())
        except asyncio.exceptions.CancelledError:
            return
        except websockets.exceptions.WebSocketException as e:
            print("watch_new_blocks reconnecting...")
            print(e)
            continue
        except Exception as e:
            print("watch_new_blocks reconnecting...")
            print(e)
            continue
        else:
            print(subscribe_result)
            status_new_blocks = True

        while True:
            try:
                message = json.loads(await websocket.recv())
            except asyncio.exceptions.CancelledError:
                return
            except websockets.exceptions.WebSocketException as e:
                print("watch_new_blocks reconnecting...")
                print(e)
                break
            except Exception as e:
                print(f"(watch_new_blocks) websocket.recv(): {e}")
                print(type(e))
                break

            if VERBOSE_TIMING:
                logger.debug("starting watch_new_blocks")
                # print("starting watch_new_blocks")
                start = time.monotonic()

            newest_block = int(
                message.get("params").get("result").get("number"),
                16,
            )
            newest_block_timestamp = int(
                message.get("params").get("result").get("timestamp"),
                16,
            )

            block_times.append(newest_block_timestamp)
            AVERAGE_BLOCK_TIME = (block_times[-1] - block_times[0]) / (
                len(block_times) - 1
            )

            if not first_new_block:
                first_new_block = newest_block
                print(f"First full block: {first_new_block}")

            last_base_fee, next_base_fee = w3.eth.fee_history(1, newest_block)[
                "baseFeePerGas"
            ]

            if VERBOSE_BLOCKS:
                print(
                    f"[{newest_block}] "
                    + f"base fee: {last_base_fee/(10**9):.2f}/{next_base_fee/(10**9):.2f} "
                    f"(+{time.time() - newest_block_timestamp:.2f}s) "
                )

            if VERBOSE_TIMING:
                # print(
                #     f"watch_new_blocks completed in {time.monotonic() - start:0.4f}s"
                # )
                logger.debug(
                    f"watch_new_blocks completed in {time.monotonic() - start:0.4f}s"
                )


def shutdown():
    """
    Cancel all tasks in the `all_tasks` set
    """

    logger.info(f"\nCancelling tasks")
    for task in [t for t in all_tasks if not (t.done() or t.cancelled())]:
        task.cancel()
        

# --- Main Execution ---

if __name__ == "__main__":
    if not DRY_RUN:
        print(
            "\n"
            "\n***************************************"
            "\n*** DRY RUN DISABLED - BOT IS LIVE! ***"
            "\n***************************************"
            "\n"
        )
    w3 = web3.Web3(web3.HTTPProvider(NODE_HTTP_URI))
    BLACKLISTED_TOKENS = []
    for filename in ["arbitrum_blacklisted_tokens.json"]:
        try:
            with open(filename) as file:
                BLACKLISTED_TOKENS.extend(json.load(file))
        except FileNotFoundError:
            with open(filename, "w") as file:
                json.dump(BLACKLISTED_TOKENS, file, indent=2)
    print(f"Found {len(BLACKLISTED_TOKENS)} blacklisted tokens")

    BLACKLISTED_ARBS = []
    for filename in ["arbitrum_blacklisted_arbs.json"]:
        try:
            with open(filename) as file:
                BLACKLISTED_ARBS.extend(json.load(file))
        except FileNotFoundError:
            with open(filename, "w") as file:
                json.dump(BLACKLISTED_ARBS, file, indent=2)
    print(f"Found {len(BLACKLISTED_ARBS)} blacklisted arbs")

    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    start = time.perf_counter()
    asyncio.run(main())
    print(f"Completed in {time.perf_counter() - start:.2f}s")
