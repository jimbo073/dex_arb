import asyncio
import functools
import json
import logging
import multiprocessing
import os
import signal
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import brownie  # type: ignore
import eth_abi
import eth_account
import web3
import websockets.client
from web3._utils.events import get_event_data
from web3._utils.filters import construct_event_filter_params

import degenbot as bot
from degenbot import LiquidityPool, V3LiquidityPool, UniswapLpCycle

BROWNIE_NETWORK = "arbitrum-local"
BROWNIE_ACCOUNT = "arbitrum_bot"

NODE_WEBSOCKET_URI = "ws://localhost:8548"
NODE_HTTP_URI = "http://localhost:8547"

ARBISCAN_API_KEY = "EDITME"

ARB_CONTRACT_ADDRESS = "EDITME"
WETH_ADDRESS = "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1"

MIN_PROFIT_ETH = int(0.000001 * 10**18)

DRY_RUN = True

VERBOSE_ACTIVATION = False
VERBOSE_BLOCKS = False
VERBOSE_EVENTS = False
VERBOSE_GAS_ESTIMATES = False
VERBOSE_PROCESSING = False
VERBOSE_SIMULATION = False
VERBOSE_TIMING = True
VERBOSE_UPDATES = False
VERBOSE_WATCHDOG = True

# require min. number of simulations before evaluating the cutoff threshold
SIMULATION_CUTOFF_MIN_ATTEMPTS = 10
# arbs that fail simulations greater than this percentage will be added to a blacklist
SIMULATION_CUTOFF_FAIL_THRESHOLD = 0.99

AVERAGE_BLOCK_TIME = 0.25
LATE_BLOCK_THRESHOLD = 10

# if True, triangle arbs will only be processed if the middle leg is updated.
REDUCE_TRIANGLE_ARBS = True


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
        signal.SIGHUP,
        signal.SIGTERM,
        signal.SIGINT,
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


async def activate_arbs():
    """
    TBD
    """

    global process_pool

    async def _activate_and_record(arbs: List):
        global active_arbs
        global inactive_arbs

        loop = asyncio.get_running_loop()

        _tasks = [
            loop.run_in_executor(
                executor=process_pool,
                func=arb_helper.calculate_arbitrage_return_best,
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
                print(f"(_activate_and_record as_completed) catch-all: {e}")
            else:
                try:
                    arb_helper = inactive_arbs[arb_id]
                except KeyError:
                    continue
                else:
                    arb_helper.best = best

                    if test_onchain_arb_gas(arb_helper):
                        record_activation(arb_helper)
                        active_arbs[arb_helper.id] = inactive_arbs.pop(
                            arb_helper.id
                        )
                    else:
                        arb_helper.clear_best()
                        check_failures(arb_helper)

    async def _reactivate(arbs: List):
        loop = asyncio.get_running_loop()

        _tasks = [
            loop.run_in_executor(
                executor=process_pool,
                func=arb_helper.calculate_arbitrage_return_best,
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
                print(f"(_reactivate as_completed) catch-all: {e}")
                print(type(e))
            else:
                try:
                    arb_helper = active_arbs[arb_id]
                except KeyError:
                    pass
                else:
                    arb_helper.best = best

    global active_arbs
    global inactive_arbs
    global arbs_to_activate

    try:
        while not status_live:
            await asyncio.sleep(AVERAGE_BLOCK_TIME)

        print("STARTING ARB ACTIVATION")

        _inactive_arbs = list(inactive_arbs.values())

        while _inactive_arbs:
            try:
                length = min(1000, len(_inactive_arbs))
                await _reactivate(_inactive_arbs[:length])
                _inactive_arbs = _inactive_arbs[length:]
            except Exception as e:
                print(f"{type(e)}: {e}")

        print("REACTIVATED ALL ARBS")

        inactive_arbs_start_position = None

        while True:
            await asyncio.sleep(AVERAGE_BLOCK_TIME)

            if status_paused:
                continue

            # Activate "fresh" arbs identified by the block watcher first.
            # If none are found, process a sample from the stale arbs.
            if arbs_to_activate:
                arbs_to_process = list(arbs_to_activate)
                arbs_to_activate.clear()
            else:
                if inactive_arbs_start_position is None:
                    inactive_arbs_start_position = 0
                else:
                    inactive_arbs_start_position = end_position
                end_position = min(
                    len(_inactive_arbs), inactive_arbs_start_position + 10
                )

                arbs_to_process = _inactive_arbs[
                    inactive_arbs_start_position:end_position
                ]

                # if we've reached the end, rebuild the list and restart at the first position
                if end_position == len(_inactive_arbs):
                    _inactive_arbs = list(inactive_arbs.values())
                    inactive_arbs_start_position = None

            await _activate_and_record(arbs_to_process)

    except asyncio.exceptions.CancelledError:
        return


async def calculate_arb_in_pool(arbs: Iterable):
    """
    TBD
    """

    loop = asyncio.get_running_loop()

    global process_pool

    _tasks = [
        loop.run_in_executor(
            executor=process_pool,
            func=arb_helper.calculate_arbitrage_return_best,
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


async def execute_arb_with_rpc(
    arbitrage: bot.arbitrage.uniswap_lp_cycle.UniswapLpCycle,
    nonce: int,
) -> bool:
    if VERBOSE_TIMING:
        start = time.monotonic()
        # print("starting execute_arb_with_rpc")
        logger.debug("starting execute_arb_with_rpc")

    tx_params = {
        "from": bot_account.address,
        "chainId": brownie.chain.id,
        # Set gas high for the estimate_gas RPC call
        # This is overwritten after estimation
        "gas": 50_000_000,
        "nonce": nonce,
        "maxFeePerGas": int(1.25 * next_base_fee),
        "maxPriorityFeePerGas": 0,
        "value": 0,
    }

    try:
        arb_payloads = arbitrage.generate_payloads(
            from_address=arb_contract.address
        )
    except bot.exceptions.ArbitrageError as e:
        # print(f"(execute_arb_with_rpc) (generate_payloads): {e}")
        arbitrage.clear_best()
        return False

    # simulate the TX
    try:
        simulated_gas_use = (
            w3.eth.contract(
                address=arb_contract.address,
                abi=arb_contract.abi,
            )
            .functions.execute_payloads(arb_payloads)
            .estimate_gas(
                tx_params,
                block_identifier="pending",
            )
        )
    except web3.exceptions.ContractLogicError as e:
        # if the simulation fails, clear it
        print(f"(execute_arb_with_rpc) (simulate): {e}")
        arbitrage.clear_best()
        return False
    except ValueError as e:
        print(f"(execute_arb_with_rpc) (ValueError): {e}")
        # print(arbitrage)
        return False
    except Exception as e:
        print(f"(execute_arb_with_rpc) (catch-all): {e}")
        print(type(e))
        return False
    else:
        arbitrage.gas_estimate = simulated_gas_use
        print(f"{arbitrage} gas use = {simulated_gas_use}")

    gas_fee = simulated_gas_use * next_base_fee
    arb_net_profit = arbitrage.best["profit_amount"] - gas_fee
    arbitrage.gas_estimate = simulated_gas_use

    if arb_net_profit > MIN_PROFIT_ETH:
        print()
        print(f"Arb    : {arbitrage}:{arbitrage.id}")
        print(f'Input  : {arbitrage.best["swap_amount"]/10**18:0.6f} ETH')
        print(f"Profit : {arbitrage.best['profit_amount']/10**18:0.6f} ETH")
        print(
            f"Gas    : {gas_fee/(10**18):0.6f} ETH ({arbitrage.gas_estimate} gas estimate)"
        )
        print(f"Net    : {arb_net_profit/(10**18):0.6f} ETH")

        if DRY_RUN:
            return False

        tx_params.update({"gas": int(1.1 * simulated_gas_use)})

        print("*** EXECUTING ONCHAIN ARB (RPC) ***")
        try:
            signed_tx = (
                eth_account.Account.from_key(bot_account.private_key)
                .sign_transaction(
                    w3.eth.contract(
                        address=arb_contract.address,
                        abi=arb_contract.abi,
                    )
                    .functions.execute_payloads(arb_payloads)
                    .buildTransaction(tx_params)
                )
                .rawTransaction
            )
            print(f"Sending raw TX at block {newest_block}")
            result = w3.eth.send_raw_transaction(signed_tx)
        except Exception as e:
            print(f"(execute_arb_with_rpc) (send_raw_transaction): {e}")
            print(type(e))
            return False
        else:
            print(f"TX: {result.hex()}")
            return True

    if VERBOSE_TIMING:
        logger.debug(
            f"send_arb_via_relay completed in {time.monotonic() - start:0.4f}s"
        )
        # print(
        #     f"send_arb_via_relay completed in {time.monotonic() - start:0.4f}s"
        # )

    return False


def handle_task_exception(loop, context):
    """
    By default, do nothing
    """


async def load_arbs():
    print("Starting arb loading function")

    global degenbot_lp_helpers
    global active_arbs
    global inactive_arbs
    global arb_simulations
    global status_live

    # liquidity_pool_and_token_addresses will filter out any blacklisted addresses, so helpers should draw from this as the "official" source of truth
    liquidity_pool_data = {}
    for filename in [
        "arbitrum_camelot_lps.json",
        "arbitrum_sushiswap_lps.json",
        "arbitrum_uniswapv3_lps.json",
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
        "arbitrum_arbs_2pool.json",
        "arbitrum_arbs_3pool.json",
    ]:
        with open(filename) as file:
            for arb_id, arb in json.load(file).items():
                passed_checks = True
                if arb_id in BLACKLISTED_ARBS:
                    passed_checks = False
                for pool_address in arb.get("path"):
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

        V3LP = w3.eth.contract(abi=bot.uniswap.v3.abi.UNISWAP_V3_POOL_ABI)

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
                pool_helper = bot.uniswap.v2.liquidity_pool.CamelotLiquidityPool(
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

            else:
                raise Exception("Could not identify pool type!")
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

    _weth_balance = weth.balanceOf(arb_contract.address)

    degenbot_erc20token_weth = bot.Erc20Token(WETH_ADDRESS)

    try:
        with open("arbitrum_activated_arbs.json", "r") as file:
            _activated_arbs = json.load(file)
    except FileNotFoundError:
        _activated_arbs = []

    # build a dict of arb helpers, keyed by arb ID
    inactive_arbs = {
        arb_id: bot.UniswapLpCycle(
            input_token=degenbot_erc20token_weth,
            swap_pools=swap_pools,
            max_input=_weth_balance,
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


def record_activation(arb_helper):
    try:
        with open("arbitrum_activated_arbs.json", "r") as file:
            _activated_arbs = set(json.load(file))
    except FileNotFoundError:
        _activated_arbs = set()

    _activated_arbs.add(arb_helper.id)

    with open("arbitrum_activated_arbs.json", "w") as file:
        json.dump(list(_activated_arbs), file, indent=2)


def shutdown():
    """
    Cancel all tasks in the `all_tasks` set
    """

    logger.info(f"\nCancelling tasks")
    for task in [t for t in all_tasks if not (t.done() or t.cancelled())]:
        task.cancel()


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
    try:
        while True:
            while not status_live:
                await asyncio.sleep(AVERAGE_BLOCK_TIME)

            await asyncio.sleep(5)

            profitable_arbs = sorted(
                [
                    arb_helper
                    for arb_helper in active_arbs.values()
                    if arb_helper.best["swap_amount"]
                    if arb_helper.best["profit_amount"] > MIN_PROFIT_ETH
                ],
                key=lambda arb_helper: arb_helper.best["profit_amount"],
                reverse=True,
            )

            if profitable_arbs:
                # execute the most profitable arb
                arb_helper = profitable_arbs[0]
                if await execute_arb_with_rpc(
                    arbitrage=arb_helper,
                    nonce=bot_account.nonce,
                ):
                    arb_helper.clear_best()

    except asyncio.exceptions.CancelledError:
        return


def test_onchain_arb_gas(arb_helper) -> bool:
    """
    Calculates the gas use for the specified arb against a particular block

    Return value: bool indicating if the gas was successfully estimated
    """

    def get_gas_estimate(
        payloads: list,
        tx_params: dict,
        arb_id=None,
    ) -> Tuple[bool, int]:
        global arb_simulations

        if VERBOSE_TIMING:
            start = time.monotonic()
            # print("starting test_gas")
            logger.debug("starting test_gas")

        try:
            arb_simulations[arb_id]["simulations"] += 1
            gas_estimate = (
                w3.eth.contract(
                    address=arb_contract.address,
                    abi=arb_contract.abi,
                )
                .functions.execute_payloads(payloads)
                .estimate_gas(tx_params)
            )
        except web3.exceptions.ContractLogicError as e:
            # print(f"(get_gas_estimate)({type(e)}: {e}")
            arb_simulations[arb_id]["failures"] += 1
            success = False
            if VERBOSE_SIMULATION:
                print(f"FAILED SIM: {inactive_arbs[arb_id]}")
                print(
                    f"({arb_simulations[arb_id]['simulations']} total, {arb_simulations[arb_id]['failures']} failed)"
                )
                print(f"{arb_id=}")
                print(f"FAILURE: {e}")
                # print(f"PAYLOAD: {inactive_arbs[arb_id].best}")
        except Exception as e:
            # pretend the simulation never happened if some other error occured
            arb_simulations[arb_id]["simulations"] -= 1
            # print(f"(get_gas_estimate)({type(e)}: {e}")
            success = False
        else:
            success = True

        if VERBOSE_TIMING:
            logging.debug(
                f"test_gas completed in {time.monotonic() - start:0.4f}s"
            )
            # print(f"test_gas completed in {time.monotonic() - start:0.4f}s")

        return (
            success,
            gas_estimate if success else 0,
        )

    if VERBOSE_TIMING:
        start = time.monotonic()
        # print("starting test_onchain_arb")
        logger.debug("starting test_onchain_arb")

    tx_params = {
        "from": bot_account.address,
        "chainId": brownie.chain.id,
        "nonce": bot_account.nonce,
        "gas": 50_000_000,
        "maxFeePerGas": int(1.25 * next_base_fee),
        "maxPriorityFeePerGas": 0,
    }

    try:
        arb_payloads = arb_helper.generate_payloads(
            from_address=arb_contract.address
        )
    except Exception as e:
        # print(f"test gas (generate_payloads): {e}")
        return False

    success, gas_estimate = get_gas_estimate(
        arb_payloads,
        tx_params,
        arb_id=arb_helper.id,
    )

    if success:
        arb_helper.gas_estimate = gas_estimate
        if VERBOSE_GAS_ESTIMATES:
            print(f"Gas estimate for arb {arb_helper}: {gas_estimate}")

    if VERBOSE_TIMING:
        # print(
        #     f"test_onchain_arb completed in {time.monotonic() - start:0.4f}s"
        # )
        logger.debug(
            f"test_onchain_arb completed in {time.monotonic() - start:0.4f}s"
        )

    return success


async def track_balance():
    global weth_balance
    weth_balance = 0

    while True:
        await asyncio.sleep(10 * AVERAGE_BLOCK_TIME)

        try:
            balance = weth.balanceOf(arb_contract.address)
        except asyncio.exceptions.CancelledError:
            return
        except Exception as e:
            print(f"(track_balance): {e}")
        else:
            if weth_balance != balance:
                weth_balance = balance
                for arb in inactive_arbs.values():
                    arb.max_input = weth_balance
                for arb in active_arbs.values():
                    arb.max_input = weth_balance

                print()
                print(f"Updated balance: {weth_balance/(10**18):.3f} WETH")
                print()


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



if __name__ == "__main__":
    if not DRY_RUN:
        print(
            "\n"
            "\n***************************************"
            "\n*** DRY RUN DISABLED - BOT IS LIVE! ***"
            "\n***************************************"
            "\n"
        )

    os.environ["ARBISCAN_TOKEN"] = ARBISCAN_API_KEY

    # Create a reusable web3 object to communicate with the node
    # (no arguments to provider will default to localhost on the default port)
    w3 = web3.Web3(web3.HTTPProvider(NODE_HTTP_URI))

    try:
        brownie.network.connect(BROWNIE_NETWORK)
    except:
        sys.exit(
            "Could not connect! Verify your Brownie network settings using 'brownie networks list'"
        )

    try:
        bot_account = brownie.accounts.load(BROWNIE_ACCOUNT)
    except:
        sys.exit(
            "Could not load account! Verify your Brownie account settings using 'brownie accounts list'"
        )

    arb_contract = brownie.Contract.from_abi(
        name="",
        address=ARB_CONTRACT_ADDRESS,
        abi=json.loads(
            """
            [{"stateMutability": "payable", "type": "constructor", "inputs": [], "outputs": []}, {"stateMutability": "payable", "type": "function", "name": "execute_payloads", "inputs": [{"name": "payloads", "type": "tuple[]", "components": [{"name": "target", "type": "address"}, {"name": "calldata", "type": "bytes"}, {"name": "value", "type": "uint256"}]}], "outputs": []}, {"stateMutability": "payable", "type": "function", "name": "uniswapV3SwapCallback", "inputs": [{"name": "amount0", "type": "int256"}, {"name": "amount1", "type": "int256"}, {"name": "data", "type": "bytes"}], "outputs": []}, {"stateMutability": "payable", "type": "fallback"}]
            """
        ),
    )

    weth = brownie.Contract.from_explorer(WETH_ADDRESS)

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

    last_base_fee = 1 * 10**9  # overridden on first received block
    next_base_fee = 1 * 10**9  # overridden on first received block
    newest_block = brownie.chain.height  # overridden on first received block
    newest_block_timestamp = int(
        time.time()
    )  # overridden on first received block
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

    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    start = time.perf_counter()
    asyncio.run(main())
    print(f"Completed in {time.perf_counter() - start:.2f}s")