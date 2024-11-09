from datetime import datetime
import datetime
from fractions import Fraction
from pickle import FALSE
import time
import json
from degenbot.src import degenbot
from decimal import Decimal as dec
from urllib3.exceptions import HTTPError
import logging
import requests
from eth_account.messages import encode_defunct
from web3 import Web3
from eth_abi import decode
from decimal import Decimal
import aiohttp
import asyncio
import time
from scipy import optimize

rpc = "https://floral-crimson-patina.arbitrum-mainnet.quiknode.pro/347691b7280c64c57237f77f7c8988972dde604d/"
public_key = "0xA791336Eea54cA199e96123e5e0774a908268A65"
private_key = "e61907a13a06d8e702a76b7651541398fa66966f91527c4f560bb7b3adadd539"
web3 = Web3(Web3.HTTPProvider(rpc))
degenbot.config.set_web3(web3)
# Definiere die maximale Anzahl an gleichzeitigen Requests und die Wartezeit zwischen den Requests (in Sekunden)
# MAX_CONCURRENT_REQUESTS = 1  # Anzahl gleichzeitiger Anfragen
# DELAY_BETWEEN_REQUESTS = 1.5  # Sekunden zwischen den Requests
# Globale Header-Konfiguration für alle Anfragen

UNISWAP_V2_FACTORY = "0xf1D7CC64Fb4452F05c498126312eBE29f30Fbcf9"
UNI_V2_FACTORY_INIT_HASH = "96e8ac4277198ff8b6f785478aa9a39f403cb768dd02cbee326c3e7da348845f"
# Gaslimit-Konfiguration
BLACKLISTED_ARBS = []
BLACKLISTED_TOKENS = []
BLACKLISTED_POOLS = []


# Hauptfunktion, die die Anfragen ausführt
async def main():
    liquidity_pool_data = {}
    for filename in [
        r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_camelotv2.json",
        r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_curvev1_registry.json",
        r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_sushiswapv2.json",
        r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_uniswapv2.json",
        r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_sushiswapv3.json",
        r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\lp_fetcher\arbitrum_lps_uniswapv3.json",
    ]:
        with open(filename, encoding="utf-8") as file:
            for pool in json.load(file):
                if pool.get("pool_address") is None:
                    continue
                if pool["pool_address"] in BLACKLISTED_POOLS:
                   continue
                if "token0" in pool:
                    if (
                        pool["token0"] in BLACKLISTED_TOKENS
                        or pool["token1"] in BLACKLISTED_TOKENS
                    ):
                        continue
                if "coin_addresses" in pool:
                    for coin_address in pool["coin_addresses"]:
                        if coin_address in BLACKLISTED_TOKENS:
                            continue
                if "underlying_coin_addresses" in pool:
                    for coin_address in pool[
                        "underlying_coin_addresses"
                    ]:
                        if coin_address in BLACKLISTED_TOKENS:
                            continue
                liquidity_pool_data[pool["pool_address"]] = pool
              
    print(f"Found {len(liquidity_pool_data)} pools")
    
    arb_paths = []
    for filename in [
       r"C:\Users\PC\Projects\dex_arb\arbitrum\arbitrage_paths\arbitrum_arbs_3pool_type.json",
    ]:
        with open(filename) as file:
            for arb_id, arb in json.load(file).items():
                passed_checks = True
                if arb_id in BLACKLISTED_ARBS:
                    passed_checks = False
                for pool_address in arb.get("path"):
                    if pool_address:
                        continue
                    if not liquidity_pool_data.get(pool_address):
                        passed_checks = False
                if passed_checks:
                    arb_paths.append(arb)
    print(f"Found {len(arb_paths)} arb paths")
    
    for arb in arb_paths:
        for pool_address in arb.get("path"):
            if liquidity_pool_data[pool_address]["type"] == "UniswapV2" or "SushiswapV2":
                lp = degenbot.UniswapV2Pool(web3.to_checksum_address(pool_address))
            if liquidity_pool_data[pool_address]["type"] == "UniswapV3" or "SushiswapV3":
                v3_lp = degenbot.UniswapV3Pool(web3.to_checksum_address(pool_address))
            if liquidity_pool_data[pool_address]["type"] == "CamelotV2":
                camelot_lp = degenbot.CamelotLiquidityPool(pool_address)
            if liquidity_pool_data[pool_address]["type"] == "CurveV1":
                curve_lp = degenbot.CurveStableswapPool(pool_address)
            if lp:
                print(lp.tokens)
            if curve_lp:
                print(curve_lp.tokens)

        # # uniswap pool amounts out
        # v3_tkns_out_from_tkns_in = Decimal(int(v3_lp.calculate_tokens_out_from_tokens_in(
        #     token_in=lp.token0,
        #     token_in_quantity=int(1 * 10**v3_lp.token0.decimals)
        # )) / Decimal(10**v3_lp.token1.decimals))

        # v2_tkns_out_from_tkns_in = Decimal(int(lp.calculate_tokens_out_from_tokens_in(
        #     token_in=lp.token0,
        #     token_in_quantity=int(1 * 10**lp.token0.decimals)
        # )) / Decimal(10**lp.token1.decimals))

        # # v2_lp_helper = UniswapV2LiquidityPoolManager(UNISWAP_V2_FACTORY,42161)
        # # pool = v2_lp_helper.get_pool("0xf01fCb630aB3b31063bDD66A640f92DB4fAd4044")  
        
        # # v3_lp_helper = UniswapV3LiquidityPoolManager("0x1F98431c8aD98523631AE4a59f267346ea31F984","0x6C9FC64A53c1b71FB3f9Af64d1ae3A4931A5f4E9",42161,r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\arbitrum_v3_liquidity_snapshot.json")
        # # v3_pool = v3_lp_helper.get_pool("0x58d7283E1cCa23ad98d4307b9e899eF3827ce7eC")
        # # aggregator_v6 = OneInchAggregator(42161,"BMYEt9qskUTWzCtyPrGKOUMEmgkX8iOK",FALSE)
        # # print(aggregator_v6.address)
        

# Starte das asynchrone Programm
if __name__ == "__main__":
    asyncio.run(main())
