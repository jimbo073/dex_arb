import traceback
import web3
from CombinedLiquidityPool import CombinedLiquidityPool
import degenbot
import json
import time
import asyncio
import sys
from degenbot.uniswap.types import UniswapV3BitmapAtWord,UniswapV3LiquidityAtTick
sys.path.append(r'C:\Users\PC\Projects')

WETH_ADDRESS = "0x82af49447d8a07e3bd95bd0d56f35241523fbab1"
DAI_ADDRESS = "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1"
USDC_ADDRESS = "0xaf88d065e77c8cc2239327c5edb3a432268e5831"
USDT_ADDRESS = "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9"
WBTC_ADDRESS = "0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f"

CURVE_USDC_USDT_ADDRESS = "0x7f90122bf0700f9e7e1f688fe926940e8839f353"

UNISWAP_V2_POOL_ADDRESS = "0xcb0e5bfa72bbb4d16ab5aa0c60601c438f04b4ad"
#UNISWAP_V2_WETH_WBTC_ADDRESS = "0x8c1D83A25eE2dA1643A5d937562682b1aC6C856B"
#UNISWAP_V2_USDC_USDT_ADDRESS = "0xdeae1ff5282d83aadd42f85c57f6e69a037bf7cd"

UNISWAP_V3_POOL_ADDRESS = "0x641c00a822e8b671738d32a431a4fb6074e5c79d"

_web3 = web3.Web3(web3.HTTPProvider("http://localhost:8547"))

degenbot.config.set_web3(_web3)

async def test_arb_calculation():
    try:   
        uniswap_v2_usdc_usdt_lp = degenbot.SushiswapV2Pool(UNISWAP_V2_POOL_ADDRESS)
        univ2pool = degenbot.UniswapV2Pool("0xd04bc65744306a5c149414dd3cd5c984d9d3470d")
        camelot_lp = degenbot.CamelotLiquidityPool(_web3.to_checksum_address("0x97b192198d164c2a1834295e302b713bc32c8f1d"))
       
        # uniswap_v2_weth_usdt_lp = LiquidityPool(UNISWAP_V2_WETH_USDT_ADDRESS)
        liquidity_snapshot = {}

        # update the snapshot to the block before our event watcher came online
        try:
            with open(r"C:\Users\PC\Projects\dex_arb\arbitrum\cryo_data\arbitrum\arbitrum_v3_liquidity_snapshot.json", "r") as file:
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
                try:
                    snapshot_tick_data:UniswapV3LiquidityAtTick = liquidity_snapshot[pool_address][
                        "tick_data"
                    ]
                except KeyError:
                    snapshot_tick_data = {}

                try:
                    snapshot_tick_bitmap:UniswapV3BitmapAtWord = liquidity_snapshot[pool_address][
                        "tick_bitmap"
                    ]
                except KeyError:
                    snapshot_tick_bitmap = {}
        
        uniswap_v3_usdc_usdt_lp = degenbot.UniswapV3Pool(
            _web3.to_checksum_address("0x42161084d0672e1d3f26a9b53e653be2084ff19c"),
            tick_data=snapshot_tick_data,
            tick_bitmap=snapshot_tick_bitmap)
        uniswap_v3_usdc_usdt_lp2 = degenbot.UniswapV3Pool(
            UNISWAP_V3_POOL_ADDRESS,
            tick_data=snapshot_tick_data,
            tick_bitmap=snapshot_tick_bitmap)
        uniswap_v3_usdc_usdt_lp3 = degenbot.UniswapV3Pool(
            _web3.to_checksum_address("0xc82819f72a9e77e2c0c3a69b3196478f44303cf4"),
            tick_data=snapshot_tick_data,
            tick_bitmap=snapshot_tick_bitmap)
        sushiswap_v3_lp = degenbot.SushiswapV3Pool(
            _web3.to_checksum_address("0x96ada81328abce21939a51d971a63077e16db26e"),
            tick_data=snapshot_tick_data,
            tick_bitmap=snapshot_tick_bitmap,
        )
        # univ2tknout = univ2pool.calculate_tokens_out_from_tokens_in(sushiswap_v3_lp.token0,100*10**18,None)
        # print("univ2tknout", univ2tknout/10**uniswap_v3_usdc_usdt_lp.token1.decimals)
        # sushiv3tknout = sushiswap_v3_lp.calculate_tokens_out_from_tokens_in(sushiswap_v3_lp.token0,100*10**18,None)
        # print("sushiv3tknout", sushiv3tknout/10**uniswap_v3_usdc_usdt_lp.token1.decimals)
        univ3tknout = uniswap_v3_usdc_usdt_lp.calculate_tokens_out_from_tokens_in(sushiswap_v3_lp.token0,1*10**18)
        print("univ3tknout", univ3tknout/10**uniswap_v3_usdc_usdt_lp.token1.decimals)

        # tkn0out = uniswap_v2_usdc_usdt_lp.calculate_tokens_out_from_tokens_in(uniswap_v3_usdc_usdt_lp.token0,100*(10**18),None)
        # print("tkn0out",tkn0out/10**uniswap_v3_usdc_usdt_lp.token1.decimals)
        # camelot_token_out = camelot_lp.calculate_tokens_out_from_tokens_in(
        #     uniswap_v3_usdc_usdt_lp.token0,
        #     10*(10**18),
        #     None
        # )
        # print("camelot_token_out", camelot_token_out/10**camelot_lp.token1.decimals)
        

        # uniswap_v2_weth_dai_lp = LiquidityPool(UNISWAP_V2_WETH_DAI_ADDRESS)
    
        # weth = degenbot.Erc20Token(WETH_ADDRESS)
        # dai = degenbot.Erc20Token(DAI_ADDRESS)
        # wbtc = degenbot.Erc20Token(WBTC_ADDRESS)
        # usdc = degenbot.Erc20Token(USDC_ADDRESS)
        # usdt = degenbot.Erc20Token(USDT_ADDRESS)
        # curve_tripool = degenbot.CurveStableswapPool(address=CURVE_USDC_USDT_ADDRESS,chain_id=_web3.eth.chain_id)
        # tknout = curve_tripool.calculate_tokens_out_from_tokens_in(uniswap_v2_usdc_usdt_lp.token0,uniswap_v2_usdc_usdt_lp.token1,1000000000000000000,None,None)
        # print(tknout)
        combined_pool = CombinedLiquidityPool(
            token_in=uniswap_v2_usdc_usdt_lp.token0,
            token_out=uniswap_v2_usdc_usdt_lp.token1,
            # max_input=1000000000*10**uniswap_v2_usdc_usdt_lp.token0.decimals,
            pools_with_states=[
                # (curve_tripool,None),
                (uniswap_v2_usdc_usdt_lp,None),
                (uniswap_v3_usdc_usdt_lp,None),
                (uniswap_v3_usdc_usdt_lp2,None),
                (uniswap_v3_usdc_usdt_lp3,None),
                (camelot_lp,None),
                (sushiswap_v3_lp,None),
                (univ2pool,None)
                ],
                silent=False
            )
        
       # best_result = combined_pool.calculate_tokens_out_from_tokens_in(100*10**18)
       # print(f"combined: {best_result/10**uniswap_v3_usdc_usdt_lp.token1.decimals}, biggest pool v3: {univ3tknout/10**uniswap_v3_usdc_usdt_lp.token1.decimals}")
       # print(f"diffrence: {(best_result-univ3tknout)/10**uniswap_v3_usdc_usdt_lp.token1.decimals}")
        
        best_brent = combined_pool.calculate_tokens_out_from_tokens_in(100*10**18)
        print(f"[brentq]   combined out      : {best_brent/10**uniswap_v3_usdc_usdt_lp.token1.decimals}")
        #print(f"[brentq]   Δ (brentq – min): {(best_brent-best_result)/10**uniswap_v3_usdc_usdt_lp.token1.decimals}")
        
    except Exception as e:
        print(e)
        traceback.print_exc()
        
asyncio.run(test_arb_calculation())