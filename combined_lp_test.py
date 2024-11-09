import traceback
import web3
from web3.exceptions import Web3RPCError
from CombinedLiquidityPool import CombinedLiquidityPool
from degenbot.src import degenbot 
import time
import asyncio
import sys
sys.path.append(r'C:\Users\PC\Projects')


WETH_ADDRESS = "0x82af49447d8a07e3bd95bd0d56f35241523fbab1"
DAI_ADDRESS = "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1"
USDC_ADDRESS = "0xaf88d065e77c8cc2239327c5edb3a432268e5831"
USDT_ADDRESS = "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9"
WBTC_ADDRESS = "0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f"

CURVE_USDC_USDT_ADDRESS = "0x7f90122bf0700f9e7e1f688fe926940e8839f353"

UNISWAP_V2_USDT_USDC_ADDRESS = "0x8165c70b01b7807351ef0c5ffd3ef010cabc16fb"
#UNISWAP_V2_WETH_WBTC_ADDRESS = "0x8c1D83A25eE2dA1643A5d937562682b1aC6C856B"
#UNISWAP_V2_USDC_USDT_ADDRESS = "0xdeae1ff5282d83aadd42f85c57f6e69a037bf7cd"

UNISWAP_V3_USDT_USDC_ADDRESS = "0xd1e1ac29b31b35646eabd77163e212b76fe3b6a2"

_web3 = web3.Web3(web3.HTTPProvider("http://localhost:8547")) 
degenbot.config.set_web3(_web3)

async def test_arb_calculation():
    
    try:
        uniswap_v2_usdc_usdt_lp = degenbot.UniswapV2Pool(UNISWAP_V2_USDT_USDC_ADDRESS)
        tkn0out = uniswap_v2_usdc_usdt_lp.calculate_tokens_out_from_tokens_in(uniswap_v2_usdc_usdt_lp.token0,1000000000000000000,None)
        tkn1out = uniswap_v2_usdc_usdt_lp.calculate_tokens_out_from_tokens_in(uniswap_v2_usdc_usdt_lp.token1,1000000000000000000,None)
        print(tkn0out)
        print(tkn1out)
        # uniswap_v2_weth_usdt_lp = LiquidityPool(UNISWAP_V2_WETH_USDT_ADDRESS)

        uniswap_v3_usdc_usdt_lp = degenbot.UniswapV3Pool(UNISWAP_V3_USDT_USDC_ADDRESS)
        tkn0out = uniswap_v3_usdc_usdt_lp.calculate_tokens_out_from_tokens_in(uniswap_v3_usdc_usdt_lp.token0,1000000000000000000,None)
        tkn1out = uniswap_v3_usdc_usdt_lp.calculate_tokens_out_from_tokens_in(uniswap_v3_usdc_usdt_lp.token1,1000000000000000000,None)
        print(tkn0out)
        print(tkn1out)
        # uniswap_v2_weth_dai_lp = LiquidityPool(UNISWAP_V2_WETH_DAI_ADDRESS)
    
        # weth = degenbot.Erc20Token(WETH_ADDRESS)
        # dai = degenbot.Erc20Token(DAI_ADDRESS)
        # wbtc = degenbot.Erc20Token(WBTC_ADDRESS)
        # usdc = degenbot.Erc20Token(USDC_ADDRESS)
        # usdt = degenbot.Erc20Token(USDT_ADDRESS)
        curve_tripool = degenbot.CurveStableswapPool(address=CURVE_USDC_USDT_ADDRESS,chain_id=_web3.eth.chain_id)
        tknout = curve_tripool.calculate_tokens_out_from_tokens_in(uniswap_v2_usdc_usdt_lp.token0,uniswap_v2_usdc_usdt_lp.token1,1000000000000000000,None,None)
        print(tknout)
        combined_pool = CombinedLiquidityPool(
            token_in=uniswap_v2_usdc_usdt_lp.token1,
            token_out=uniswap_v2_usdc_usdt_lp.token0,
            pools_with_states=[
                (curve_tripool,None),
                (uniswap_v2_usdc_usdt_lp,None),
                (uniswap_v3_usdc_usdt_lp,None)
                ]
            )
        best_result = await combined_pool.test_combinations(1000000000000000000)
        print(best_result)
        print("test")
    except Exception as e: 
        print(e)
        traceback.print_exc()
        
asyncio.run(test_arb_calculation())

    # weth = Erc20Token(WETH_ADDRESS)

    # for swap_pools in [
    #     (
    #         uniswap_v2_weth_dai_lp, 
    #         curve_tripool, 
    #         uniswap_v2_weth_usdc_lp
    #     ),
    #     (
    #         uniswap_v2_weth_dai_lp, 
    #         curve_tripool, 
    #         uniswap_v2_weth_usdt_lp
    #     ),
    #     (
    #         uniswap_v2_weth_usdc_lp, 
    #         curve_tripool, 
    #         uniswap_v2_weth_dai_lp
    #     ),
    #     (
    #         uniswap_v2_weth_usdc_lp, 
    #         curve_tripool, 
    #         uniswap_v2_weth_usdt_lp
    #     ),
    #     (
    #         uniswap_v2_weth_usdt_lp, 
    #         curve_tripool, 
    #         uniswap_v2_weth_dai_lp
    #     ),
    #     (
    #         uniswap_v2_weth_usdt_lp, 
    #         curve_tripool, 
    #         uniswap_v2_weth_usdc_lp
    #     ),
    # ]:
    #     arb = UniswapCurveCycle(
    #         input_token=weth,
    #         swap_pools=swap_pools,
    #         id="test",
    #         max_input=10 * 10**18,
    #     )
    #     result = arb.calculate()
