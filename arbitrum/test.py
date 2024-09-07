import degenbot.arbitrage
import degenbot.manager
from oneinch_py import *
import web3
import web3.eth
import json
import asyncio
import degenbot
from degenbot import LiquidityPool
from degenbot.uniswap.abi import UNISWAP_V2_POOL_ABI
import web3.tools
from curbe_pool_abi import POOL_ABI
from degenbot.erc20_token import Erc20Token
from fractions import Fraction
from decimal import *

rpc = "http://localhost:8547"

async def oneinch_to_uni_arb():
    ...

def from_wei(amount, token):
    if token["symbol"] == "USDC":
        return node_web3.from_wei(amount, "mwei")
    elif token["symbol"] == "WBTC":
        return float(amount / (10**8))
    elif token["symbol"] == "WETH":
        return node_web3.from_wei(amount, "ether")

def to_wei(amount, token):
    if token["symbol"] == "USDC":
        return node_web3.to_wei(amount, "mwei")
    elif token["symbol"] == "WBTC":
        return float(amount * 10**8)
    elif token["symbol"] == "WETH":
        return node_web3.to_wei(amount, "ether")
    
node_web3 = web3.Web3(web3.HTTPProvider(rpc))
# node_web3 = web3.Web3(web3.IPCProvider(ipc_path=r"C:\Users\PC\Projects\DockerData\arbitrum\arbitrum.ipc",timeout=60))
degenbot.config.set_web3(node_web3)
    
UNISWAP_V2_FACTORY = "0xf1D7CC64Fb4452F05c498126312eBE29f30Fbcf9"
UNI_V2_FACTORY_INIT_HASH = "96e8ac4277198ff8b6f785478aa9a39f403cb768dd02cbee326c3e7da348845f"
public_key = "0xA791336Eea54cA199e96123e5e0774a908268A65"
private_key = "e61907a13a06d8e702a76b7651541398fa66966f91527c4f560bb7b3adadd539"
api_key = "Mn27Wch56xgl5IlDXp5IZsrd74hnwOLt"
exchange = OneInchSwap(api_key, public_key, chain='arbitrum')
helper = TransactionHelper(api_key, rpc, public_key, private_key, chain='arbitrum')
oracle = OneInchOracle(rpc, chain='arbitrum')
tokens_list = exchange.tokens
aggregator = exchange.get_spender()["address"]
WETH = tokens_list["WETH"]
USDC = tokens_list["USDC"]
WBTC = tokens_list["WBTC"]
tokens = []
with open("1inch_tokens.json","w") as file:
    for key , value in tokens_list.items():
        tokens.append(
            {
             "token": web3.Web3.to_checksum_address(value.get("address")),
             "name": value.get("name"),
             "aggregator":aggregator,
             "providers": value.get("providers"),
             "symbol": value.get("symbol"),
             "pool_type": "1inch_Aggregator"
            }
        )
    json.dump(
        tokens,
        file,
        indent=2
    )
    
# Verbindung über HTTP
arbs = []

async def main():
    lp = LiquidityPool(node_web3.to_checksum_address("0xf64dfe17c8b87f012fcf50fbda1d62bfa148366a"), factory_address=UNISWAP_V2_FACTORY, factory_init_hash=UNI_V2_FACTORY_INIT_HASH, abi=UNISWAP_V2_POOL_ABI, silent=False)
    v3_lp = degenbot.V3LiquidityPool(node_web3.to_checksum_address("0xc6962004f452be9203591991d15f6b388e09e8d0"))
    # curve_lp = degenbot.CurveStableswapPool(node_web3.to_checksum_address("0x7f90122bf0700f9e7e1f688fe926940e8839f353"), abi=POOL_ABI)

    raw_tx = exchange.get_swap(
        from_token_symbol=WETH["address"],
        to_token_symbol=USDC["address"],
        amount=1,
        decimal=WETH["decimals"],
        slippage=0.1,
        disableEstimate="true",
        includeTokensInfo="true",
        includeProtocols="true",
        compatibility="true",
        parts=100,
        mainRouteParts=50,
        complexityLevel=3
    )
    calldata1 = raw_tx["tx"]["data"]
    oneinch_amount = Decimal(int(raw_tx['toAmount'])) / Decimal(10**USDC["decimals"])

    v3_tkns_out_from_tkns_in = Decimal(int(v3_lp.calculate_tokens_out_from_tokens_in(
        token_in=lp.token0,
        token_in_quantity=int(1 * 10**v3_lp.token0.decimals)
    )) / Decimal(10**v3_lp.token1.decimals))

    v2_tkns_out_from_tkns_in = Decimal(int(lp.calculate_tokens_out_from_tokens_in(
        token_in=lp.token0,
        token_in_quantity=int(1 * 10**lp.token0.decimals)
    )) / Decimal(10**lp.token1.decimals))

    # Print the results using the `Decimal` format for full precision
    print(oneinch_amount)
    print(calldata1)
    print(v3_tkns_out_from_tkns_in)
    print(v2_tkns_out_from_tkns_in)


asyncio.run(main())
