from datetime import datetime
import datetime
import time
import ape.contracts
import ape.managers
import simplejson as json
from oneinch_py import *
from decimal import *
from decimal import Decimal as dec
from urllib3.exceptions import HTTPError
import web3 as W3
import logging
from ape import project, accounts,Contract
from eth_account.messages import encode_defunct
import ape.exceptions
import degenbot
import ape_accounts

rpc = "https://floral-crimson-patina.arbitrum-mainnet.quiknode.pro/347691b7280c64c57237f77f7c8988972dde604d/"
public_key = "0xA791336Eea54cA199e96123e5e0774a908268A65"
private_key = "e61907a13a06d8e702a76b7651541398fa66966f91527c4f560bb7b3adadd539"
api_key = "Mn27Wch56xgl5IlDXp5IZsrd74hnwOLt"

exchange = OneInchSwap(api_key, public_key, chain='arbitrum')
helper = TransactionHelper(api_key, rpc, public_key, private_key, chain='arbitrum')
oracle = OneInchOracle(rpc, chain='arbitrum')

tokens_list = exchange.tokens
WETH = tokens_list["WETH"]
USDC = tokens_list["USDC"]
WBTC = tokens_list["WBTC"]

raw_tx = exchange.get_swap(
    from_token_symbol = WETH["address"],
    to_token_symbol = USDC["address"],
    amount = 1,
    decimal = WETH["decimals"],
    slippage = 0.1,
    disableEstimate ="true",
    includeTokensInfo="true",
    includeProtocols="true",
    compatibility="true",
    parts=100,
    mainRouteParts=50,
    complexityLevel=3
    )

amount = int(raw_tx['toAmount'])/10**USDC["decimals"]
print(amount)



