from datetime import datetime
import datetime
from fractions import Fraction
from pickle import FALSE
import time
import json
from oneinch_py import *
from decimal import *
from decimal import Decimal as dec
from urllib3.exceptions import HTTPError
import logging
import requests
from eth_account.messages import encode_defunct
import degenbot
from web3 import Web3
from eth_abi import decode
from decimal import Decimal
import aiohttp
import asyncio
import time
from degenbot.uniswap.abi import UNISWAP_V2_POOL_ABI
from degenbot.erc20_token import Erc20Token
from degenbot import LiquidityPool
from scipy import optimize
from degenbot.uniswap.managers import UniswapV2LiquidityPoolManager,UniswapLiquidityPoolManager,UniswapV3LiquidityPoolManager
from dex_arb.arbitrum._1inch_aggregator_classes import OneInchAggregator
rpc = "https://floral-crimson-patina.arbitrum-mainnet.quiknode.pro/347691b7280c64c57237f77f7c8988972dde604d/"
public_key = "0xA791336Eea54cA199e96123e5e0774a908268A65"
private_key = "e61907a13a06d8e702a76b7651541398fa66966f91527c4f560bb7b3adadd539"
api_key = "Mn27Wch56xgl5IlDXp5IZsrd74hnwOLt"
web3 = Web3(Web3.HTTPProvider(rpc))
degenbot.config.set_web3(web3)
# Definiere die maximale Anzahl an gleichzeitigen Requests und die Wartezeit zwischen den Requests (in Sekunden)
MAX_CONCURRENT_REQUESTS = 1  # Anzahl gleichzeitiger Anfragen
DELAY_BETWEEN_REQUESTS = 1.5  # Sekunden zwischen den Requests
# Globale Header-Konfiguration für alle Anfragen
headers = {"Authorization": "Bearer BMYEt9qskUTWzCtyPrGKOUMEmgkX8iOK"}

UNISWAP_V2_FACTORY = "0xf1D7CC64Fb4452F05c498126312eBE29f30Fbcf9"
UNI_V2_FACTORY_INIT_HASH = "96e8ac4277198ff8b6f785478aa9a39f403cb768dd02cbee326c3e7da348845f"
# Gaslimit-Konfiguration
MAX_GAS_LIMIT = 150_000_000
DEFAULT_GAS_BUFFER = 3_000_000

# Berechne das maximale Gaslimit für Multicall
def calculate_gas_limit(node_gas_limit, max_gas_limit=MAX_GAS_LIMIT, gas_buffer=DEFAULT_GAS_BUFFER):
    return min(node_gas_limit, max_gas_limit) - gas_buffer

# Asynchrone Funktion zur Ausführung von HTTP-Requests mit Rate-Limit
async def fetch(session, method, url, params=None, semaphore=None):
    async with semaphore:  # Begrenze die Anzahl der gleichzeitigen Anfragen
        async with session.request(method=method, url=url, headers=headers, params=params) as response:
            data = await response.json()
            await asyncio.sleep(DELAY_BETWEEN_REQUESTS)  # Wartezeit zwischen den Requests
            return data

def convert_token_amount(amount, token, to_readable=True):
    if to_readable:
        # Von Wei zu menschenlesbar konvertieren
        readable_value = Fraction(amount, 10**token["decimals"])
        return float(readable_value)  # Falls gewünscht, als Float ausgeben
    else:
        # Von menschenlesbar zu Wei konvertieren
        wei_value = int(Fraction(amount) * 10**token["decimals"])
        return wei_value

async def get_tokens_list():
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)  # Erstelle die Semaphore
    async with aiohttp.ClientSession() as session:
        # Definiere die URLs und Parameter
        url_tokens = "https://api.1inch.dev/swap/v6.0/42161/tokens"
        # Sende den Request
        response = await fetch(session, "GET", url_tokens, semaphore=semaphore)
        # Extrahiere die Tokens aus der Antwort und formatiere sie in ein Dictionary mit Symbolen als Schlüssel
        tokens_data = response.get('tokens', {})
        tokens_by_symbol = {token['symbol']: token for token in tokens_data.values()}
        
        return tokens_by_symbol
    
async def get_api_swap(src_token, dst_token , amt:Fraction, wallet_addr:str):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)  # Erstelle die Semaphore
    async with aiohttp.ClientSession() as session:
        url_swap = "https://api.1inch.dev/swap/v6.0/42161/swap"
        swap_params = {
                "src": src_token["address"],
                "dst": dst_token["address"],
                "amount": str(int(amt*10**src_token["decimals"])),
                "from": wallet_addr,
                "origin": wallet_addr,
                "slippage": 0.4,
                # "protocols": "nc",
                # "fee": "nc",
                # "gasPrice": "nc",
                # "complexityLevel": 3,
                #TODO TEST "parts": "ncg",
                #TODO "mainRouteParts": "ngc",
                # "gasLimit": "ncg",
                "includeTokensInfo": "true",
                "includeProtocols": "true",
                "includeGas": "true",
                # "connectorTokens": "cng",
                # "excludedProtocols": "ng",
                "allowPartialFill": "false",
                "disableEstimate": "true",
                "usePermit2": "false"
            }
        # Sende die Requests asynchron mit Rate-Limit
        tasks = [
            fetch(session, "GET", url_swap, params=swap_params, semaphore=semaphore)
        ]
        # Ergebnisse parallel abwarten
        responses = await asyncio.gather(*tasks)
        return responses[0]
    
def calc_profit(borrow_amount, pool):
    # Berechne den benötigten Rückzahlungsbetrag (Input) für den geliehenen Betrag (Output)
    flash_repay_amount = pool.calculate_tokens_in_from_tokens_out(
        token_out_quantity=borrow_amount,
        token_out=pool.token0
    )

    # Berechne den Output, den wir nach dem Swap erhalten (ohne Berücksichtigung von Flashloan)
    swap_amount_out = pool.calculate_tokens_out_from_tokens_in(
        token_in_quantity=borrow_amount,
        token_in=pool.token1
    )

    # Gewinn ist der Unterschied zwischen dem erhaltenen Output und dem zurückzuzahlenden Betrag
    return swap_amount_out - flash_repay_amount

def find_optimal_borrow(pool):
    """
    Funktion, um den optimalen Borrow-Wert für den Pool zu berechnen.
    Die Berechnung basiert auf den aktuellen Pool-Reserven und Gebühren, um den maximalen Profit zu erzielen.
    """
    
    # Definiere die Grenzen für das Borrowing: 1 bis zur maximalen Reserve des Tokens im Pool
    bounds = (1, pool.reserves_token0)  # Hier reservieren wir den größeren Token als Obergrenze
    bracket = (0.01 * pool.reserves_token0, 0.05 * pool.reserves_token0)  # Erste Schätzung zwischen 1% und 5% der Reserve

    # Optimierungslauf mit Brent's Methode
    result = optimize.minimize_scalar(
        lambda x: -float(calc_profit(borrow_amount=x, pool=pool)),  # Negieren, da wir das Maximum suchen
        method="bounded",
        bounds=bounds,
        bracket=bracket
    )

    # Optimale Borrow-Wert und der dazugehörige Profit
    optimal_borrow = result.x
    optimal_profit = -result.fun  # Da wir es vorher negiert haben
    
    return optimal_borrow, optimal_profit

# Hauptfunktion, die die Anfragen ausführt
async def main():
        tokens_list = await get_tokens_list()
        WETH = tokens_list["WETH"]
        USDC = tokens_list["USDC"]
        lp = LiquidityPool(web3.to_checksum_address("0xf64dfe17c8b87f012fcf50fbda1d62bfa148366a"), factory_address=UNISWAP_V2_FACTORY, factory_init_hash=UNI_V2_FACTORY_INIT_HASH, abi=UNISWAP_V2_POOL_ABI, silent=False)
        v3_lp = degenbot.V3LiquidityPool(web3.to_checksum_address("0xc6962004f452be9203591991d15f6b388e09e8d0"))


        # ABI-Definitionen für die MultiCall- und OffChainOracle-Verträge
        multi_call_abi = '''
        [{"inputs":[],"name":"gasLeft","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"gaslimit","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"components":[{"internalType":"address","name":"to","type":"address"},{"internalType":"bytes","name":"data","type":"bytes"}],"internalType":"struct MultiCall.Call[]","name":"calls","type":"tuple[]"}],"name":"multicall","outputs":[{"internalType":"bytes[]","name":"results","type":"bytes[]"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"components":[{"internalType":"address","name":"to","type":"address"},{"internalType":"bytes","name":"data","type":"bytes"}],"internalType":"struct MultiCall.Call[]","name":"calls","type":"tuple[]"}],"name":"multicallWithGas","outputs":[{"internalType":"bytes[]","name":"results","type":"bytes[]"},{"internalType":"uint256[]","name":"gasUsed","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"components":[{"internalType":"address","name":"to","type":"address"},{"internalType":"bytes","name":"data","type":"bytes"}],"internalType":"struct MultiCall.Call[]","name":"calls","type":"tuple[]"},{"internalType":"uint256","name":"gasBuffer","type":"uint256"}],"name":"multicallWithGasLimitation","outputs":[{"internalType":"bytes[]","name":"results","type":"bytes[]"},{"internalType":"uint256","name":"lastSuccessIndex","type":"uint256"}],"stateMutability":"nonpayable","type":"function"}]
        '''
        off_chain_oracle_abi = '''
        [{"inputs":[{"internalType":"contract MultiWrapper","name":"_multiWrapper","type":"address"},{"internalType":"contract IOracle[]","name":"existingOracles","type":"address[]"},{"internalType":"enum OffchainOracle.OracleType[]","name":"oracleTypes","type":"uint8[]"},{"internalType":"contract IERC20[]","name":"existingConnectors","type":"address[]"},{"internalType":"contract IERC20","name":"wBase","type":"address"},{"internalType":"address","name":"owner","type":"address"}],"stateMutability":"nonpayable","type":"constructor"},{"inputs":[],"name":"ArraysLengthMismatch","type":"error"},{"inputs":[],"name":"ConnectorAlreadyAdded","type":"error"},{"inputs":[],"name":"InvalidOracleTokenKind","type":"error"},{"inputs":[],"name":"OracleAlreadyAdded","type":"error"},{"inputs":[],"name":"SameTokens","type":"error"},{"inputs":[],"name":"TooBigThreshold","type":"error"},{"inputs":[],"name":"UnknownConnector","type":"error"},{"inputs":[],"name":"UnknownOracle","type":"error"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"contract IERC20","name":"connector","type":"address"}],"name":"ConnectorAdded","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"contract IERC20","name":"connector","type":"address"}],"name":"ConnectorRemoved","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"contract MultiWrapper","name":"multiWrapper","type":"address"}],"name":"MultiWrapperUpdated","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"contract IOracle","name":"oracle","type":"address"},{"indexed":false,"internalType":"enum OffchainOracle.OracleType","name":"oracleType","type":"uint8"}],"name":"OracleAdded","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"contract IOracle","name":"oracle","type":"address"},{"indexed":false,"internalType":"enum OffchainOracle.OracleType","name":"oracleType","type":"uint8"}],"name":"OracleRemoved","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"previousOwner","type":"address"},{"indexed":true,"internalType":"address","name":"newOwner","type":"address"}],"name":"OwnershipTransferred","type":"event"},{"inputs":[{"internalType":"contract IERC20","name":"connector","type":"address"}],"name":"addConnector","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"contract IOracle","name":"oracle","type":"address"},{"internalType":"enum OffchainOracle.OracleType","name":"oracleKind","type":"uint8"}],"name":"addOracle","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"connectors","outputs":[{"internalType":"contract IERC20[]","name":"allConnectors","type":"address[]"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"contract IERC20","name":"srcToken","type":"address"},{"internalType":"contract IERC20","name":"dstToken","type":"address"},{"internalType":"bool","name":"useWrappers","type":"bool"}],"name":"getRate","outputs":[{"internalType":"uint256","name":"weightedRate","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"contract IERC20","name":"srcToken","type":"address"},{"internalType":"bool","name":"useSrcWrappers","type":"bool"}],"name":"getRateToEth","outputs":[{"internalType":"uint256","name":"weightedRate","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"contract IERC20","name":"srcToken","type":"address"},{"internalType":"bool","name":"useSrcWrappers","type":"bool"},{"internalType":"contract IERC20[]","name":"customConnectors","type":"address[]"},{"internalType":"uint256","name":"thresholdFilter","type":"uint256"}],"name":"getRateToEthWithCustomConnectors","outputs":[{"internalType":"uint256","name":"weightedRate","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"contract IERC20","name":"srcToken","type":"address"},{"internalType":"bool","name":"useSrcWrappers","type":"bool"},{"internalType":"uint256","name":"thresholdFilter","type":"uint256"}],"name":"getRateToEthWithThreshold","outputs":[{"internalType":"uint256","name":"weightedRate","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"contract IERC20","name":"srcToken","type":"address"},{"internalType":"contract IERC20","name":"dstToken","type":"address"},{"internalType":"bool","name":"useWrappers","type":"bool"},{"internalType":"contract IERC20[]","name":"customConnectors","type":"address[]"},{"internalType":"uint256","name":"thresholdFilter","type":"uint256"}],"name":"getRateWithCustomConnectors","outputs":[{"internalType":"uint256","name":"weightedRate","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"contract IERC20","name":"srcToken","type":"address"},{"internalType":"contract IERC20","name":"dstToken","type":"address"},{"internalType":"bool","name":"useWrappers","type":"bool"},{"internalType":"uint256","name":"thresholdFilter","type":"uint256"}],"name":"getRateWithThreshold","outputs":[{"internalType":"uint256","name":"weightedRate","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"multiWrapper","outputs":[{"internalType":"contract MultiWrapper","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"oracles","outputs":[{"internalType":"contract IOracle[]","name":"allOracles","type":"address[]"},{"internalType":"enum OffchainOracle.OracleType[]","name":"oracleTypes","type":"uint8[]"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"owner","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"contract IERC20","name":"connector","type":"address"}],"name":"removeConnector","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"contract IOracle","name":"oracle","type":"address"},{"internalType":"enum OffchainOracle.OracleType","name":"oracleKind","type":"uint8"}],"name":"removeOracle","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"renounceOwnership","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"contract MultiWrapper","name":"_multiWrapper","type":"address"}],"name":"setMultiWrapper","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"newOwner","type":"address"}],"name":"transferOwnership","outputs":[],"stateMutability":"nonpayable","type":"function"}]
        '''
    
        # Adressen der Verträge
        off_chain_oracle_address = web3.to_checksum_address("0x0AdDd25a91563696D8567Df78D5A01C9a991F9B8")
        multi_call_contract_address = web3.to_checksum_address("0x11DEE30E710B8d4a8630392781Cc3c0046365d4c")

        # Erstellen der Vertragsinstanzen
        multi_call_contract = web3.eth.contract(address=multi_call_contract_address, abi=multi_call_abi)
        off_chain_oracle_contract = web3.eth.contract(address=off_chain_oracle_address, abi=off_chain_oracle_abi)

        # Vorbereitung der Call-Daten für Multicall
        # Erstelle die Liste der Tokens mit nur "address" und "decimals"
        tokens = [{"address": web3.to_checksum_address(token["address"]), "decimals": token["decimals"]} for token in tokens_list.values()]
        # connectors = [
        #     "0x0000000000000000000000000000000000000000",
        #     "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
        #     "0xFFfFfFffFFfffFFfFFfFFFFFffFFFffffFfFFFfF",
        #     "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
        #     ]
        
        while True:
            call_data = []
            for token in tokens:
                # Erstelle die Daten für jeden Aufruf der getRateWithCustomConnectors-Funktion
                data = off_chain_oracle_contract.encodeABI(
                    fn_name="getRate",
                    args=[
                        web3.to_checksum_address(WETH["address"]),
                        web3.to_checksum_address(USDC["address"]),
                        True,
                    ]
                )
                # Umwandlung der String-Daten in Bytes
                data_bytes = bytes.fromhex(data[2:])  # Entfernt '0x' und konvertiert in Bytes
                call_data.append({
                    "to": off_chain_oracle_address,
                    "data": data_bytes
                })
            print(f"Call-Data: {data}")
            # Hole das aktuelle Gaslimit von der Node
            node_gas_limit = web3.eth.get_block('latest')['gasLimit']
            # Berechne das geeignete Gaslimit für Multicall
            gas_limit_for_multicall = calculate_gas_limit(node_gas_limit)
           
            try:
                # Verwenden von multicallWithGasLimitation und Festlegen eines Gas-Puffers
                result = multi_call_contract.functions.multicallWithGasLimitation(call_data, gas_limit_for_multicall).call()
                prices = {}
                # Ergebnisse verarbeiten
                for i, res in enumerate(result[0]):
                    if res:
                        # Konvertiere das Ergebnis direkt in Bytes, falls noch nicht im richtigen Format
                        decoded_rate = decode(['uint256'], res)[0]
                        price = float(Fraction(decoded_rate)/10**USDC["decimals"])   
                        prices[tokens[i]["address"]] = str(price)
            except Exception as e:
                print(f"Fehler: {e}")
                
            # uniswap pool amounts out
            v3_tkns_out_from_tkns_in = Decimal(int(v3_lp.calculate_tokens_out_from_tokens_in(
                token_in=lp.token0,
                token_in_quantity=int(1 * 10**v3_lp.token0.decimals)
            )) / Decimal(10**v3_lp.token1.decimals))

            v2_tkns_out_from_tkns_in = Decimal(int(lp.calculate_tokens_out_from_tokens_in(
                token_in=lp.token0,
                token_in_quantity=int(1 * 10**lp.token0.decimals)
            )) / Decimal(10**lp.token1.decimals))

            api_res = await get_api_swap(src_token=WETH, dst_token=USDC, amt=Fraction(1), wallet_addr=public_key)
            result_api = float(Fraction(api_res["dstAmount"])/10**USDC["decimals"])
            print("v3_tkns_out_from_tkns_in",v3_tkns_out_from_tkns_in)
            print("v2_tkns_out_from_tkns_in",v2_tkns_out_from_tkns_in)
            print("api    ",result_api)
            print("oracle ",price)
            if result_api > price:
                rech = result_api - price
            elif price > result_api:
                rech = price - result_api
            print("difference", rech)
            print("")

            # v2_lp_helper = UniswapV2LiquidityPoolManager(UNISWAP_V2_FACTORY,42161)
            # pool = v2_lp_helper.get_pool("0xf01fCb630aB3b31063bDD66A640f92DB4fAd4044")  
            
            # v3_lp_helper = UniswapV3LiquidityPoolManager("0x1F98431c8aD98523631AE4a59f267346ea31F984","0x6C9FC64A53c1b71FB3f9Af64d1ae3A4931A5f4E9",42161,r"C:\Users\PC\Projects\dex_arb\arbitrum\degen_scripts\arbitrum_v3_liquidity_snapshot.json")
            # v3_pool = v3_lp_helper.get_pool("0x58d7283E1cCa23ad98d4307b9e899eF3827ce7eC")
            # aggregator_v6 = OneInchAggregator(42161,"BMYEt9qskUTWzCtyPrGKOUMEmgkX8iOK",FALSE)
            # print(aggregator_v6.address)
            

# Starte das asynchrone Programm
if __name__ == "__main__":
    asyncio.run(main())
