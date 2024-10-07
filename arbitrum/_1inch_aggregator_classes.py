from ctypes import Union
from fractions import Fraction
from types import UnionType
import degenbot
from degenbot.baseclasses import BaseLiquidityPool, BaseArbitrage , BaseTransaction
from eth_typing import BlockIdentifier
import requests
import subprocess
from degenbot.erc20_token import Erc20Token
import ujson as json
import time
import asyncio
import aiohttp
from degenbot.manager.token_manager import Erc20TokenHelperManager
import dataclasses


class OneInchAggregator(BaseLiquidityPool):
    def __init__(
        self,
        chain_id: int,
        api_key: str,
        retrieve_price_data_updates: bool | None = False,
        pairs_to_retrieve_price_data_updates = list[dict],
        silent: bool = False,
        aggregator_address: str | None = None,
        block_number: int | None = None,    
    ) -> None:
        """Initialize the OneInchAggregator with basic parameters."""
        self.chain_id = chain_id
        self.block_number = block_number
        self.api_key = api_key
        self.js_process = None  # Placeholder for JS subprocess
        self.retrieve_price_data_updates = retrieve_price_data_updates
        self.token_list = None
        self.aggregator_address = aggregator_address
        self.price_update_task = None
        self.pairs_to_retrieve_price_data_updates = pairs_to_retrieve_price_data_updates
        
        try:
            self.web3 = degenbot.config.get_web3()
        except NameError as e:
            raise RuntimeError(f"Web3 setup failed: {e}")
        
        _token_manager = Erc20TokenHelperManager(chain_id)
        self.tokens = tuple(
            [
                _token_manager.get_erc20token(
                    address=token_address,
                    silent=silent,
                )
                for token_address in self.token_list
            ]
        )

    async def async_initialize(self):
        """Asynchronously fetch the aggregator address and token list."""
        if self.aggregator_address is None:
            self.aggregator_address = await self._get_aggregator_address()

        self.token_list = await self._get_token_list()

        # Start the persistent JS process
        self.start_js_script()

        # Update latest block info
        self.update_latest_block_info()

        # Perform an initial multicall
        self.initial_request_duration = await self.perform_initial_multicall() 
        # Start periodic price updates if required
        # Start periodic price updates as a background task
        if self.retrieve_price_data_updates and self.pairs_to_retrieve_price_data_updates is None:
            raise ValueError("pairs_to_retrieve_price_data_updates must be provided when retrieve_price_data_updates is True !")
        else: 
            return asyncio.create_task(self.schedule_price_updates(self.pairs_to_retrieve_price_data_updates))
            
    async def _get_aggregator_address(self) -> str:
        """ Returns the address of the current aggregator contract"""
        url = f"https://api.1inch.dev/swap/v6.0/{self.chain_id}/approve/spender"
        AggregatorAddress = await self._get_request(url,self.api_key)
        return AggregatorAddress["address"]
    
    
    async def _get_token_list(self) -> dict:
        """ Retrieves a of all tokens from the OneInch API
        and returns a dictionary with token addresses as keys and their respective names, decimals """
        url = f"https://api.1inch.dev/swap/v6.0/{self.chain_id}/tokens" 
        try :
            tokens_list = await self._get_request(url, self.api_key)
            checksummed_tokens = {}

            # Iterate over the original tokens list
            for address, token_data in tokens_list["tokens"].items():
                # Convert the address to checksum format
                checksummed_addr = self.web3.to_checksum_address(address)
                
                # Assign the token data to the new checksummed address
                checksummed_tokens[checksummed_addr] = token_data

            # Replace the original tokens list with the checksummed version
            tokens_list["tokens"] = checksummed_tokens
        except Exception as e:
            raise f"Failed to fetch token list: {e}"
        return tokens_list["tokens"]
        
    async def _get_request(self, url: str, api_key :str) -> str:
        method = "get"
        apiUrl = url
        requestOptions = {
            "headers": {
        "Authorization": f"Bearer {api_key}"
        },
            "body": "",
            "params": {}
        }
        # Prepare request components
        headers = requestOptions.get("headers", {})
        body = requestOptions.get("body", {})
        params = requestOptions.get("params", {})
        
        async with asyncio.Semaphore(1):
            response = None
            try:
                response = await asyncio.to_thread(
                    requests.request, method, apiUrl, headers=headers, data=body, params=params
                )
                response.raise_for_status()
                await asyncio.sleep(1)
            except Exception as e:
                if response:
                    print(response.status_code, response.text, e)
                else:
                    print("Request failed:", e)
        
        return response.json() if response else {}
        
    def start_js_script(self):
        """Start the persistent JS script for multicalls."""
        self.js_process = subprocess.Popen(
            ["node", r"C:\Users\PC\Projects\dex_arb\arbitrum\multicall.js"],  # Replace with your actual JS script path
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

    def _send_to_js(self, data: str) -> dict:
        """Send requests to the persistent JS process and retrieve results synchronously."""
        if self.js_process is None:
            raise RuntimeError("JS script is not running.")

        try:
            # Write data to the JS process input
            self.js_process.stdin.write(f"{data}\n")
            
            # Read the output synchronously
            raw_output = self.js_process.stdout.readline()

            # Check if the output is empty
            if not raw_output:
                print("Empty output from JS process")
                return {}

            try:
                # Attempt to decode the output as JSON
                result = json.loads(raw_output)

                return result
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                return {}

        except BrokenPipeError as e:
            raise RuntimeError(f"Communication with the JS script failed: {str(e)}")
        except subprocess.SubprocessError as e:
            raise RuntimeError(f"Subprocess error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {str(e)}")
        
    def stop_js_script(self):
        """Stop the persistent JS script process."""
        if self.js_process is not None:
            self.js_process.terminate()  # Beendet den JS-Prozess
            self.js_process.wait()       # Wartet, bis der Prozess komplett beendet ist
            self.js_process = None       # Setzt den Prozess auf None, um anzuzeigen, dass er nicht mehr läuft
            print("JS process terminated.")
        else:
            print("No JS process running.") 
            
    def update_latest_block_info(self):
        """Update the latest block number and timestamp."""
        latest_block = self.web3.eth.get_block('latest')
        self.block_number = latest_block["number"]
        self.block_timestamp = latest_block["timestamp"]
        
    def _get_multicall_prices(self, pairs: list[dict]) -> list[dict]:
        """Fetch multicall prices for given token pairs using the JS script."""
        # Convert the list of dictionaries to a JSON string that JS can parse
        pairs_data = json.dumps(pairs)
        prices = self._send_to_js(pairs_data)  # Ensure _prices is properly parsed
        # Print token list to ensure it's being retrieved correctly
        # Process each price entry
        for price in prices:
            try:
                # Convert addresses to checksummed format before lookup
                from_token_checksummed = self.web3.to_checksum_address(price["from_token"])
                to_token_checksummed = self.web3.to_checksum_address(price["to_token"])
                # Retrieve token dictionaries using checksummed addresses
                from_token_dict = self.token_list.get(from_token_checksummed)
                to_token_dict = self.token_list.get(to_token_checksummed)
                # Check if the token dictionaries were found
                if not from_token_dict or not to_token_dict:
                    print(f"Error: Token data for {from_token_checksummed} or {to_token_checksummed} not found.")
                    continue
                # Adjust the price based on the decimals of the destination token
            except Exception as e:
                print(f"Error processing the price: {e}")
        # Return the processed list of prices
        return prices
    
    async def stop_price_updates(self):
        """Cancel the background price updates task."""
        if self.price_update_task:
            self.price_update_task.cancel()
            try:
                await self.price_update_task
            except asyncio.CancelledError:
                print("Price updates task cancelled.")
                
    async def schedule_price_updates(self, pairs:list[dict], ) -> list[dict]:
            """Schedule price updates based on the estimated request duration and block time.
            pairs = [
                    {
                        "src_token": address,
                        "dst_token": address,
                        "gas": int
                    }
                    ]"""
   
            estimated_request_duration = self.initial_request_duration
            try:
                while True:
                    current_time = time.time()
                    next_block_time = self.block_timestamp + 0.25
                    time_to_next_block = next_block_time - current_time

                    # Wenn die geschätzte Request-Dauer länger als die Blockzeit ist, direkt handeln
                    if estimated_request_duration >= 0.25:
                        wait_time = 0
                        print("Request time is longer than block time. Acting immediately.")
                    else:
                        wait_time = max(0, time_to_next_block - estimated_request_duration)
                    
                    # Sleep if there is time to wait, otherwise process immediately
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)

                    # Update the latest block and timestamp
                    self.update_latest_block_info()

                    # Get prices using _get_multicall_prices with pairs
                    start_time = time.time()
                    multicall_prices_per_block = self._get_multicall_prices(pairs)

                    # Adaptively update the estimated request duration based on how long the request took
                    request_duration = time.time() - start_time
                    estimated_request_duration = (estimated_request_duration + request_duration) / 2

                    return multicall_prices_per_block

            except asyncio.CancelledError:
                print("Price updates loop cancelled.")

        
    
    async def perform_initial_multicall(self) -> float:
        """Perform an initial multicall with all tokens against Ethereum to estimate request duration."""
        pairs = [
            {
                "src_token": "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
                "dst_token": self.token_list[token]["address"],
                "gas": int(self.web3.eth.get_block('latest')['gasLimit'])
            }
                for token in self.token_list if self.token_list[token]["address"]!= "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1"
                ]
        start_time = time.time()
        self._get_multicall_prices(pairs)
        initial_duration = time.time() - start_time
        print(f"Initial request duration: {initial_duration:.4f} seconds")
        return initial_duration
    
    async def fetch(self, session, method, url, params=None, semaphore=None):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with semaphore:  # Begrenze die Anzahl der gleichzeitigen Anfragen
            async with session.request(method=method, url=url, headers=headers, params=params) as response:
                data = await response.json()
                await asyncio.sleep(1)  # Wartezeit zwischen den Requests
                return data
            
    async def get_api_swap(self , src_token, dst_token , amt:Fraction, wallet_addr:str):
        semaphore = asyncio.Semaphore(1)  # Erstelle die Semaphore
        async with aiohttp.ClientSession() as session:
            url_swap = "https://api.1inch.dev/swap/v6.0/42161/swap"
            swap_params = {
                    "src": src_token,
                    "dst": dst_token,
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
                self.fetch(session, "GET", url_swap, params=swap_params, semaphore=semaphore)
            ]
            # Ergebnisse parallel abwarten
            responses = await asyncio.gather(*tasks)
            return responses[0]
        
@dataclasses.dataclass(slots=True, frozen=True)
class OneInchSwapAmounts:
    token_in: Erc20Token
    token_in_amount: int
    token_out: Erc20Token
    token_out_amount: int
    
@dataclasses.dataclass(slots=True, frozen=True)
class OneInchSwapVector:
    token_in: Erc20Token
    token_out: Erc20Token