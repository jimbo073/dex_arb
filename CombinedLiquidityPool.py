from scipy.optimize import minimize
from itertools import chain, combinations
import dataclasses
from typing import List, Tuple, Union, Any, Dict, Iterable
import asyncio
import sys 
sys.path.append(r'C:\Users\PC\Projects\degenbot')

from degenbot.src import degenbot
from degenbot.src.degenbot.types import *
from degenbot.src.degenbot.curve.types import CurveStableswapPoolState
from eth_typing import ChecksumAddress
import numpy as np
import collections
import time

@dataclasses.dataclass(slots=True, frozen=True)
class CombinedLiquidityArbitrageCalculationResult(degenbot.ArbitrageCalculationResult):
    distribution: Any  # Stores the distribution of liquidity among the pools

@dataclasses.dataclass(slots=True, frozen=True)
class CombinedSwapVector:
    token_in: degenbot.Erc20Token  # The input token for the swap
    token_out: degenbot.Erc20Token  # The output token for the swap
    pool_addresses: List[str]  # List of addresses of the pools involved in the swap

@dataclasses.dataclass(slots=True)
class CombinedSwapAmounts:
    swap_amounts: List[Tuple[int, int]]  # List of input and output amounts for each swap
    recipient: ChecksumAddress  # The address of the recipient
    total_input: int  # Total input amount for all swaps
    total_output: int  # Total output amount after all swaps
    pool_combination_distribution: Any  # Stores the distribution of liquidity across pools

class CombinedLiquidityPool(AbstractLiquidityPool):
    def __init__(
        self,
        token_in: degenbot.Erc20Token,
        token_out: degenbot.Erc20Token,
        pools_with_states: Iterable[
            tuple[degenbot.UniswapV2Pool | degenbot.CamelotLiquidityPool, degenbot.UniswapV2PoolState | None] |
            tuple[degenbot.UniswapV3Pool, degenbot.UniswapV3PoolState | None] |
            tuple[degenbot.CamelotLiquidityPool,degenbot.UniswapV2PoolState | None] |
            tuple[degenbot.CurveStableswapPool, CurveStableswapPoolState | None]
        ],
        pools: Iterable[AbstractLiquidityPool] | None= None, 
        silent: bool = False
    ):
        self.token_in = token_in
        self.token_out = token_out
        self.pools = pools
        self.pools_with_states = pools_with_states
        self.silent = silent
        self.cache = collections.OrderedDict()  # Cache to store calculated outputs for better performance
        self.cache_max_size = 100  # Set a limit to the cache size
        self.cache_expiry_time = 300  # Cache entries expire after 300 seconds
        
        if self.pools_with_states == None and self.pools == None:
            raise ValueError("Either 'pools_with_states' or 'pools' must be provided!")
        
        # Ensure that the input token is present in all provided pools
        if self.pools == None and self.pools_with_states != None:
            self.pools: Iterable[AbstractLiquidityPool] = [pool_tuple[0] for pool_tuple in self.pools_with_states]
        
        for pool in self.pools:
            if self.token_in not in pool.tokens:
                raise ValueError(f"Token {token_in.symbol} @ {token_in.address} is not in pool {pool.address} tokens! [Pool tokens: {pool.tokens}]")

        self.pool_states: Dict[ChecksumAddress, AbstractPoolState] = {}
        self._update_pool_states(self.pools)
        
    def _update_pool_states(self, pools: Iterable[AbstractLiquidityPool]) -> None:
        """
        Update `self.pool_states` with state values from the `pools` iterable
        """
        self.pool_states.update({pool.address: pool.state for pool in pools})
        
    def calculate_tokens_out_from_tokens_in(
        self,
        token_in: degenbot.Erc20Token,
        token_out: degenbot.Erc20Token,
        amount_in: int
    ) -> int:
        start_time = time.time()
        current_time = time.time()
        cache_key = (token_in.address, token_out.address, amount_in)

        # Check if the result is in cache and still valid
        if cache_key in self.cache:
            cache_entry = self.cache.pop(cache_key)
            if current_time - cache_entry['time'] < self.cache_expiry_time:
                if not self.silent:
                    print(f"Cache hit for key: {cache_key}")
                self.cache[cache_key] = cache_entry  # Move to the end to mark as recently used
                return cache_entry['value']

        if not self.silent:
            print(f"Cache miss for key: {cache_key}, calculating tokens out.")
        total_output = 0

        # Iterate through each pool and calculate the output amount
        for pool_tuple in self.pools_with_states:
            pool, override_state = pool_tuple

            if isinstance(pool, degenbot.CurveStableswapPool):
                output = pool.calculate_tokens_out_from_tokens_in(
                    token_in=token_in,
                    token_out=token_out,
                    token_in_quantity=amount_in,
                    override_state=override_state
                )
            elif isinstance(pool, (degenbot.UniswapV2Pool, degenbot.CamelotLiquidityPool)):
                output = pool.calculate_tokens_out_from_tokens_in(
                    token_in=token_in,
                    token_in_quantity=amount_in,
                    override_state=override_state
                )
            elif isinstance(pool,degenbot.UniswapV3Pool):
                output = pool.calculate_tokens_out_from_tokens_in(
                    token_in=token_in,
                    token_in_quantity=amount_in,
                    override_state=override_state
                )
            else:
                raise ValueError(f"Unsupported pool type for pool {pool.address}")
            if not self.silent:
                print(f"Pool {pool.address}: Calculated output: {output}")
            total_output += output

        # Add the result to cache with expiry time and size limit handling
        if len(self.cache) >= self.cache_max_size:
            if not self.silent:
                print("Cache full. Removing oldest entry.")
            self.cache.popitem(last=False)  # Remove the oldest entry to maintain cache size
        self.cache[cache_key] = {'value': total_output, 'time': current_time}
        if not self.silent:
            print(f"Cache updated for key: {cache_key} with value: {total_output}")
        end_time = time.time()
        if not self.silent:
            print(f"Time taken for calculate_tokens_out_from_tokens_in: {end_time - start_time:.4f} seconds")
        return total_output

    def objective_function(self, amounts_in):
        start_time = time.time()
        total_output = 0
        # Iterate over each pool and calculate the output based on the input amount
        for pool_tuple, amount_in in zip(self.pools_with_states, amounts_in):
            if not isinstance(pool_tuple, tuple):
                print(f"Unexpected format for pool_tuple: {pool_tuple}")
                raise ValueError(f"Expected tuple but got: {type(pool_tuple)}")
            if len(pool_tuple) != 2:
                print(f"Unexpected length for pool_tuple: {len(pool_tuple)} - Content: {pool_tuple}")
                raise ValueError(f"Expected a tuple of length 2 but got length {len(pool_tuple)}")

            pool, override_state = pool_tuple

            if not isinstance(pool, (degenbot.CurveStableswapPool, degenbot.UniswapV2Pool, degenbot.UniswapV3Pool, degenbot.CamelotLiquidityPool)):
                raise ValueError(f"Unexpected pool type: {type(pool)} - Pool content: {pool}")

            # Flexibles Entpacken des Tuples, standardmäßig None, wenn kein State vorhanden ist
            if isinstance(pool_tuple, tuple):
                if len(pool_tuple) == 1:
                    pool = pool_tuple[0]
                    override_state = None
                elif len(pool_tuple) == 2:
                    pool, override_state = pool_tuple
                else:
                    raise ValueError(f"Unexpected tuple length: {len(pool_tuple)} in {pool_tuple}")
            else:
                pool = pool_tuple
                override_state = None
                
            if isinstance(pool,  degenbot.CurveStableswapPool):
                output = pool.calculate_tokens_out_from_tokens_in(
                    token_in=self.token_in,
                    token_out=self.token_out,
                    token_in_quantity=amount_in,
                    override_state=override_state
                )
            elif isinstance(pool, (degenbot.UniswapV2Pool, degenbot.CamelotLiquidityPool)):
                output = pool.calculate_tokens_out_from_tokens_in(
                    token_in=self.token_in,
                    token_in_quantity=amount_in,
                    override_state=override_state
                )
            elif isinstance(pool,degenbot.UniswapV3Pool):
                output = pool.calculate_tokens_out_from_tokens_in(
                    token_in=self.token_in,
                    token_in_quantity=amount_in,
                    override_state=override_state
                )
            else:
                raise ValueError(f"Unsupported pool type for pool {pool.address}")
            if not self.silent:
                print(f"Pool {pool.address}: Objective function output: {output}")
            total_output += output

        end_time = time.time()
        if not self.silent:
            print(f"Time taken for objective_function: {end_time - start_time:.4f} seconds")
        return -total_output  # We negate the output because we want to maximize it, but the optimizer minimizes by default

    def constraint(self, amounts_in, total_amount_in):
        start_time = time.time()
        # Ensure that the sum of the input amounts matches the total input amount
        constraint_value = np.sum(amounts_in) - total_amount_in
        end_time = time.time()
        if not self.silent:
            print(f"Constraint check: {constraint_value}")
            print(f"Time taken for constraint check: {end_time - start_time:.4f} seconds")
        return constraint_value

    def optimize_combined_swap(self, total_amount_in:int):
        start_time = time.time()
        # Initial guess: Even distribution of the total amount across all pools
        initial_guess = np.full(len(self.pools), total_amount_in / len(self.pools))
        bounds = [(0, total_amount_in) for _ in range(len(self.pools))]  # Bounds for each pool input
        constraints = {'type': 'eq', 'fun': lambda x: self.constraint(x, total_amount_in)}  # Constraint to ensure valid input distribution

        # Run the optimization synchronously to support usage with ProcessPoolExecutor
        if not self.silent:
            print("Starting optimization.")
        result = minimize(self.objective_function, initial_guess, bounds=bounds, constraints=constraints)
        if not self.silent:
            print(f"Optimization result: {result}")
        end_time = time.time()
        if not self.silent:
            print(f"Time taken for optimize_combined_swap: {end_time - start_time:.4f} seconds")
        return result.x, -result.fun  # Return the optimized input distribution and corresponding output

    async def test_combinations(self, total_amount_in:int):
        start_time = time.time()
        best_output = -np.inf
        best_combination = None

        tasks = []
        # Generate combinations of pools to find the most profitable one
        for combination in self.generate_combinations():
            if len(combination) == 0:
                continue
            if not self.silent:
                print(f"Testing combination: {combination}")
            input_combination = np.zeros(len(self.pools))
            input_combination[list(combination)] = int(round(total_amount_in / len(combination))) # Evenly distribute input among the selected pools
            tasks.append(self.optimize_combined_swap(total_amount_in))

        # Run all optimizations concurrently and gather the results
        results = await asyncio.gather(*tasks)
        for result in results:
            output = result[1]
            if not self.silent:
                print(f"Combination result output: {output}")
            if output > best_output:
                best_output = output
                best_combination = result[0]

        self.best_combination = best_combination  # Store the best combination found
        end_time = time.time()
        if not self.silent:
            print(f"Best combination found: {best_combination} with output: {best_output}")
            print(f"Time taken for test_combinations: {end_time - start_time:.4f} seconds")
        return best_output

    def generate_combinations(self):
        # Generate all possible combinations of pool indices, excluding the empty set
        indices = range(len(self.pools))
        return chain.from_iterable(combinations(indices, r) for r in range(1, len(indices) + 1))