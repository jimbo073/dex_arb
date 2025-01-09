from decimal import Decimal
from fractions import Fraction
from degenbot import erc20_token
from degenbot.camelot.pools import CamelotLiquidityPool
from degenbot.curve.curve_stableswap_liquidity_pool import CurveStableswapPool
from degenbot.erc20_token import Erc20Token
from degenbot.uniswap.types import UniswapV3PoolState, UniswapV2PoolState
from degenbot.uniswap.v2_liquidity_pool import UniswapV2Pool
from degenbot.uniswap.v3_liquidity_pool import UniswapV3Pool
from degenbot.exceptions import IncompleteSwap, DegenbotTypeError
import functools
import scipy
from scipy.optimize import minimize, minimize_scalar
from itertools import chain, combinations
import dataclasses
from typing import List, Tuple, Union, Any, Dict, Iterable, Sequence
import asyncio
import sys
import web3
import scipy.optimize

from dex_arb.arbitrum.CombinedPoolStates import UniswapV2CombinedPoolState , UniswapV3CombinedPoolState, CurveCombinedPoolState
sys.path.append(r'C:\Users\PC\Projects\degenbot')

import degenbot
from degenbot.curve.types import CurveStableswapPoolState
from degenbot.types import AbstractLiquidityPool, AbstractPoolState
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
    involved_pools: List[AbstractLiquidityPool]  # List of the pools involved in the swap

@dataclasses.dataclass(slots=True)
class CombinedSwapAmounts:
    swap_amounts: List[Tuple[int, int]]  # List of input and output amounts for each swap
    total_input: int  # Total input amount for all swaps
    total_output: int  # Total output amount after all swaps
    pool_combination_distribution: Any  # Stores the distribution of liquidity across pools

class CombinedLiquidityPool(AbstractLiquidityPool):
    def __init__(
        self,
        token_in: degenbot.Erc20Token,
        token_out: degenbot.Erc20Token,
        pools_with_states: Iterable[
            tuple[UniswapV2Pool | CamelotLiquidityPool, UniswapV2PoolState | None] |
            tuple[UniswapV3Pool, UniswapV3PoolState | None] |
            tuple[CurveStableswapPool, CurveStableswapPoolState | None]
        ],
        max_input: int | None = None,
        silent: bool = False
    ):
        self.token_in = token_in
        self.token_out = token_out
        self.max_input = max_input
        self.pools: Iterable[UniswapV2Pool | CamelotLiquidityPool | UniswapV3Pool | CurveStableswapPool] = [pool_tuple[0] for pool_tuple in self.pools_with_states]
        self.pools_with_states = pools_with_states
        self.states: Iterable[UniswapV2PoolState | UniswapV3PoolState | CurveStableswapPoolState] = [pool_tuple[1] for pool_tuple in self.pools_with_states]
        self.distribution = None
        self.silent = silent
        self.cache = collections.OrderedDict()  # Cache to store calculated outputs for better performance
        self.cache_max_size = 100  # Set a limit to the cache size
        self.cache_expiry_time = 300  # Cache entries expire after 300 seconds
        
        if self.pools_with_states == None:
            raise ValueError("Either 'pools_with_states' or 'pools' must be provided!")
        
        
        if self.distribution == None and self.silent == False:
            print("no distribution found! swap with a input dustribution to access value")
        
        for pool in self.pools:
            if self.token_in not in pool.tokens:
                raise ValueError(f"Token {token_in.symbol} @ {token_in.address} is not in pool {pool.address} tokens! [Pool tokens: {pool.tokens}]")
        
        self.address = web3.Web3.keccak(
                        hexstr="".join(
                            [
                                pool.address[2:]
                                for pool in self.pools
                            ]
                        )
                    ).hex()

        self.pool_states: Dict[ChecksumAddress, AbstractPoolState] = {}
        self._update_pool_states(self.pools)
        # if self.max_input == None:
        #     self.max_input = 10*10**self.token_in.decimals
        # elif self.max_input != None:
        #     pass
        # self.max_input_dict = self.total_input_for_pool(self.max_input)
        
        
    def _update_pool_states(self, pools: Iterable[AbstractLiquidityPool]) -> None:
            """
            Update `self.pool_states` with state values from the `pools` iterable
            """
            self.pool_states.update({pool.address: pool.state for pool in pools})
            
    
    def maximum_inputs(self,  token_in: Erc20Token) -> Sequence[int]:
        maximum_inputs: list[int] = []

        for pool in self.pools:  
            match pool:
                case UniswapV2Pool() | CamelotLiquidityPool():
                    # V2 pools have no maximum input and can accept an "unlimited"
                    # deposit. However there is a real limit on the withdrawal amount
                    # (reserves - 1). So identify a concrete maximum by calculating
                    # the input that could swap for a maximum withdrawal.
                    zero_for_one = token_in == pool.token0
                    max_input = pool.calculate_tokens_in_from_tokens_out(
                        token_out_quantity=pool.state.reserves_token1 - 1
                        if zero_for_one
                        else pool.state.reserves_token0 - 1,
                        token_out=pool.token1 if zero_for_one else pool.token0,
                    )
                    maximum_inputs.append(max_input)
                    
                case UniswapV3Pool():
                    amount = 200_000_000_000_000 * 10**token_in.decimals
                    while True:
                        try:
                            pool.calculate_tokens_out_from_tokens_in(
                                token_in_quantity=amount,
                                token_in=token_in,
                            )
                            amount *= 100
                        except IncompleteSwap as e:
                            max_input = e.amount_in
                            print(max_input)
                            maximum_inputs.append(max_input)
                            break
                        
                        if amount > 10**40: 
                            raise RuntimeError("Konnte keine IncompleteSwap auslösen, obwohl riesige Mengen getestet wurden.")
                case _:
                    raise DegenbotTypeError(message=f"Pool type {type(pool)} not supported!")
                
        return maximum_inputs
            
    def total_amount_out(
        self,
        amounts_in: np.array,
        token_in: Erc20Token,
        pools: Sequence[AbstractLiquidityPool],
    ) -> float:
        print(f"Calculating amount out for inputs {amounts_in/10**self.token_in.decimals}")

        result = 0
        for amount_in, pool in zip(amounts_in, pools, strict=True):
            _amount_in = int(amount_in)
            if _amount_in <= 0:
                continue
            amount_out = pool.calculate_tokens_out_from_tokens_in(
                token_in=token_in,
                token_in_quantity=_amount_in,
            )
            result += amount_out
        print(f"{result/10**self.token_out.decimals=}")
        return -float(result)

        
    def optimize_combined_swap(self, total_amount_in):
        start_time = time.time()
        self.total_amount_in = total_amount_in

        maximum_input_balance = self.total_amount_in
        max_inputs = self.maximum_inputs(token_in=self.token_in)
        
        bounds = [
            (
                1,
                min(maximum_input_balance,max_input_to_pool)
            )
            for max_input_to_pool in max_inputs
        ]
        print(f"{bounds=}")
        sum_max = sum(max_inputs)
        ratios = [m / sum_max for m in max_inputs]
        
        # => initial_guess[i] = total_input * ratios[i]
        initial_guess = [maximum_input_balance * r for r in ratios]        
        objective_function = functools.partial(self.total_amount_out, token_in=self.token_in, pools=self.pools)

        # Ensure total equals total_amount_in
        result = minimize(
            fun=objective_function,
            x0=initial_guess,
            bounds=bounds,
            constraints={
                "type": "eq",
                "fun": lambda inputs: maximum_input_balance - np.sum(inputs),
            },
            # method='SLSQP',
            # tol=1.0,
            options={"disp": True,}
        )

        end_time = time.time()
        if not self.silent:
            print(f"Time taken for optimize_combined_swap: {end_time - start_time:.4f} seconds")
        print("Got results:")
        for i, swap_input in enumerate(result.x):
            print(f"Swap {int(swap_input)/10**self.token_in.decimals} at pool {i}") 
            
        distribution = {}
        for pool, swap_input in zip(self.pools , result.x):
            print(f"Swap {int(swap_input)/10**self.token_in.decimals} at pool {pool.address}") 
            distribution[pool.address] = int(swap_input)
            

        print(f"Total input : {int(np.sum(result.x))/10**self.token_in.decimals} {self.token_in}")
        print(f"Total output: {int(-result.fun)/10**self.token_out.decimals} {self.token_out}")
        print(f"{np.sum(result.x)} <= {maximum_input_balance=}")
        
        return distribution, int(-result.fun)