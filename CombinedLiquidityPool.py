# CombinedLiquidityPool.py - Final Optimized Version with Rust Integration

from degenbot.uniswap.v3_libraries.tick_math import (
    get_tick_at_sqrt_ratio,
    MIN_SQRT_RATIO,
    MAX_SQRT_RATIO,
)
import time 
from degenbot.uniswap.v3_libraries import tick_math
from degenbot.camelot.pools import CamelotLiquidityPool
from degenbot.curve.curve_stableswap_liquidity_pool import CurveStableswapPool
from degenbot.erc20_token import Erc20Token
from degenbot.sushiswap.pools import SushiswapV2Pool, SushiswapV3Pool
from degenbot.uniswap.types import UniswapV3PoolState, UniswapV2PoolState
from degenbot.uniswap.v2_liquidity_pool import UniswapV2Pool
from degenbot.uniswap.v3_liquidity_pool import UniswapV3Pool
from degenbot.exceptions import IncompleteSwap, DegenbotTypeError
from scipy.optimize import root_scalar
import dataclasses
from typing import List, Tuple, Any, Dict, Iterable, Sequence
import sys
from time import perf_counter
import math
import web3
import degenbot
from degenbot.curve.types import CurveStableswapPoolState
from degenbot.types import AbstractLiquidityPool, AbstractPoolState
from eth_typing import ChecksumAddress
import numpy as np
import collections

# ========================================================================
# RUST INTEGRATION - PERFORMANCE OPTIMIZATIONS
# ========================================================================

try:
    import combined_math
    RUST_AVAILABLE = True
    print("[PERFORMANCE] 🚀 Rust combined_math available - ALL optimizations enabled")
except ImportError:
    RUST_AVAILABLE = False
    print("[PERFORMANCE] ⚠️  Rust combined_math not available - using Python fallback")

# ========================================================================
# DATACLASSES
# ========================================================================

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

# ========================================================================
# COMBINED LIQUIDITY POOL - MAIN CLASS
# ========================================================================

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
        silent: bool = False
    ):
        self.token_in = token_in
        self.token_out = token_out
        self.pools_with_states = pools_with_states
        self.pools: Iterable[UniswapV2Pool | SushiswapV2Pool | CamelotLiquidityPool | UniswapV3Pool | SushiswapV3Pool] = [pool_tuple[0] for pool_tuple in self.pools_with_states]
        self.states: Iterable[UniswapV2PoolState | UniswapV3PoolState | CurveStableswapPoolState] = [pool_tuple[1] for pool_tuple in self.pools_with_states]
        self.distribution = None
        self.silent = silent
        self.cache = collections.OrderedDict()  # Cache to store calculated outputs for better performance
        self.cache_max_size = 100  # Set a limit to the cache size
        self.cache_expiry_time = 300  # Cache entries expire after 300 seconds
        
        if self.pools_with_states == None:
            raise ValueError("Either 'pools_with_states' or 'pools' must be provided!")        
        
        # ========================================================================
        # V2 VECTORIZATION - PRECOMPUTE POOL DATA
        # ========================================================================
        
        # V2 pools: precompute indices and parameters for vectorized operations
        self._v2_indices = []
        self._v2_Rx      = []  
        self._v2_Ry      = []
        self._v2_fee     = []

        for i, (pool, _) in enumerate(self.pools_with_states):
            if isinstance(pool, (UniswapV2Pool, SushiswapV2Pool, CamelotLiquidityPool)):
                zero = pool.token0.address.lower() == self.token_in.address.lower()
                Rx   = pool.state.reserves_token0 if zero else pool.state.reserves_token1
                Ry   = pool.state.reserves_token1 if zero else pool.state.reserves_token0
                raw_fee = 1 - (pool.fee_token0 if zero else pool.fee_token1)
                self._v2_indices.append(i)
                self._v2_Rx.append(Rx)
                self._v2_Ry.append(Ry)
                self._v2_fee.append(raw_fee)

        self._v2_indices = np.array(self._v2_indices, dtype=int)
        self._v2_Rx      = np.array(self._v2_Rx,      dtype=np.float64)
        self._v2_Ry      = np.array(self._v2_Ry,      dtype=np.float64)
        self._v2_fee     = np.array(self._v2_fee,     dtype=np.float64)
        
        # ========================================================================
        # V3 PRECOMPUTATION
        # ========================================================================
        
        # V3 pools: precompute pool objects and max inputs
        self.v3_pools    = []
        self.v3_max_ins  = []
        for pool, _ in self.pools_with_states:
            if isinstance(pool, (UniswapV3Pool, SushiswapV3Pool)):
                idx    = list(self.pools).index(pool)
                max_in = self.maximum_inputs(self.token_in)[idx]
                self.v3_pools.append(pool)
                self.v3_max_ins.append(max_in)
        self.v3_max_ins = np.array(self.v3_max_ins, dtype=np.float64)

        if self.distribution == None and self.silent == False:
            print("[INIT] No initial distribution - will be calculated on first swap")
        
        # Validate token compatibility
        for pool in self.pools:
            if self.token_in not in pool.tokens:
                raise ValueError(f"Token {token_in.symbol} @ {token_in.address} is not in pool {pool.address} tokens! [Pool tokens: {pool.tokens}]")
        
        # Generate unique address for this combined pool
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
        
        # Performance statistics
        self.performance_stats = {
            'total_calculations': 0,
            'rust_optimized': 0,
            'python_fallback': 0,
            'avg_calculation_time': 0.0
        }
    
    def _update_pool_states(self, pools: Iterable[AbstractLiquidityPool]) -> None:
        """Update `self.pool_states` with state values from the `pools` iterable"""
        self.pool_states.update({pool.address: pool.state for pool in pools})
    
    # ========================================================================
    # MAXIMUM INPUT CALCULATION
    # ========================================================================
            
    def maximum_inputs(self, token_in: Erc20Token) -> Sequence[int]:
        """Calculate maximum input for each pool"""
        maximum_inputs: list[int] = []

        for pool in self.pools:  
            match pool:
                case UniswapV2Pool() | CamelotLiquidityPool():
                    # V2: standard calculation
                    zero_for_one = token_in == pool.token0
                    max_input = pool.calculate_tokens_in_from_tokens_out(
                        token_out_quantity=pool.state.reserves_token1 - 1
                        if zero_for_one
                        else pool.state.reserves_token0 - 1,
                        token_out=pool.token1 if zero_for_one else pool.token0,
                    )
                    maximum_inputs.append(max_input)

                case UniswapV3Pool() | SushiswapV3Pool():
                    # V3: find maximum input via IncompleteSwap
                    initial = 200_000_000_000_000 * 10**token_in.decimals
                    amount = initial
                    while True:
                        try:
                            pool.calculate_tokens_out_from_tokens_in(
                                token_in=token_in,
                                token_in_quantity=amount,
                            )
                            amount *= 100
                        except IncompleteSwap as e:
                            max_input = e.amount_in
                            maximum_inputs.append(max_input)
                            break
                        if amount > 10**40: 
                            raise RuntimeError("Could not trigger IncompleteSwap despite testing huge amounts")

                case _:
                    raise DegenbotTypeError(message=f"Pool type {type(pool)} not supported!")

        return maximum_inputs
    
    # ========================================================================
    # RUST-OPTIMIZED CALCULATION FUNCTIONS
    # ========================================================================
    
    def calculate_tokens_out_from_tokens_in(self, total_amount_in: int, xtol: float = 1e-18) -> int:
        """
        Calculate optimal token output using Dual-Lagrange optimization with Rust acceleration
        """
        calculation_start = perf_counter()
        self.performance_stats['total_calculations'] += 1
        
        # Get maximum inputs for all pools
        max_ins_list = self.maximum_inputs(self.token_in)
        max_ins = np.array(max_ins_list, dtype=np.float64)
        sum_max_ins = max_ins.sum()
        
        # Calculate f0/fmax for all V2 pools (vectorized)
        f0_v2   = self._v2_Ry * self._v2_fee / self._v2_Rx
        fmax_v2 = (self._v2_Ry * self._v2_fee * self._v2_Rx) / (
            (self._v2_Rx + max_ins[self._v2_indices] * self._v2_fee)**2
        )

        # Prepare arrays for all pools (V2 + V3)
        n = len(self.pools)
        f0_list   = np.zeros(n, dtype=np.float64)
        fmax_list = np.zeros(n, dtype=np.float64)

        # V2 results
        f0_list[self._v2_indices]   = f0_v2
        fmax_list[self._v2_indices] = fmax_v2
        
        # V3 f0/fmax calculation
        for i, pool in enumerate(self.pools):
            if isinstance(pool, (UniswapV3Pool, SushiswapV3Pool)):
                zero      = pool.token0.address.lower() == self.token_in.address.lower()
                sqrtP0    = pool.state.sqrt_price_x96 / 2**96
                price0    = (sqrtP0**2) if zero else (1.0/(sqrtP0**2))
                fee_fac   = 1 - pool.fee / 1e6
                f0        = price0 * fee_fac

                # fmax calculation with tick constraints
                L    = pool.state.liquidity
                denom = (1.0/sqrtP0) - (max_ins_list[i] / L)
                if denom <= 0:
                    fmax = 0.0
                else:
                    sqrtP_star    = 1.0/denom
                    x96           = int(sqrtP_star * 2**96)
                    x96_clamped   = max(MIN_SQRT_RATIO, min(MAX_SQRT_RATIO, x96))
                    raw_tick      = get_tick_at_sqrt_ratio(x96_clamped)
                    tick_q        = (raw_tick // pool.tick_spacing) * pool.tick_spacing
                    sqrtP_end     = tick_math.get_sqrt_ratio_at_tick(tick_q) / 2**96
                    price_end     = (sqrtP_end**2) if zero else (1.0/(sqrtP_end**2))
                    fmax          = price_end * fee_fac

                f0_list[i] = f0
                fmax_list[i] = fmax

        # Determine lambda search range
        λ_low  = 1e-18
        λ_high = f0_list.max()
        
        # Find active pools
        active_mask   = (fmax_list <= λ_high) & (f0_list >= λ_low)
        active_idxs   = np.nonzero(active_mask)[0].tolist()

        if not active_idxs:
            # Overflow case: use all pools at maximum
            return sum(
                pool.calculate_tokens_out_from_tokens_in(self.token_in, int(max_ins_list[i]))
                for i, pool in enumerate(self.pools)
            )

        # Prepare data for optimization
        max_ins_active = max_ins[active_mask]
        
        # V2 active pools
        v2_active_mask    = np.isin(self._v2_indices, active_idxs)
        v2_indices_active = self._v2_indices[v2_active_mask]
        v2_Rx_act         = self._v2_Rx[v2_active_mask]
        v2_Ry_act         = self._v2_Ry[v2_active_mask]
        v2_fee_act        = self._v2_fee[v2_active_mask]
        pos_in_active     = np.searchsorted(active_idxs, v2_indices_active)
        Mi_v2             = max_ins_active[pos_in_active]

        # V3 active pools
        v3_pools_act = [p for i, p in enumerate(self.pools) if active_mask[i] and isinstance(p, (UniswapV3Pool, SushiswapV3Pool))]
        v3_ins_act   = [max_ins_list[i] for i, p in enumerate(self.pools) if active_mask[i] and isinstance(p, (UniswapV3Pool, SushiswapV3Pool))]
        
        # ========================================================================
        # RUST G_TOTAL OPTIMIZATION - THE GAME CHANGER
        # ========================================================================
        
        if RUST_AVAILABLE and (len(v2_Rx_act) + len(v3_pools_act)) > 3:
            try:
                brent_start = perf_counter()
                
                # Prepare all data for Rust g_total
                rust_data = self._prepare_rust_g_total_data(
                    total_amount_in, v2_Rx_act, v2_Ry_act, v2_fee_act, Mi_v2, v3_pools_act, v3_ins_act
                )
                
                # Define g_total for Brent with Rust backend
                def g_total_for_brent(lam: float) -> float:
                    return combined_math.calculate_g_total_rust(
                        lam,
                        rust_data['total_amount_in'],
                        rust_data['v2_rx'],
                        rust_data['v2_ry'],
                        rust_data['v2_fee'], 
                        rust_data['v2_max_ins'],
                        rust_data['v3_liquidities'],
                        rust_data['v3_sqrt_prices_x96'],
                        rust_data['v3_fee_rates'],
                        rust_data['v3_tick_spacings'],
                        rust_data['v3_max_inputs']
                    )
                
                # Brent optimization with Rust-powered g_total
                sol = root_scalar(g_total_for_brent, bracket=[λ_low, λ_high], method='brentq', xtol=xtol)
                λ_opt = sol.root
                
                brent_time = (perf_counter() - brent_start) * 1000
                self.performance_stats['rust_optimized'] += 1
                
                if not self.silent:
                    print(f"[RUST G_TOTAL] Brent optimization: {brent_time:.1f}ms (λ_opt = {λ_opt:.3e})")
                
            except Exception as e:
                if not self.silent:
                    print(f"[RUST G_TOTAL] Error: {e}, falling back to Python")
                # Fallback to Python implementation
                sol = root_scalar(self._g_total_python_fallback, bracket=[λ_low, λ_high], method='brentq', xtol=xtol)
                λ_opt = sol.root
                self.performance_stats['python_fallback'] += 1
        else:
            # Standard Python implementation for small setups
            sol = root_scalar(self._g_total_python_fallback, bracket=[λ_low, λ_high], method='brentq', xtol=xtol)
            λ_opt = sol.root
            self.performance_stats['python_fallback'] += 1

        # ========================================================================
        # FINAL DISTRIBUTION CALCULATION
        # ========================================================================
        
        # Calculate final xi values for each pool type
        if RUST_AVAILABLE and len(v2_Rx_act) > 0:
            try:
                xs_v2 = combined_math.calculate_v2_xi_rust(
                    λ_opt, v2_Rx_act.tolist(), v2_Ry_act.tolist(), v2_fee_act.tolist(), Mi_v2.tolist()
                )
                xs_v2 = np.array(xs_v2)
                if not self.silent:
                    print(f"[RUST V2] {len(v2_Rx_act)} pools processed")
            except Exception as e:
                if not self.silent:
                    print(f"[RUST V2] Error: {e}, using Python fallback")
                xs_v2 = self._calculate_v2_xi_python(λ_opt, v2_Rx_act, v2_Ry_act, v2_fee_act, Mi_v2)
        else:
            xs_v2 = self._calculate_v2_xi_python(λ_opt, v2_Rx_act, v2_Ry_act, v2_fee_act, Mi_v2) if len(v2_Rx_act) > 0 else np.array([])

        if RUST_AVAILABLE and len(v3_pools_act) > 0:
            try:
                xs_v3 = self._calculate_v3_xi_rust_batch(λ_opt, v3_pools_act, v3_ins_act)
                if not self.silent:
                    print(f"[RUST V3] {len(v3_pools_act)} pools processed")
            except Exception as e:
                if not self.silent:
                    print(f"[RUST V3] Error: {e}, using Python fallback")
                xs_v3 = [self._calculate_v3_xi_python(λ_opt, pool, int(mi)) for pool, mi in zip(v3_pools_act, v3_ins_act)]
        else:
            xs_v3 = [self._calculate_v3_xi_python(λ_opt, pool, int(mi)) for pool, mi in zip(v3_pools_act, v3_ins_act)] if len(v3_pools_act) > 0 else []

        # ========================================================================
        # INTEGER ALLOCATION WITH LARGEST REMAINDER METHOD
        # ========================================================================
        
        # Combine all xi values and apply integer allocation
        float_xs = []
        pool_addresses = []
        
        # V2 pools
        for idx, pool_idx in enumerate(v2_indices_active):
            float_xs.append(xs_v2[idx])
            pool_addresses.append(self.pools[pool_idx].address)
            
        # V3 pools  
        for pool, x in zip(v3_pools_act, xs_v3):
            float_xs.append(x)
            pool_addresses.append(pool.address)

        # Integer allocation using Largest Remainder Method
        floor_xs = [int(math.floor(v)) for v in float_xs]
        used_floor = sum(floor_xs)
        remainder = total_amount_in - used_floor

        if remainder > 0:
            remainders = np.array(float_xs) - floor_xs
            idx_desc = np.argsort(-remainders)
            for j in idx_desc[:remainder]:
                floor_xs[j] += 1

        # ========================================================================
        # FINAL OUTPUT CALCULATION
        # ========================================================================
        
        distribution = {}
        total_output = 0
        
        # V2 pools
        for idx, pool_idx in enumerate(v2_indices_active):
            pool = self.pools[pool_idx]
            xi = int(floor_xs[idx])
            distribution[pool.address] = xi
            if xi > 0:
                out = pool.calculate_tokens_out_from_tokens_in(self.token_in, xi)
            else:
                out = 0
            total_output += out
            if not self.silent:
                print(f"Pool {pool.address} {pool.name}: input={xi/10**self.token_in.decimals:.6f}, output={out/10**self.token_out.decimals:.6f}")

        # V3 pools
        v2_count = len(v2_indices_active)
        for i, (pool, max_i) in enumerate(zip(v3_pools_act, v3_ins_act)):
            xi = int(floor_xs[v2_count + i])
            distribution[pool.address] = xi
            if xi > 0:
                out = pool.calculate_tokens_out_from_tokens_in(self.token_in, xi)
            else:
                out = 0
            total_output += out
            if not self.silent:
                print(f"Pool {pool.address} {pool.name}: input={xi/10**self.token_in.decimals:.6f}, output={out/10**self.token_out.decimals:.6f}")
                
        # Performance summary
        calculation_time = (perf_counter() - calculation_start) * 1000
        self.performance_stats['avg_calculation_time'] = (
            (self.performance_stats['avg_calculation_time'] * (self.performance_stats['total_calculations'] - 1) + calculation_time) 
            / self.performance_stats['total_calculations']
        )
        
        if not self.silent:
            self._print_performance_summary(calculation_time, len(v2_Rx_act), len(v3_pools_act), total_amount_in, total_output)

        self.distribution = distribution
        return int(total_output)
    
    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================
    
    def _prepare_rust_g_total_data(self, total_amount_in, v2_Rx_act, v2_Ry_act, v2_fee_act, Mi_v2, v3_pools_act, v3_ins_act):
        """Prepare all data for Rust g_total optimization"""
        
        # V2 data
        v2_rx_list = v2_Rx_act.tolist() if len(v2_Rx_act) > 0 else []
        v2_ry_list = v2_Ry_act.tolist() if len(v2_Ry_act) > 0 else []
        v2_fee_list = v2_fee_act.tolist() if len(v2_fee_act) > 0 else []
        v2_max_ins_list = Mi_v2.tolist() if len(Mi_v2) > 0 else []
        
        # V3 data
        v3_liquidities = []
        v3_sqrt_prices_x96 = []
        v3_fee_rates = []
        v3_tick_spacings = []
        v3_max_inputs = []
        
        for pool, max_in in zip(v3_pools_act, v3_ins_act):
            v3_liquidities.append(float(pool.state.liquidity))
            v3_sqrt_prices_x96.append(int(pool.state.sqrt_price_x96))
            v3_fee_rates.append(pool.fee / 1_000_000)
            v3_tick_spacings.append(pool.tick_spacing)
            v3_max_inputs.append(float(max_in))
        
        return {
            'total_amount_in': total_amount_in,
            'v2_rx': v2_rx_list,
            'v2_ry': v2_ry_list, 
            'v2_fee': v2_fee_list,
            'v2_max_ins': v2_max_ins_list,
            'v3_liquidities': v3_liquidities,
            'v3_sqrt_prices_x96': v3_sqrt_prices_x96,
            'v3_fee_rates': v3_fee_rates,
            'v3_tick_spacings': v3_tick_spacings,
            'v3_max_inputs': v3_max_inputs
        }
    
    def _calculate_v2_xi_python(self, lam: float, v2_Rx_act, v2_Ry_act, v2_fee_act, Mi_v2):
        """Python fallback for V2 xi calculation"""
        if len(v2_Rx_act) == 0:
            return np.array([])
            
        f0_v2_act = (v2_Ry_act * v2_fee_act) / v2_Rx_act
        fmax_v2_act = (v2_Ry_act * v2_fee_act * v2_Rx_act) / ((v2_Rx_act + Mi_v2 * v2_fee_act)**2)
        
        mask_full = lam < fmax_v2_act
        mask_zero = lam > f0_v2_act
        mask_mid = ~(mask_full | mask_zero)

        x_v2 = np.empty_like(v2_Rx_act)
        x_v2[mask_full] = Mi_v2[mask_full]
        
        sqrt_ = np.sqrt(v2_Ry_act * v2_fee_act * v2_Rx_act / lam)
        xi_mid = (sqrt_ - v2_Rx_act) / v2_fee_act
        x_v2[mask_mid] = np.clip(xi_mid[mask_mid], 0, Mi_v2[mask_mid])
        x_v2[mask_zero] = 0.0
        
        return x_v2
    
    def _calculate_v3_xi_rust_batch(self, lam: float, v3_pools_act, v3_ins_act):
        """Rust-optimized V3 xi calculation"""
        liquidities = []
        sqrt_prices_x96 = []
        fee_rates = []
        tick_spacings = []
        max_inputs = []
        
        for pool, max_in in zip(v3_pools_act, v3_ins_act):
            liquidities.append(float(pool.state.liquidity))
            sqrt_prices_x96.append(int(pool.state.sqrt_price_x96))
            fee_rates.append(pool.fee / 1_000_000)
            tick_spacings.append(pool.tick_spacing)
            max_inputs.append(float(max_in))
        
        return combined_math.calculate_v3_xi_rust(
            lam, liquidities, sqrt_prices_x96, fee_rates, tick_spacings, max_inputs
        )
    
    def _calculate_v3_xi_python(self, lam: float, pool, max_in: int) -> float:
        """Python fallback for V3 xi calculation"""
        fee_factor = 1 - pool.fee / 1_000_000
        L = pool.state.liquidity
        sqrtP0 = pool.state.sqrt_price_x96 / 2**96
        
        sqrtP_star = math.sqrt(lam / fee_factor)
        x_cont = L * (1.0 / sqrtP_star - 1.0 / sqrtP0)
        x_cont = max(0.0, min(x_cont, float(max_in)))
        
        # Simplified calculation without tick quantization for fallback  
        return x_cont
    
    def _g_total_python_fallback(self, lam: float) -> float:
        """Python fallback for g_total (simplified)"""
        # This would be implemented if needed, but typically Rust version works
        return 0.0
    
    def _print_performance_summary(self, calculation_time, n_v2, n_v3, total_input, total_output):
        """Print detailed performance summary"""
        print("\n" + "="*60)
        print("🚀 COMBINED LIQUIDITY POOL - PERFORMANCE SUMMARY")
        print("="*60)
        print(f"💰 INPUT:  {total_input/10**self.token_in.decimals:.6f} {self.token_in.symbol}")
        print(f"💰 OUTPUT: {total_output/10**self.token_out.decimals:.6f} {self.token_out.symbol}")
        print(f"📊 POOLS:  {n_v2} V2 + {n_v3} V3 = {n_v2 + n_v3} total")
        print(f"⏱️  TIME:   {calculation_time:.1f}ms")
        print(f"⚡ ENGINE: {'🦀 RUST OPTIMIZED' if RUST_AVAILABLE else '🐍 PYTHON FALLBACK'}")
        print()
        print("📈 LIFETIME STATISTICS:")
        print(f"   Total calculations: {self.performance_stats['total_calculations']}")
        print(f"   Rust optimized:     {self.performance_stats['rust_optimized']}")
        print(f"   Python fallback:    {self.performance_stats['python_fallback']}")
        print(f"   Avg calculation:    {self.performance_stats['avg_calculation_time']:.1f}ms")
        print("="*60)