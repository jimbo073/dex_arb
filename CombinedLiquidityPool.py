from degenbot.uniswap.v3_libraries.tick_math import (
    get_tick_at_sqrt_ratio,
    MIN_SQRT_RATIO,
    MAX_SQRT_RATIO,
)
import combined_math
from degenbot.uniswap.v3_libraries import tick_math
from degenbot.camelot.pools import CamelotLiquidityPool
from degenbot.curve.curve_stableswap_liquidity_pool import CurveStableswapPool
from degenbot.erc20_token import Erc20Token
from degenbot.sushiswap.pools import SushiswapV2Pool, SushiswapV3Pool
from degenbot.uniswap.types import UniswapV3PoolState, UniswapV2PoolState
from degenbot.uniswap.v2_liquidity_pool import UniswapV2Pool
from degenbot.uniswap.v3_liquidity_pool import UniswapV3Pool
# from functools import lru_cache
from degenbot.exceptions import IncompleteSwap, DegenbotTypeError
from scipy.optimize import  root_scalar
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

sys.path.append(r'C:\Users\PC\Projects\degenbot')

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
        
        # === V2 vectorization start ===
        
        # gleiche Indizes wie in self.pools
        self._v2_indices = []
        self._v2_Rx      = []
        self._v2_Ry      = []
        self._v2_fee     = []

        for i, (pool, _) in enumerate(self.pools_with_states):
            if isinstance(pool, (UniswapV2Pool, SushiswapV2Pool, CamelotLiquidityPool)):
                zero = pool.token0.address.lower() == self.token_in.address.lower()
                Rx   = pool.state.reserves_token0 if zero else pool.state.reserves_token1
                Ry   = pool.state.reserves_token1 if zero else pool.state.reserves_token0
                raw_fee =1 - (pool.fee_token0 if zero else pool.fee_token1)
                self._v2_indices.append(i)
                self._v2_Rx.append(Rx)
                self._v2_Ry.append(Ry)
                self._v2_fee.append(raw_fee)

        self._v2_indices = np.array(self._v2_indices, dtype=int)
        self._v2_Rx      = np.array(self._v2_Rx,      dtype=np.float64)
        self._v2_Ry      = np.array(self._v2_Ry,      dtype=np.float64)
        self._v2_fee     = np.array(self._v2_fee,     dtype=np.float64)
        # === V2 vectorization end ===
        # V3 preloads 
        self.v3_pools    = []
        self.v3_max_ins  = []
        for pool, _ in self.pools_with_states:
            if isinstance(pool, (UniswapV3Pool, SushiswapV3Pool)):
                idx    = self.pools.index(pool)
                max_in = self.maximum_inputs(self.token_in)[idx]
                self.v3_pools.append(pool)
                self.v3_max_ins.append(max_in)
        self.v3_max_ins = np.array(self.v3_max_ins, dtype=np.float64)

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
        
        
    def _update_pool_states(self, pools: Iterable[AbstractLiquidityPool]) -> None:
            """
            Update `self.pool_states` with state values from the `pools` iterable
            """
            self.pool_states.update({pool.address: pool.state for pool in pools})
            
    
    def maximum_inputs(self, token_in: Erc20Token) -> Sequence[int]:
        maximum_inputs: list[int] = []

        for pool in self.pools:  
            match pool:
                case UniswapV2Pool() | CamelotLiquidityPool():
                    # V2: unverändert
                    zero_for_one = token_in == pool.token0
                    max_input = pool.calculate_tokens_in_from_tokens_out(
                        token_out_quantity=pool.state.reserves_token1 - 1
                        if zero_for_one
                        else pool.state.reserves_token0 - 1,
                        token_out=pool.token1 if zero_for_one else pool.token0,
                    )
                    print(f"[MAXIN V2] pool={pool.address}, max_in={max_input}")
                    maximum_inputs.append(max_input)

                case UniswapV3Pool() | SushiswapV3Pool():
                    # V3: suche maximalen Input per IncompleteSwap
                    initial = 200_000_000_000_000 * 10**token_in.decimals
                    amount = initial
                    print(f"[MAXIN V3] pool={pool.address} – starte Suche mit amount={amount}")
                    while True:
                        try:
                            pool.calculate_tokens_out_from_tokens_in(
                                token_in=token_in,
                                token_in_quantity=amount,
                            )
                            print(f"[MAXIN V3] pool={pool.address} – amount {amount} OK, *100 →")
                            amount *= 100
                        except IncompleteSwap as e:
                            max_input = e.amount_in
                            print(f"[MAXIN V3] pool={pool.address} – IncompleteSwap bei amount={amount}, e.amount_in={max_input}")
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


    def calculate_tokens_out_from_tokens_in(self, total_amount_in: int, xtol: float = 1e-18) -> int:
        """
        : Dual‐Lagrange mittels Brent’s root_scalar:
        Wir finden λ so, dass sum_i x_i(λ) = total_amount_in.
        Für V2 haben wir eine geschlossene Formel für x_i(λ),
        für V3 nutzen wir Closed‑Form mit Tick‑Quantisierung.
        """
        start = perf_counter()
        v2_idx,v2_Rx,v2_Ry, v2_fee = self._v2_indices, self._v2_Rx, self._v2_Ry, self._v2_fee
        pools = self.pools
       
       # 1) Max‐Inputs als NumPy-Array
        max_ins_list = self.maximum_inputs(self.token_in)
        # print(f"[DEBUG] max_ins_list (Wei) = {[int(mi) for mi in max_ins_list]}")
        
        max_ins  = np.array(max_ins_list, dtype=np.float64)
        sum_max_ins  = max_ins.sum()
        
        # 1) Nach dem Erzeugen von max_ins_list:
        max_ins = np.array(max_ins_list, dtype=np.float64)

        # 2) f0/fmax für alle V2-Pools vektoriell berechnen:
        f0_v2   = v2_Ry * v2_fee / v2_Rx
        fmax_v2 = (v2_Ry * v2_fee * v2_Rx) / (
            (v2_Rx + max_ins[v2_idx] * v2_fee)**2
        )

        # 3) Leere Arrays für alle Pools vorbereiten:
        n = len(self.pools)
        f0_list   = np.zeros(n, dtype=np.float64)
        fmax_list = np.zeros(n, dtype=np.float64)

        # 4) V2-Ergebnisse an den richtigen Positionen eintragen:
        f0_list[v2_idx]   = f0_v2
        fmax_list[v2_idx] = fmax_v2
        
        for i, pool in enumerate(pools):
            if isinstance(pool, (UniswapV3Pool, SushiswapV3Pool)):
                # ── V3 f0/fmax via TickMath ────────────────────────────────────────────
                zero      = pool.token0.address.lower() == self.token_in.address.lower()
                sqrtP0    = pool.state.sqrt_price_x96 / 2**96
                price0    = (sqrtP0**2) if zero else (1.0/(sqrtP0**2))
                fee_fac   = 1 - pool.fee / 1e6
                f0        = price0 * fee_fac

                # fmax: clampte sqrtP_star_x96 in [MIN_SQRT_RATIO, MAX_SQRT_RATIO] …
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
            else:
                # z.B. Curve pools: überspringen
                continue

            f0_list[i] = f0
            fmax_list[i] = fmax
            print(f"[DEBUG] Pool {pool.address}")
            print(f"        → f0={f0:.12e}, fmax={fmax:.12e}")

         # ─── Schritt 2: globalen λ-Suchraum einschränken ────────────────────────────
        λ_low  = 1e-18
        λ_high = f0_list.max()
        print(f"[BRACKET DEBUG] λ_low = {λ_low:.3e}, λ_high = {λ_high:.3e}")
        # ─── Schritt 3: boolean-Maske für aktive Pools ──────────────────────────────
        active_mask   = (fmax_list <= λ_high) & (f0_list >= λ_low)
        active_idxs   = np.nonzero(active_mask)[0].tolist()
        print(f"[BRACKET DEBUG] active_idxs = {active_idxs}")

        if not active_idxs:
            # Overflow-Fall: volles Volumen über alle Pools
            return sum(
                pool.calculate_tokens_out_from_tokens_in(self.token_in, int(max_ins_list[i]))
                for i, pool in enumerate(self.pools)
            )
        # jetzt nur noch die aktiven max_ins
        max_ins_active = max_ins[active_mask]

        # V2: aktive Indizes und Parameter extrahieren
        v2_active_mask    = np.isin(self._v2_indices, active_idxs)
        v2_indices_active = self._v2_indices[v2_active_mask]
        v2_Rx_act         = v2_Rx[v2_active_mask]
        v2_Ry_act         = v2_Ry[v2_active_mask]
        v2_fee_act        = v2_fee[v2_active_mask]
        # Mi für V2: indexiere max_ins_active via Position in active_idxs
        pos_in_active     = np.searchsorted(active_idxs, v2_indices_active)
        Mi_v2             = max_ins_active[pos_in_active]

        # V3-Pools und -Inputs
        v3_pools_act = [p for i,p in enumerate(pools) if active_mask[i] and isinstance(p, (UniswapV3Pool, SushiswapV3Pool))]
        v3_ins_act   = [max_ins_list[i] for i,p in enumerate(pools) if active_mask[i] and isinstance(p, (UniswapV3Pool, SushiswapV3Pool))]


        def xi_for_v2_vec(lam: float, max_ins: np.ndarray) -> np.ndarray:
            Mi   = max_ins[np.searchsorted(active_idxs, v2_idx)]
            f0_act  = (v2_Ry_act * v2_fee_act)  / v2_Rx_act
            fmax_act = (v2_Ry_act * v2_fee_act * v2_Rx_act) / ((v2_Rx_act + Mi*v2_fee_act)**2)
            x = np.zeros_like(v2_Rx_act)
            # Masken
            mask_full = lam < fmax_act   # komplett füllen
            mask_zero = lam > f0_act     # nullen
            mask_mid  = ~(mask_full | mask_zero)
            x[mask_full] = Mi[mask_full]
            # invertiere
            sqrt_ = np.sqrt(v2_Ry_act * v2_fee_act * v2_Rx_act / lam)
            x_int = (sqrt_ - v2_Rx_act) / v2_fee_act
            x[mask_mid] = np.clip(x_int[mask_mid], 0, Mi[mask_mid])
            return x

        def xi_for_v3(lam: float, pool: UniswapV3Pool | SushiswapV3Pool, max_in: int) -> float:
            fee_factor = 1 - pool.fee / 1_000_000
            # Liquidity und aktueller sqrt-Price
            L = pool.state.liquidity
            sqrtP0 = pool.state.sqrt_price_x96 / 2**96
            # 1) Ziel-sqrt-Preis aus lam: sqrtP* = sqrt(lam/fee)
            sqrtP_star = math.sqrt(lam / fee_factor)
            # 2) Continuous Input x_cont
            x_cont = L * (1.0 / sqrtP_star - 1.0 / sqrtP0)
            # Clamp
            x_cont = max(0.0, min(x_cont, float(max_in)))
            # 3) Exakter Tick via TickMath (Q64.96 → Tick)
            #    sqrtP_star hier im float-Bereich, konvertieren wir zurück in X96-Format:
            sqrtP_star_x96 = int(sqrtP_star * 2**96)
            raw_tick_rust = combined_math.get_tick_at_sqrt_ratio(sqrtP_star_x96)
            raw_tick = get_tick_at_sqrt_ratio(sqrtP_star_x96)
            print(f"python site sqrtP_star_x96: {sqrtP_star_x96} ")
            print(f"  raw_tick_rust    = {raw_tick_rust}")
            
            # jetzt auf Tick-Spacing abrunden
            tick = (raw_tick // pool.tick_spacing) * pool.tick_spacing
            # 4) Exaktes sqrt-Price des quantisierten Ticks (ebenfalls Q64.96)
            sqrtP_tick_str_rust = combined_math.get_sqrt_ratio_at_tick(tick)
            sqrtP_tick_str = tick_math.get_sqrt_ratio_at_tick(tick)            
            sqrtP_tick = sqrtP_tick_str/ 2**96
            # 5) Feintuning des Inputs auf Tick-Grenze
            x_final = L * (1.0 / sqrtP_tick - 1.0 / sqrtP0) / fee_factor
            x_final = max(0.0, min(x_final, float(max_in)))
            print(f"[V3 DEBUG PY] pool={pool.address}")
            print(f"  lam              = {lam:.6e}")
            print(f"  fee_factor       = {fee_factor:.6f}")
            print(f"  sqrtP_star       = {sqrtP_star:.6e}")
            print(f"  sqrtP_star_x96   = {sqrtP_star_x96}")
            print(f"  raw_tick_rust    = {raw_tick_rust}")
            print(f"  raw_tick_python  = {raw_tick}")
            print(f"  quantized tick   = {tick}")
            print(f"  sqrtP_tick_rust  = {sqrtP_tick_str_rust}")
            print(f"  sqrtP_tick_py    = {sqrtP_tick_str}")
            print(f"  x_cont           = {x_cont:.6e}")
            print(f"  x_final          = {x_final:.6e}")
            return x_final
        
        # ─── Schritt 4: g_total mit vektorisiertem V2 ───────────────────────────────
        def g_total(lam: float) -> float:
            print(f"[G_TOTAL DEBUG] lam = {lam:.3e}")
            # V2:
            f0_v2_act   = (v2_Ry_act * v2_fee_act)   / v2_Rx_act
            fmax_v2_act = (v2_Ry_act * v2_fee_act * v2_Rx_act) / ((v2_Rx_act + Mi_v2 * v2_fee_act)**2)
            mask_full   = lam < fmax_v2_act
            mask_zero   = lam > f0_v2_act
            mask_mid    = ~(mask_full | mask_zero)

            x_v2 = np.empty_like(v2_Rx_act)
            x_v2[mask_full] = Mi_v2[mask_full]
            # für die mittleren Pools:
            sqrt_    = np.sqrt(v2_Ry_act * v2_fee_act * v2_Rx_act / lam)
            xi_mid   = (sqrt_ - v2_Rx_act) / v2_fee_act
            x_v2[mask_mid] = np.clip(xi_mid[mask_mid], 0, Mi_v2[mask_mid])
            x_v2[mask_zero] = 0.0
            sum_v2 = x_v2.sum()
            print(f"[G_TOTAL DEBUG] sum_v2 = {sum_v2:.3e}")

            # V3 (kleiner Python-Loop)
            sum_v3 = 0.0
            for pool, mi in zip(v3_pools_act, v3_ins_act):
                x3 = xi_for_v3(lam, pool, int(mi))
                print(f"[G_TOTAL DEBUG] xi_v3 for {pool.address} = {x3:.3e}")
                sum_v3 += x3
            print(f"[G_TOTAL DEBUG] sum_v3 = {sum_v3:.3e}")

            return sum_v2 + sum_v3 - total_amount_in
        
        
        print(f"[BRACKET DEBUG] g(λ_low) = {g_total(λ_low):.3e}, g(λ_high) = {g_total(λ_high):.3e}")
        sol     = root_scalar(g_total, bracket=[λ_low, λ_high], method='brentq', xtol=xtol)
        λ_opt   = sol.root
        print(f"[RESULT DEBUG] λ_opt = {λ_opt:.3e}")
        # 6) Ausgabe verteilen und summieren
        xs_v2 = xi_for_v2_vec(λ_opt, max_ins)
        # Für V3 analog collecten
        xs_v3 = []
        for i, pool in enumerate(self.pools):
            if isinstance(pool, (UniswapV3Pool, SushiswapV3Pool)):
                xs_v3.append(xi_for_v3(λ_opt, pool, int(max_ins[i])))
        # === NEW: Floor + Largest Remainder Method (LRM) ===
        # Pools in einer Liste zusammenführen, in gleicher Reihenfolge wie active_idxs
        float_xs = []
        pool_addresses = []
        # V2-Pools
        for idx, pool_idx in enumerate(v2_indices_active):
            float_xs.append(xs_v2[idx])
            pool_addresses.append(self.pools[pool_idx].address)
        # V3-Pools
        for pool, x in zip(v3_pools_act, xs_v3):
            float_xs.append(x)
            pool_addresses.append(pool.address)

        floor_xs = [int(math.floor(v)) for v in float_xs]

        used_floor = sum(floor_xs)
        remainder  = total_amount_in - used_floor

        # 2) Reste berechnen und absteigend sortieren
        remainders = np.array(float_xs) - floor_xs
        # Indexe der größten Reste
        idx_desc = np.argsort(-remainders)

        # 3) Restbudget auf größten Reste verteilen
        for j in idx_desc[:remainder]:
            floor_xs[j] += 1

        # Jetzt gilt sum(floor_xs) == total_amount_in
        # =================================================

        # 7) Distribution mit floor_xs füllen
        distribution = {}
        for addr, xi in zip(pool_addresses, floor_xs):
            distribution[addr] = int(xi)
        total_output = 0

        # V2-Pools
        for idx, pool_idx in enumerate(v2_indices_active):
            pool = self.pools[pool_idx]
            xi   = int(xs_v2[idx])
            distribution[pool.address] = xi
            if xi > 0:
                out = pool.calculate_tokens_out_from_tokens_in(self.token_in, xi)
            else:
                out = 0
            total_output += out
            if not self.silent:
                print(f"Pool {pool.address} {pool.name}: input={xi/10**self.token_in.decimals:.6f}, output={out/10**self.token_out.decimals:.6f}")

        # V3-Pools
        for pool, max_i in zip(v3_pools_act, v3_ins_act):
            xi = int(xi_for_v3(λ_opt, pool, max_i))
            distribution[pool.address] = xi
            if xi > 0:
                out = pool.calculate_tokens_out_from_tokens_in(self.token_in, xi)
            else:
                out = 0
            total_output += out
            if not self.silent:
                print(f"Pool {pool.address} {pool.name}: input={xi/10**self.token_in.decimals:.6f}, output={out/10**self.token_out.decimals:.6f}")
                
        end = perf_counter()
        total_input_used = sum(distribution.values())
        # ─── DEBUG: Vollständige Konsistenz-Checks ───────────────────────────────────
        if not self.silent:
            print("--- DEBUG SUMMARY ---")
            print("f0_list:          ", [f"{v:.3e}" for v in f0_list])
            print("fmax_list:        ", [f"{v:.3e}" for v in fmax_list])
            print(f"λ_lo = {λ_low:.3e}, λ_hi = {λ_high:.3e}")
            print("active_idxs:      ", active_idxs)
            print("max_ins_active:   ", [f"{v:.3e}" for v in max_ins_active])
            print(f"λ_optimal = {λ_opt:.3e}")
            # V2
            print("xs_v2 (per pool):", [f"{x:.3e}" for x in xs_v2])
            # V3
            print("xs_v3 (per pool):", [f"{x:.3e}" for x in xs_v3])
            print(f"sum(xs_v2) = {xs_v2.sum():.3e}, sum(xs_v3) = {sum(xs_v3):.3e}")
            print(f"total_input_used = {total_input_used/10**self.token_in.decimals:.6f} {self.token_in.symbol}")
            print(f"total_output      = {total_output/10**self.token_out.decimals:.6f} {self.token_out.symbol}")
            print("Brent took", (end-start)*1e3, "ms")
            print("----------------------")

        self.distribution = distribution
        return int(total_output)

