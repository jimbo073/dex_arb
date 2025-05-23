import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from doctest import OutputChecker
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Awaitable, Dict, Iterable, List, Sequence, Tuple, TypeAlias

import eth_abi.abi
from eth_typing import ChecksumAddress
from eth_utils.address import to_checksum_address
from scipy.optimize import minimize_scalar
from web3 import Web3
from itertools import chain
from CombinedPoolStates import CombinedLiquidityPoolStates
from degenbot.types import (
    AbstractArbitrage,
    Message,
    AbstractLiquidityPool,
    AbstractPoolState,
    Publisher,
    Subscriber,
)
from degenbot.uniswap.types import UniswapSimulationResult
from degenbot.config import get_web3
from degenbot.constants import MAX_UINT256
from degenbot.curve.types import CurveStableswapPoolState
from degenbot.curve.curve_stableswap_liquidity_pool import CurveStableswapPool
from degenbot.erc20_token import Erc20Token
from degenbot.exceptions import ArbitrageError, EVMRevertError, LiquidityPoolError, NoLiquidity
from degenbot.logging import logger
from degenbot.uniswap.types import UniswapV2PoolSimulationResult, UniswapV2PoolState
from degenbot.uniswap.v2_liquidity_pool import UniswapV2Pool
from degenbot.camelot.pools import CamelotLiquidityPool
from degenbot.uniswap.types import UniswapV3PoolSimulationResult, UniswapV3PoolState
from degenbot.uniswap.v3_libraries import tick_math
from degenbot.uniswap.v3_liquidity_pool import UniswapV3Pool
from degenbot.arbitrage.types import (
    ArbitrageCalculationResult,
    CurveStableSwapPoolSwapAmounts,
    CurveStableSwapPoolVector,
    UniswapPoolSwapVector,
    UniswapV2PoolSwapAmounts,
    UniswapV3PoolSwapAmounts,
)
from CombinedLiquidityPool import CombinedLiquidityPool,CombinedSwapVector,CombinedSwapAmounts,CombinedLiquidityArbitrageCalculationResult
from dex_arb.arbitrum import CombinedPoolStates


SwapAmount: TypeAlias = (
    CurveStableSwapPoolSwapAmounts | UniswapV2PoolSwapAmounts | UniswapV3PoolSwapAmounts | CombinedSwapAmounts
)
# Default discount applied to amount received.
# This masks small differences in get_dy() vs exchange().
CURVE_V1_DEFAULT_DISCOUNT_FACTOR = 0.9999

class CombinedArbitrage(AbstractLiquidityPool):
    def _notify_subscribers(self: Publisher, message: Message) -> None:
        for subscriber in self._subscribers:
            subscriber.notify(publisher=self, message=message)

    def __init__(
        self,
        input_token: Erc20Token,
        swap_pools: Iterable[
            UniswapV2Pool | UniswapV3Pool | CurveStableswapPool | CombinedLiquidityPool
            ],
        id: str,
        max_input: int | None = None,
    ):
        # Überprüfen, ob der letzte Pool vom Typ CombinedLiquidityPool ist
        if not isinstance(swap_pools[-1], CombinedLiquidityPool):
            raise ValueError("The last pool must be a CombinedLiquidityPool.")
        self.last_pool = swap_pools[-1]
        self.swap_pools = tuple(swap_pools)
        self.name = " → ".join([pool.name for pool in self.swap_pools])

        self.pool_states: Dict[ChecksumAddress, AbstractPoolState] = {}
        self._update_pool_states(self.swap_pools)

        # Standard-Discount-Faktor für Curve (wenn Curve-Pools verwendet werden)
        self.curve_discount_factor = CURVE_V1_DEFAULT_DISCOUNT_FACTOR

        for pool in swap_pools:
            pool.subscribe(self)

        self.id = id
        self.input_token = input_token

        if max_input == 0:
            raise ValueError("Maximum input must be positive.")

        if max_input is None:
            logger.warning("No maximum input provided, setting to 100 WETH")
            max_input = 100 * 10**18
        self.max_input = max_input

        self.gas_estimate: int
        # Erstellen der Swap-Vektoren für den Arbitrage-Pfad
        _swap_vectors: List[UniswapPoolSwapVector | CurveStableSwapPoolVector | CombinedSwapVector] = []

        for i, pool in enumerate(self.swap_pools):
            match pool:
                case UniswapV2Pool() | UniswapV3Pool() | CamelotLiquidityPool():
                    if i == 0:
                        if self.input_token == pool.token0:
                            token_in = pool.token0
                            token_out = pool.token1
                            zero_for_one = True
                        elif self.input_token == pool.token1:
                            token_in = pool.token1
                            token_out = pool.token0
                            zero_for_one = False
                        else:
                            raise ValueError("Input token could not be identified!")
                    else:
                        if token_out == pool.token0:
                            token_in = pool.token0
                            token_out = pool.token1
                            zero_for_one = True
                        elif token_out == pool.token1:
                            token_in = pool.token1
                            token_out = pool.token0
                            zero_for_one = False
                        else:
                            raise ValueError("Input token could not be identified!")

                    _swap_vectors.append(
                        UniswapPoolSwapVector(
                            token_in=token_in,
                            token_out=token_out,
                            zero_for_one=zero_for_one,
                        )
                    )

                # case CurveStableswapPool():
                #     if i != 1:
                #         raise ValueError(f"Not implemented for Curve pools at position != 1, {i=}, {pool=}, {self.id=}")

                #     token_in = token_out
                    
                #     next_pool = self.swap_pools[i + 1]
                    
                #     if isinstance(next_pool, CombinedLiquidityPool):
                #         # Falls der nächste Pool ein CombinedLiquidityPool ist, extrahiere alle Tokens aus den darin enthaltenen Pools
                #         next_pool_tokens = set(chain.from_iterable(p[0].tokens for p in next_pool.pools)) 
                #     else:
                #         # Falls der nächste Pool ein einzelner Pool ist, wie in der bisherigen Implementierung
                #         next_pool_tokens = set(next_pool.tokens)
                    
                #     # Bestimme die gemeinsamen Tokens zwischen dem Curve Pool und dem nächsten Pool (oder den nächsten Pools)
                #     shared_tokens = list(set(pool.tokens).intersection(next_pool_tokens))
                #     assert len(shared_tokens) > 0, f"this: {pool.tokens}, next: {next_pool_tokens}"

                #     token_out = shared_tokens[0]  # Wähle den ersten gemeinsamen Token aus
                    
                #     _swap_vectors.append(
                #         CurveStableSwapPoolVector(token_in=token_in, token_out=token_out)
                #     )

                case CombinedLiquidityPool(): # TODO <----------------------------------------------------------------------
                    token_in = token_out
                    if i != len(self.swap_pools) - 1: # <----------------- TODO erweitern aauf den letzten pool von self.swap_pools
                        raise ValueError(f"Not implemented for Curve pools at position != 1, {i=}, {pool=}, {self.id=}")
                    # Der letzte Swap verwendet kombinierte Liquidität
                    _swap_vectors.append(
                        CombinedSwapVector(
                            token_in=token_in, # token_out from pool before
                            token_out=self.input_token, # profit token
                            involved_pools=[part_pool for part_pool in pool.pools] # List of all pool addresses in CombinedLiquidityPool
                        )
                    )
                    
                case _:  # pragma: no cover
                    raise ValueError("Pool type could not be identified")
                    
        self._swap_vectors = tuple(_swap_vectors)
    
    
    def _sort_overrides(
        self,
        overrides: Sequence[
            Tuple[
                AbstractLiquidityPool,
                AbstractPoolState | UniswapSimulationResult | CombinedLiquidityPoolStates,
            ]
        ]
        | None,
    ) -> Dict[ChecksumAddress, AbstractPoolState]:
        """
        Validiert die Overrides und extrahiert die Pool-Zustände in ein Dictionary.
        """

        if overrides is None:
            return {}

        sorted_overrides: Dict[ChecksumAddress, AbstractPoolState] = {}

        for pool, override in overrides:
            if isinstance(
                override,
                (UniswapV2PoolState, UniswapV3PoolState, CurveStableswapPoolState),
            ):
                # Wenn es ein direkter Zustand für einen Pool ist, speichere ihn direkt
                logger.debug(f"Applying override {override} to {pool}")
                sorted_overrides[pool.address] = override

            elif isinstance(
                override,
                (UniswapV2PoolSimulationResult, UniswapV3PoolSimulationResult),
            ):
                # Für Simulationsergebnisse verwenden wir den finalen Zustand
                logger.debug(f"Applying override {override.final_state} to {pool}")
                sorted_overrides[pool.address] = override.final_state

            elif isinstance(
                override,
                CombinedLiquidityPoolStates
            ):
                # Falls es sich um einen CombinedLiquidityPool handelt, erstellen wir ein Objekt zum Speichern der Poolzustände
                match pool:
                    case CombinedLiquidityPool(): 
                        for part_pool, state in zip(pool.pools, override):
                            if isinstance(part_pool, UniswapV2Pool | CamelotLiquidityPool) and isinstance(state, UniswapV2PoolState):
                                # Uniswap V2 Pool State hinzufügen
                                override.add_uniswap_v2_pool_state(
                                    pool_address=part_pool.address,
                                    state=state,
                                )
                            elif isinstance(part_pool, UniswapV3Pool) and isinstance(state, UniswapV3PoolState):
                                # Uniswap V3 Pool State hinzufügen
                                override.add_uniswap_v3_pool_state(
                                    pool_address=part_pool.address,
                                    state=state,
                                )
                            elif isinstance(part_pool, CurveStableswapPool) and isinstance(state, CurveStableswapPoolState):
                                # Curve Pool State hinzufügen
                                override.add_curve_pool_state(
                                    pool_address=part_pool.address,
                                    state=state,
                                )

                # Speichere den CombinedLiquidityPool in den Overrides
                sorted_overrides[pool.address] = override

            else:
                raise ValueError(f"Override for {pool} has unsupported type {type(override)}")
        return sorted_overrides

    def _build_amounts_out(
        self,
        token_in: Erc20Token,
        token_in_quantity: int,
        pool_state_overrides: Dict[ChecksumAddress, AbstractPoolState] | None = None,
        block_number: int | None = None,
    ) -> List[SwapAmount]:
        """
        Berechnet die Eingabe- und Ausgangswerte für die Swaps entlang des Arbitragepfads.
        :param token_in: Eingabe-Token.
        :param token_in_quantity: Eingabemenge des Tokens.
        :param pool_state_overrides: Optionale Zustandsüberschreibungen für die Pools.
        :param block_number: Die Blocknummer, um den Zustand eines Curve-Pools zu berechnen.
        :return: Eine Liste der Swap-Mengen für jeden Pool.
        """

        if pool_state_overrides is None:
            pool_state_overrides = {}

        pools_amounts_out: List[SwapAmount] = []
        _token_in_quantity: int = 0
        _token_out_quantity: int = 0

        for i, (pool, swap_vector) in enumerate(zip(self.swap_pools, self._swap_vectors)):
            match pool:
                case UniswapV2Pool() | UniswapV3Pool() | CamelotLiquidityPool():
                    assert isinstance(swap_vector, UniswapPoolSwapVector)
                    token_in = swap_vector.token_in
                    token_out = swap_vector.token_out
                    zero_for_one = swap_vector.zero_for_one

                case CurveStableswapPool():
                    assert isinstance(swap_vector, CurveStableSwapPoolVector)
                    token_in = swap_vector.token_in
                    token_out = swap_vector.token_out

                case CombinedLiquidityPool():
                    assert isinstance(swap_vector, CombinedSwapVector)
                    token_in = swap_vector.token_in
                    token_out = swap_vector.token_out
                    
                case _:  # pragma: no cover
                    raise ValueError("Swap Vector could not be identified")

            if i == 0:
                _token_in_quantity = token_in_quantity
            else:
                _token_in_quantity = _token_out_quantity

            try:
                match pool:
                    case UniswapV2Pool() | CamelotLiquidityPool():
                        pool_state_override = pool_state_overrides.get(pool.address)
                        _token_out_quantity = pool.calculate_tokens_out_from_tokens_in(
                            token_in=token_in,
                            token_in_quantity=_token_in_quantity,
                            override_state=pool_state_override,
                        )

                    case UniswapV3Pool():
                        pool_state_override = pool_state_overrides.get(pool.address)
                        _token_out_quantity = pool.calculate_tokens_out_from_tokens_in(
                            token_in=token_in,
                            token_in_quantity=_token_in_quantity,
                            override_state=pool_state_override,
                        )

                    case CurveStableswapPool():
                        pool_state_override = pool_state_overrides.get(pool.address)
                        _token_out_quantity = int(
                            self.curve_discount_factor
                            * pool.calculate_tokens_out_from_tokens_in(
                                token_in=token_in,
                                token_out=token_out,
                                token_in_quantity=_token_in_quantity,
                                override_state=pool_state_override,
                                block_identifier=block_number,
                            )
                        )

                    case CombinedLiquidityPool():
                        # Verwende die `test_combinations` Methode, um den besten Output und die beste Verteilung zu finden
                        distribution, best_output = pool.calculate_tokens_out_from_tokens_in(_token_in_quantity)
                        _token_out_quantity = best_output  # Der maximale Output durch die Kombination
                    
                    case _:  # pragma: no cover
                        raise ValueError("Unsupported pool type",pool)

            except LiquidityPoolError as e:
                raise ArbitrageError(f"(calculate_tokens_out_from_tokens_in): {e}")
            else:
                if _token_out_quantity == 0:
                    raise ArbitrageError(f"Zero-output swap through pool {pool} @ {pool.address}")

            # Speichere die Swap-Daten für jeden Pool
            match pool:
                case UniswapV2Pool() | CamelotLiquidityPool():
                    pools_amounts_out.append(
                        UniswapV2PoolSwapAmounts(
                            pool=pool.address,
                            amounts_in=(_token_in_quantity, 0)
                            if zero_for_one
                            else (0, _token_in_quantity),
                            amounts_out=(0, _token_out_quantity)
                            if zero_for_one
                            else (_token_out_quantity, 0),
                        )
                    )

                case UniswapV3Pool():
                    pools_amounts_out.append(
                        UniswapV3PoolSwapAmounts(
                            pool=pool.address,
                            amount_specified=_token_in_quantity,
                            zero_for_one=zero_for_one,
                            sqrt_price_limit_x96=tick_math.MIN_SQRT_RATIO + 1
                            if zero_for_one
                            else tick_math.MAX_SQRT_RATIO - 1,
                        )
                    )

                case CurveStableswapPool():
                    pools_amounts_out.append(
                        CurveStableSwapPoolSwapAmounts(
                            token_in=token_in,
                            token_in_index=pool.tokens.index(token_in),
                            token_out=token_out,
                            token_out_index=pool.tokens.index(token_out),
                            amount_in=_token_in_quantity,
                            min_amount_out=_token_out_quantity,
                            underlying=True
                            if (
                                pool.is_metapool
                                and (
                                    token_in in pool.tokens_underlying
                                    or token_out in pool.tokens_underlying
                                )
                            )
                            else False,
                        )
                    )

                case CombinedLiquidityPool():
                    # Füge die kombinierten Swap-Mengen hinzu, basierend auf der besten Kombination
                    pools_amounts_out.append(
                        CombinedSwapAmounts(
                            swap_amounts=[(_token_in_quantity, _token_out_quantity)],
                            total_input=_token_in_quantity,
                            total_output=_token_out_quantity,
                            pool_combination_distribution=pool.distribution
                        )
                    )
        return pools_amounts_out
    
    
    def _pre_calculation_check(
        self,
        override_state: Sequence[
            Tuple[
                AbstractLiquidityPool,
                  AbstractPoolState | UniswapSimulationResult
            ]
        ]
        | None = None,
    ) -> None:
        state_overrides = self._sort_overrides(override_state)

        # Ein Skalierungswert, der die Nettomenge des Eingabetokens im gesamten Pfad darstellt (einschließlich Gebühren).
        profit_factor: float = 1.0
        rates: list[float] = []
        # Überprüfe jeden Pool auf Liquidität und berechne den Preis und die Gebühr für den Trade
        for pool, vector in zip(self.swap_pools, self._swap_vectors):
            pool_state = state_overrides.get(pool.address) or pool.state | pool.states

            match pool, pool_state, vector:
                case UniswapV2Pool() | CamelotLiquidityPool(), UniswapV2PoolState(), UniswapPoolSwapVector():
                        
                    if pool_state.reserves_token0 == 0 or pool_state.reserves_token1 == 0:
                        raise NoLiquidity(f"V2 pool {pool.address} has no liquidity")
                    price = pool_state.reserves_token1 / pool_state.reserves_token0
                    fee = pool.fee_token0 if vector.zero_for_one else pool.fee_token1
                    profit_factor *= (price if vector.zero_for_one else 1 / price) * (
                        (fee.denominator - fee.numerator) / fee.denominator
                    )

                case UniswapV3Pool(), UniswapV3PoolState(), UniswapPoolSwapVector() :
    
                    if pool_state.sqrt_price_x96 == 0:
                        raise NoLiquidity(f"V3 pool {pool.address} has no liquidity")
                    price = pool_state.sqrt_price_x96**2 / (2**192)
                    fee = Fraction(pool.fee, 1000000)
                    profit_factor *= (price if vector.zero_for_one else 1 / price) * (
                        (fee.denominator - fee.numerator) / fee.denominator
                    )

                # case CurveStableswapPool(), _, _:
                #     price = 1.0 * (10**vector.token_out.decimals) / (10**vector.token_in.decimals)
                #     fee = Fraction(pool.fee, pool.FEE_DENOMINATOR)
                #     profit_factor *= price * ((fee.denominator - fee.numerator) / fee.denominator)

                case CombinedLiquidityPool(), CombinedLiquidityPoolStates(), CombinedSwapVector(): # TODO ---------------------------------------------------------------------------------------------------------------------------------------------------
                    # Da der CombinedLiquidityPool mehrere Pools kombiniert, müssen wir die Liquidität und Preise
                    # über alle Pools hinweg prüfen.
                    for part_pool, part_state in zip(pool.pools, pool.states):
                        match part_pool, part_state:
                            case UniswapV2Pool() | CamelotLiquidityPool(), UniswapV2PoolState():
                                
                                if part_state.reserves_token0 == 0 or part_state.reserves_token1 == 0:
                                    raise NoLiquidity(f"V2 pool {part_pool.address} has no liquidity")
                                price = part_state.reserves_token1 / part_state.reserves_token0                            
                                fee = part_pool.fee_token0 if pool.token_in == part_pool.token0 else part_pool.fee_token1
                                rate = (price if pool.token_in == part_pool.token0 else 1 / price) * (
                                    (fee.denominator - fee.numerator) / fee.denominator
                                )
                                rates.append(rate)
                                
                            case UniswapV3Pool(), UniswapV3PoolState():
                                if part_state.sqrt_price_x96 == 0:
                                    raise NoLiquidity(f"V3 pool {part_pool.address} has no liquidity")
                                price = part_state.sqrt_price_x96**2 / (2**192)
                                fee = Fraction(part_pool.fee, 1000000)
                                rate = (price if pool.token_in == part_pool.token0 else 1 / price) * (
                                    (fee.denominator - fee.numerator) / fee.denominator
                                )
                                rates.append(rate)

                                
                            # case CurveStableswapPool(),CurveStableswapPoolState():
                            #     price = 1.0 * (10**vector.token_out.decimals) / (10**vector.token_in.decimals)
                            #     fee = Fraction(part_pool.fee, part_pool.FEE_DENOMINATOR)
                            #     profit_factor *= price * ((fee.denominator - fee.numerator) / fee.denominator)
                                
                     # Teste die Liquidität mit einem kleinen Input
                     # TODO testen ob die klakulation im combned pool mit price*=.... richtig ist 
                     
                    profit_factor*= price

            # Falls der Profit-Faktor < 1.0 ist, ist der Trade nicht profitabel
            print("profit factor: ",profit_factor)
            if profit_factor < 1.0:
                raise ArbitrageError(
                    f"No profitable arbitrage at current prices. Profit factor: {profit_factor}"
                )
                
    def calculate(
        self,
        override_state: Sequence[
            Tuple[
                AbstractLiquidityPool,
                AbstractPoolState | UniswapSimulationResult,
            ]
        ]
        | None = None,
    ) -> CombinedLiquidityArbitrageCalculationResult:
        """
        Calculate the optimum arbitrage input and intermediate swap values for the current pool states.
        """
        self._pre_calculation_check(override_state)
        return self._calculate(override_state=override_state)
    
    def _calculate(
        self,
        override_state: Sequence[
            Tuple[AbstractLiquidityPool, AbstractPoolState | UniswapSimulationResult]
        ]
        | None = None,
        block_number: int | None = None,
    ) -> CombinedLiquidityArbitrageCalculationResult:
        """
        Diese Methode berechnet den besten Input und Output für den Arbitragepfad,
        einschließlich des letzten Swaps, der kombinierte Liquidität verwendet.
        """
        state_overrides = self._sort_overrides(override_state)

        # Begrenzung der zu tauschenden Menge (Input)
        bounds = (
            1.0,
            max(2.0, float(self.max_input)),
        )

        # Startwerte für die Optimierung (Brackets)
        bracket_amount = self.max_input
        bracket = (
            0.45 * bracket_amount,
            0.50 * bracket_amount,
            0.55 * bracket_amount,
        )

        def arb_profit(x: float) -> float:
            token_in_quantity = int(x)  # Runden der Eingabemenge
            token_out_quantity: int = 0

            # Iteriere über alle Pools und berechne den Output
            for i, (pool, swap_vector) in enumerate(zip(self.swap_pools, self._swap_vectors)):
                pool_override = state_overrides.get(pool.address)

                try:
                    match pool:
                        case UniswapV2Pool() | CamelotLiquidityPool():
                            token_out_quantity = pool.calculate_tokens_out_from_tokens_in(
                                token_in=swap_vector.token_in,
                                token_in_quantity=token_in_quantity if i == 0 else token_out_quantity,
                                override_state=pool_override,
                            )

                        case UniswapV3Pool():
                            token_out_quantity = pool.calculate_tokens_out_from_tokens_in(
                                token_in=swap_vector.token_in,
                                token_in_quantity=token_in_quantity if i == 0 else token_out_quantity,
                                override_state=pool_override,
                            )

                        case CurveStableswapPool():
                            token_out_quantity = int(
                                self.curve_discount_factor
                                * pool.calculate_tokens_out_from_tokens_in(
                                    token_in=swap_vector.token_in,
                                    token_in_quantity=(token_in_quantity if i == 0 else token_out_quantity),
                                    token_out=swap_vector.token_out,
                                    override_state=pool_override,
                                    block_identifier=block_number,
                                )
                            )
                            
                        case CombinedLiquidityPool():
                            # Verwende die test_combinations Methode, um den besten Output und die beste Verteilung zu finden
                            token_out_quantity = pool.calculate_tokens_out_from_tokens_in(
                                total_amount_in=token_out_quantity  # Input für den CombinedLiquidityPool
                            )

                except (EVMRevertError, LiquidityPoolError):
                    # Bei einem Fehler während der Optimierung wird der Output auf 0 gesetzt
                    token_out_quantity = 0
                    break

            # Die Optimierungsfunktion gibt den negierten Profit zurück (für minimize_scalar)
            return -float(token_out_quantity - token_in_quantity)

        # Führe die Optimierung durch
        opt = minimize_scalar(
            fun=arb_profit,
            method="bounded",
            bounds=bounds,
            bracket=bracket,
            options={"xatol": 1.0},
        )

        # Negiere das Ergebnis, um den maximalen Profit zu erhalten
        best_profit = -int(opt.fun)
        swap_amount = int(opt.x)

        try:
            best_amounts = self._build_amounts_out(
                token_in=self.input_token,
                token_in_quantity=swap_amount,
                pool_state_overrides=state_overrides,
                block_number=block_number,
            )
        except ArbitrageError as e:
            # Fange eventuelle Fehler auf, wenn die berechneten Swaps nicht möglich sind
            raise ArbitrageError(f"No possible arbitrage: {e}") from None
        except Exception as e:
            raise ArbitrageError(f"No possible arbitrage: {e}") from e # TODO------------------------------------------------------------------
        except self.last_pool.distribution is None:
            raise ArbitrageError(f"No possible arbitrage: No best combination found {e}")
        return CombinedLiquidityArbitrageCalculationResult(
            id=self.id,
            input_token=self.input_token,
            profit_token=self.input_token,
            input_amount=swap_amount,
            profit_amount=best_profit,
            swap_amounts=best_amounts,
            distribution=self.last_pool.distribution
            )
        
    async def calculate_with_pool(
        self,
        executor: ProcessPoolExecutor | ThreadPoolExecutor,
        override_state: Sequence[
            Tuple[
                AbstractLiquidityPool,
                AbstractPoolState | UniswapSimulationResult,
            ]
        ]
        | None = None,
        ) -> Awaitable[Any]:
            """
            Wrap the arbitrage calculation into an asyncio future using the
            specified executor.

            Arguments
            ---------
            executor : Executor
                An executor (from `concurrent.futures`) to process the calculation
                work. Both `ThreadPoolExecutor` and `ProcessPoolExecutor` are
                supported, but `ProcessPoolExecutor` is recommended.
            override_state : StateOverrideTypes, optional
                An sequence of tuples, representing an ordered pair of helper
                objects for Uniswap V2 / V3 pools and their overridden states.

            Returns
            -------
            A future which returns a `ArbitrageCalculationResult` (or exception)
            when awaited.

            Notes
            -----
            This is an async function that must be called with the `await` keyword.
            """

            self._pre_calculation_check(override_state)

            if any(
                [pool.sparse_liquidity_map for pool in self.swap_pools if isinstance(pool, UniswapV3Pool)]
            ):
                raise ValueError(
                    f"Cannot calculate {self} with executor. One or more V3 pools has a sparse bitmap."
                )

            curve_pool = self.swap_pools[1]
            curve_swap_vector = self._swap_vectors[1]

            if TYPE_CHECKING:
                assert isinstance(curve_pool, CurveStableswapPool)
                assert isinstance(curve_swap_vector, CurveStableSwapPoolVector)

            block_number = get_web3().eth.get_block_number()

            # Some Curve pools utilize on-chain lookups in their calc, so do a simple pre-calc to
            # cache those values for a given block since the pool will be disconnected once sent
            # into the process pool, e.g. it will have no web3 object for communication with the chain
            curve_pool.calculate_tokens_out_from_tokens_in(
                token_in=curve_swap_vector.token_in,
                token_in_quantity=1,
                token_out=curve_swap_vector.token_out,
                block_identifier=block_number,
            )

            return asyncio.get_running_loop().run_in_executor(
                executor,
                self._calculate,
                override_state,
                block_number,
            )
            
    def notify(self, publisher: Publisher, message: Any) -> None:
        match publisher:
            case UniswapV2Pool() | UniswapV3Pool() | CurveStableswapPool() | CamelotLiquidityPool():
                self._update_pool_states((publisher,))
            case _:  # pragma: no cover
                logger.info(
                    f"{self} received message {message} from unsupported subscriber {publisher}"
                )

    def _update_pool_states(self, pools: Iterable[AbstractLiquidityPool]) -> None:
        """
        Update `self.pool_states` with state values from the `pools` iterable
        """
        self.pool_states.update({pool.address: pool.state for pool in pools})