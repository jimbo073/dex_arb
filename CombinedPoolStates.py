from typing import Sequence, Union, Dict,List
from eth_typing import ChecksumAddress
import dataclasses
import degenbot
from degenbot.uniswap.types import UniswapV3BitmapAtWord, UniswapV3LiquidityAtTick, UniswapV2PoolState, UniswapV3PoolState
from degenbot.curve.types import CurveStableswapPoolState


@dataclasses.dataclass(slots=True)
class CombinedLiquidityPoolStates:
    """
    Diese Klasse speichert die Zustände für alle Pools im CombinedLiquidityPool.
    Verwende ein Dictionary, um die Zustände effizient nach Pool-Adresse zu speichern.
    """
    pool_states: Dict[ChecksumAddress, Sequence[UniswapV2PoolState | UniswapV3PoolState | CurveStableswapPoolState]] = dataclasses.field(default_factory=dict)
    
    def add_uniswap_v2_pool_state(
        self, 
        pool_address: ChecksumAddress, 
        state: UniswapV2PoolState
    ) -> None:
        """
        Fügt den Zustand eines Uniswap V2 Pools zur Liste der CombinedLiquidityPoolStates hinzu.
        """
        self.pool_states[pool_address] = UniswapV2PoolState(
            pool_address=pool_address,
            reserves_token0=state.reserves_token0,
            reserves_token1=state.reserves_token1,
        )
    
    def add_uniswap_v3_pool_state(
        self, 
        pool_address: ChecksumAddress, 
        state: UniswapV3PoolState
    ) -> None:
        """
        Fügt den Zustand eines Uniswap V3 Pools zur Liste der CombinedLiquidityPoolStates hinzu.
        """
        self.pool_states[pool_address] = UniswapV3PoolState(
            pool_address=pool_address,
            liquidity=state.liquidity,
            sqrt_price_x96=state.sqrt_price_x96,
            tick=state.tick,
            tick_bitmap=state.tick_bitmap,
            tick_data=state.tick_data
        )
    def add_curve_pool_state(
        self, 
        pool_address: ChecksumAddress, 
        state: CurveStableswapPoolState
    ) -> None:
        """
        Fügt den Zustand eines Curve Pools zur Liste der CombinedLiquidityPoolStates hinzu.
        """
        self.pool_states[pool_address] = CurveStableswapPoolState(
            pool_address=pool_address,
            balances=state.balances
            
        )

    def get_pool_state(self, pool_address: ChecksumAddress) -> Union[UniswapV2PoolState, UniswapV3PoolState, CurveStableswapPoolState, None]:
        """
        Gibt den Zustand eines Pools anhand der Adresse zurück.
        """
        return self.pool_states.get(pool_address)

    def get_all_pool_addresses(self) -> List[ChecksumAddress]:
        """
        Gibt eine Liste aller Pool-Adressen im CombinedLiquidityPool zurück.
        """
        return list(self.pool_states.keys())