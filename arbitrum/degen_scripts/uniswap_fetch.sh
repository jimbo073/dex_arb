#!/usr/bin/env bash

# This script extracts logs for the following events:
# - Uniswap V2 pool creation
# - Uniswap V3 pool creation 
# - Sushiswap V2 pool creation
# - Sushiswap V3 pool creation
# - Mint & Burn liquidity events for Uniswap V3-based pools

# This script is
# - idempotent: run the script as many times as you want, the data will be fine
# - interruptable: you can interrupt the script whenever you want, the data will be fine
# - incremental: only missing data is collected. re-runing the script does not re-collect data

RPC_URL="http://localhost:8543"
REORG_BUFFER=32
DATA_DIR=C:\Users\jimbo\projects\arbitrum\cryo_data

# UniswapV3 liquidity events
cryo logs \
	--rpc $RPC_URL \
	--blocks 12_369_621: \
	--reorg-buffer $REORG_BUFFER \
	--u256-types binary \
	--event 0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853ae16239d0bde \
	--event-signature "Mint(address sender, address indexed owner, int24 indexed tickLower, int24 indexed tickUpper, uint128 amount, uint256 amount0, uint256 amount1)" \
	--output-dir $DATA_DIR/uniswap_v3_mint_events
cryo logs \
	--rpc $RPC_URL \
	--blocks 12_369_621: \
	--reorg-buffer $REORG_BUFFER \
	--u256-types binary \
	--event 0x0c396cd989a39f4459b5fa1aed6a9a8dcdbc45908acfd67e028cd568da98982c \
	--event-signature "Burn(address indexed owner, int24 indexed tickLower, int24 indexed tickUpper, uint128 amount, uint256 amount0, uint256 amount1)" \
	--output-dir $DATA_DIR/uniswap_v3_burn_events

# UniswapV2 new pools
cryo logs \
	--rpc $RPC_URL \
	--blocks 10_000_835: \
	--reorg-buffer $REORG_BUFFER \
	--u256-types binary \
	--contract 0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f \
	--event 0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9 \
	--event-signature "PairCreated(address indexed token0, address indexed token1, address pair, uint)" \
	--output-dir $DATA_DIR/uniswap_v2_paircreated_events

# SushiswapV2 new pools
cryo logs \
	--rpc $RPC_URL \
	--blocks 10_794_229: \
	--reorg-buffer $REORG_BUFFER \
	--u256-types binary \
	--contract 0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac \
	--event 0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9 \
	--event-signature "PairCreated(address indexed token0, address indexed token1, address pair, uint)" \
	--output-dir $DATA_DIR/sushiswap_v2_paircreated_events

# UniswapV3 new pools
cryo logs \
	--rpc $RPC_URL \
	--blocks 12_369_621: \
	--reorg-buffer $REORG_BUFFER \
	--u256-types binary \
	--contract 0x1f98431c8ad98523631ae4a59f267346ea31f984 \
	--event 0x783cca1c0412dd0d695e784568c96da2e9c22ff989357a2e8b1d9b2b4e6b7118 \
	--event-signature "PoolCreated(address indexed token0, address indexed token1, uint24 indexed fee, int24 tickSpacing, address pool)" \
	--output-dir $DATA_DIR/uniswap_v3_poolcreated_events

# SushiswapV3 new pools
cryo logs \
	--rpc $RPC_URL \
	--blocks 16_955_547: \
	--reorg-buffer $REORG_BUFFER \
	--u256-types binary \
	--contract 0xbACEB8eC6b9355Dfc0269C18bac9d6E2Bdc29C4F \
	--event 0x783cca1c0412dd0d695e784568c96da2e9c22ff989357a2e8b1d9b2b4e6b7118 \
	--event-signature "PoolCreated(address indexed token0, address indexed token1, uint24 indexed fee, int24 tickSpacing, address pool)" \
	--output-dir $DATA_DIR/sushiswap_v3_poolcreated_events

python3 $DATA_DIR/sushiswapv3_pool_fetcher_parquet.py
python3 $DATA_DIR/uniswapv3_pool_fetcher_parquet.py
python3 $DATA_DIR/v3_liquidity_events_processor_parquet.py
