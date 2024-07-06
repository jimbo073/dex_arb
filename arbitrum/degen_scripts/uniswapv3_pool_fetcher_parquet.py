import sys

import pyarrow
import pyarrow.parquet
import ujson
from eth_utils import to_checksum_address
from hexbytes import HexBytes

POOL_CREATED_PATH = "./uniswap_v3_poolcreated_events/"
V3_POOL_FILE = "ethereum_lps_uniswapv3.json"
FACTORY_ADDRESS = "0x1f98431c8ad98523631ae4a59f267346ea31f984"

try:
    with open(V3_POOL_FILE, "r") as file:
        lp_data = ujson.load(file)
        lp_metadata = lp_data.pop(-1)
        last_pool_block = lp_metadata["block_number"]
        last_pool_count = lp_metadata["number_of_pools"]
        print(f"Found {last_pool_count} pools up to block {last_pool_block}")
except FileNotFoundError:
    lp_data = []
    last_pool_block = 0
    last_pool_count = 0

poolcreated_events = pyarrow.parquet.read_table(
    source=POOL_CREATED_PATH,
    filters=[
        ("block_number", ">", last_pool_block),
    ],
)

if not poolcreated_events:
    print("No new results")
    sys.exit()

# print(f"Read table with columns: {poolcreated_events.column_names}")
_factory_address = HexBytes(FACTORY_ADDRESS)
for row in range(len(poolcreated_events)):
    event = {k: v[0] for k, v in poolcreated_events.take([row]).to_pydict().items()}

    if event["address"] != _factory_address:
        continue

    lp_data.append(
        {
            "pool_address": to_checksum_address(event["event__pool"]),
            "fee": event["event__fee"],
            "token0": to_checksum_address(event["event__token0"]),
            "token1": to_checksum_address(event["event__token1"]),
            "type": "UniswapV3",
        }
    )

lp_data.append({"block_number": event["block_number"], "number_of_pools": len(lp_data)})

with open(V3_POOL_FILE, "w") as file:
    ujson.dump(lp_data, file, indent=2)
    print(f"Stored {lp_data[-1]['number_of_pools']} pools")
