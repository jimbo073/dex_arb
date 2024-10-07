from fractions import Fraction
from dex_arb.arbitrum._1inch_aggregator_classes import OneInchAggregator
import web3
import degenbot
import asyncio

_web3 = web3.Web3(web3.HTTPProvider("https://floral-crimson-patina.arbitrum-mainnet.quiknode.pro/347691b7280c64c57237f77f7c8988972dde604d/"))
degenbot.set_web3(_web3)

async def main():
    # Aggregator-Objekt erstellen
    aggregator_v6 = OneInchAggregator(42161, "BMYEt9qskUTWzCtyPrGKOUMEmgkX8iOK", True,[{"src_token":_web3.to_checksum_address("0x82aF49447D8a07e3bd95BD0d56f35241523fBab1"),"dst_token":_web3.to_checksum_address("0xFa7F8980b0f1E64A2062791cc3b0871572f1F7f0")}])
    #await aggregator_v6.async_initialize()
    # pairs = {"src_token":_web3.to_checksum_address("0x82aF49447D8a07e3bd95BD0d56f35241523fBab1"),"dst_token":_web3.to_checksum_address("0xFa7F8980b0f1E64A2062791cc3b0871572f1F7f0")}
    prices_dict = await aggregator_v6.async_initialize()
    raw_tx = aggregator_v6.get_api_swap(src_token="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",dst_token="0xFa7F8980b0f1E64A2062791cc3b0871572f1F7f0",amt=Fraction(10),wallet_addr="0xA791336Eea54cA199e96123e5e0774a908268A65")
    print("api_price: ", raw_tx)
    print("prices_dict: ", prices_dict)
# Asynchronen Hauptprozess starten
if __name__ == "__main__":
    asyncio.run(main())
