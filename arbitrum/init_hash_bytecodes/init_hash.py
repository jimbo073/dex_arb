import web3
from web3 import Web3
from init_hash_bytecodes.factory_bytecode import FACTORY_BYTECODE
# Verbindung zum Arbitrum-Netzwerk
arb_web3 = Web3(Web3.HTTPProvider("https://rpc.ankr.com/eth/2a331a8be02227942c118c77d35a404581716f7364de6082220f72865804e042"))

# Uniswap V2 Pair Creation Code
uniswap_v2_pair_creation_code = FACTORY_BYTECODE

# Berechne den Init Code Pair Hash
init_code_hash = Web3.keccak(hexstr=uniswap_v2_pair_creation_code).hex()
print(f"Init Code Pair Hash: {init_code_hash}")
