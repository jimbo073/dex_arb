# @version ^0.3

from vyper.interfaces import ERC20 as IERC20

interface IWETH:    
    def deposit(): payable

interface IUniswapV3Pool:
    def factory() -> address: view
    def fee() -> uint24: view
    def tickSpacing() -> int24: view
    def token0() -> address: view
    def token1() -> address: view
    def maxLiquidityPerTick() -> uint128: view

OWNER_ADDR: immutable(address)
WETH_ADDR: constant(address) = 0x82aF49447D8a07e3bd95BD0d56f35241523fBab1
V3_FACTORY: constant(address) = 0x1F98431c8aD98523631AE4a59f267346ea31F984

MAX_PAYLOADS: constant(uint256) = 16
MAX_PAYLOAD_BYTES: constant(uint256) = 1024

struct payload:
    target: address
    calldata: Bytes[MAX_PAYLOAD_BYTES]
    value: uint256


@external
@payable
def __init__():

    OWNER_ADDR = msg.sender    
    
    # wrap initial Ether to WETH
    if msg.value > 0:
        IWETH(WETH_ADDR).deposit(value=msg.value)


@external
@payable
def execute_payloads(
    payloads: DynArray[payload, MAX_PAYLOADS],
):

    assert msg.sender == OWNER_ADDR, "!OWNER"
   
    for _payload in payloads:
        raw_call(
            _payload.target,
            _payload.calldata,
            value=_payload.value,
        )


@internal
@pure
def verifyCallback(
    tokenA: address, 
    tokenB: address, 
    fee: uint24
) -> address:   
            
    token0: address = tokenA
    token1: address = tokenB

    if convert(tokenA,uint160) > convert(tokenB,uint160):        
        token0 = tokenB
        token1 = tokenA
        
    return convert(
        slice(
            convert(
                convert(
                    keccak256(
                        concat(
                            b'\xFF',
                            convert(V3_FACTORY,bytes20),
                            keccak256(
                                _abi_encode(
                                    token0,
                                    token1,
                                    fee
                                )
                            ),
                            0xe34f199b19b2b4f47f68442619d555527d244f78a3297ea89325f843f87b8b54,
                        )
                    ),
                    uint256
                ),
                bytes32
            ),
            12,
            20,
        ),
        address
    )


@external
def uniswapV3SwapCallback(
    amount0_delta: int256, 
    amount1_delta: int256, 
    data: Bytes[32]
):
    assert amount0_delta > 0 or amount1_delta > 0, "REJECTED 0 LIQUIDITY SWAP"

    # get the token0/token1 addresses and fee reported by msg.sender   
    token0: address = IUniswapV3Pool(msg.sender).token0()
    token1: address = IUniswapV3Pool(msg.sender).token1()
    fee: uint24 = IUniswapV3Pool(msg.sender).fee()
    
    assert msg.sender == self.verifyCallback(token0,token1,fee), "!V3LP"

    # transfer token back to pool
    if amount0_delta > 0:
        IERC20(token0).transfer(msg.sender,convert(amount0_delta, uint256))        
    elif amount1_delta > 0:
        IERC20(token1).transfer(msg.sender,convert(amount1_delta, uint256))        


@external
@payable
def __default__():
    # accept basic Ether transfers to the contract with no calldata
    if len(msg.data) == 0:
        return
    
    # revert on all other calls
    else:
        raise