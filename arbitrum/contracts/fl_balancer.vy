# @version 0.3.10

from vyper.interfaces import ERC20 as IERC20

OWNER: immutable(address)
WETH: constant(address) = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2
USDC: constant(address) = 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48
WBTC: constant(address) = 0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599
aggregator: constant(address) = 0x1111111254EEB25477B68fb85Ed929f73A960582
BALANCER_VAULT: constant(address) = 0xBA12222222228d8Ba445958a75a0704d566BF2C8
MAX_PAYLOAD_BYTES: constant(uint256) = 4800

interface IWETH:    
    def deposit(): payable

interface IBalancerVault:
    def flashLoan(
        recipient:address,
        tokens_to_borrow:IERC20,
        amount:uint256,
        data:Bytes[3*MAX_PAYLOAD_BYTES +320] 
    ): payable

interface IFlashLoanRecipient:
    def receiveFlashLoan(
        tokens:DynArray[address, 1],
        amounts:DynArray[uint256, 1], 
        feeAmounts:DynArray[uint256, 1], 
        data:Bytes[3*MAX_PAYLOAD_BYTES +320]
    ):payable

implements: IFlashLoanRecipient 

@external
@payable
def __init__():
    IERC20(WETH).approve(aggregator,max_value(uint256))
    IERC20(USDC).approve(aggregator,max_value(uint256))
    IERC20(WBTC).approve(aggregator,max_value(uint256))

    OWNER = msg.sender

@external
@payable
def _flashLoan(
    token_to_borrow:DynArray[address,1] ,
    amount:DynArray[uint256,1],
    calldata1:Bytes[MAX_PAYLOAD_BYTES],
    calldata2:Bytes[MAX_PAYLOAD_BYTES],
    calldata3:Bytes[MAX_PAYLOAD_BYTES],
    amt_a:uint256,
    amt_b:uint256
    ):
    data: Bytes[3*MAX_PAYLOAD_BYTES +320] =  _abi_encode(
        calldata1,
        calldata2,
        calldata3,
        amt_a,
        amt_b
        )
    raw_call(
        BALANCER_VAULT, 
        _abi_encode(
            self,
            token_to_borrow,
            amount,
            data,
            method_id = method_id("flashLoan(address,address[],uint256[],bytes)")
            ),\
        max_outsize=0
        )

event one_inch_call:
    calldata:Bytes[MAX_PAYLOAD_BYTES]
    response:Bytes[32] 

@external
def receiveFlashLoan(
    tokens: DynArray[address,1], 
    amounts: DynArray[uint256,1], 
    feeAmounts: DynArray[uint256,1], 
    data:Bytes[3*MAX_PAYLOAD_BYTES +320] 
    ):
    assert msg.sender == BALANCER_VAULT, "!VAULT"

    coin_balance_before: uint256 = empty(uint256)
    coin_balance_before = IERC20(tokens[0]).balanceOf(self)

    calldata1:Bytes[MAX_PAYLOAD_BYTES] = b""
    calldata2:Bytes[MAX_PAYLOAD_BYTES] = b""
    calldata3:Bytes[MAX_PAYLOAD_BYTES] = b""
    amt_a:uint256 = empty(uint256)
    amt_b:uint256 = empty(uint256)
    
    calldata1,calldata2,calldata3,amt_a,amt_b = _abi_decode(
        data,
        (Bytes[MAX_PAYLOAD_BYTES],Bytes[MAX_PAYLOAD_BYTES],Bytes[MAX_PAYLOAD_BYTES],uint256,uint256)
        )

    # logic here for calls to 1inch and balance checks
    #1st call 
    success1: bool = False
    response1: Bytes[32] = b""
    success1,response1 = raw_call(
        aggregator, # 1inch aggregator
        calldata1,\
        max_outsize=32,
        revert_on_failure=False
        )
    if success1 == False:
        log one_inch_call(calldata1,response1)
    assert success1 ,"1st call failed"

    balance_after_1_swap: uint256 = empty(uint256)
    balance_after_1_swap = IERC20(WBTC).balanceOf(self)
    assert balance_after_1_swap >= amt_a , "1st call after balance too low"

    #2nd call 
    success2: bool = False
    response2: Bytes[32] = b""
    success2,response2 = raw_call(
        aggregator, # 1inch aggregator
        calldata2,\
        max_outsize=32,
        revert_on_failure=False
        )
    if success2 == False:
        log one_inch_call(calldata2,response2)
    assert success2 ,"2nd call failed"

    balance_after_2_swap: uint256 = empty(uint256)
    balance_after_2_swap = IERC20(WETH).balanceOf(self)
    assert balance_after_2_swap >= amt_b , "2nd call after balance too low"

    #3rd call 
    success3: bool = False
    response3: Bytes[32] = b""
    success3,response3 = raw_call(
        aggregator, # 1inch aggregator
        calldata3,\
        max_outsize=32,
        revert_on_failure=False
        )
    if success3 != True:
        log one_inch_call(calldata3,response3)
    assert success3 ,"3rd call failed"
        
    # get coin balance to check after swaps are executed
    coin_balance_after: uint256 = empty(uint256)
    coin_balance_after = IERC20(tokens[0]).balanceOf(self)

    assert coin_balance_after > coin_balance_before , "!BIGGER"

    # loan repayment
    amt0:uint256 = amounts[0] + feeAmounts[0]
    IERC20(tokens[0]).transfer(msg.sender,amt0)

    # sending the profit to wallet
    profit: uint256 = empty(uint256)
    profit = coin_balance_after - amounts[0] 
    IERC20(tokens[0]).transfer(OWNER,profit)


@external
@payable
def __default__():
    # accept basic Ether transfers to the contract with no calldata
    if len(msg.data) == 0:
        return
    # revert on all other calls
    else:
        raise