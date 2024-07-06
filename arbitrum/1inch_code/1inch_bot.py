from datetime import datetime
import datetime
import time
import ape.contracts
import ape.managers
import simplejson as json
from oneinch_py import *
from decimal import *
from urllib3.exceptions import HTTPError
import web3 as W3
import logging
from ape import project, accounts,Contract
from eth_account.messages import encode_defunct
import ape.exceptions
import degenbot
import ape_accounts 


jimbo = accounts.load("jimbo")
jimbo.set_autosign(True, passphrase="supersayanblue2")
accounts.test_accounts[1].transfer(value=10*10**18,account=jimbo)
fl_contract = jimbo.deploy(project.fl_balancer)

message = encode_defunct(text="Sign:")
signature = jimbo.sign_message(message)

# fl_contract = Contract("0x121124d7Fc1E06028D3B1EF9c48f198A592C7D95")

################################ oneinch module settings ###############################
rpc = "https://rpc.ankr.com/eth"
public_key =  fl_contract.address
private_key = "5591507be3393cb70baaab5e3d8aa57d5e4feb60bce24de5f6b78004f3c2e812"
api_key = "BMYEt9qskUTWzCtyPrGKOUMEmgkX8iOK"

exchange = OneInchSwap(api_key, public_key, chain='ethereum')
helper = TransactionHelper(api_key, rpc, public_key, private_key, chain='ethereum')
oracle = OneInchOracle(rpc, chain='ethereum')
#########################################################################################
BOT_ROOT_PATH =r"/home/ubuntu/Desktop/anvil_fork_test"

try:
    tokens_list = exchange.tokens
    ARBITRUM_WETH_TOKEN_DATA = tokens_list["WETH"]
    ARBITRUM_USDC_TOKEN_DATA = tokens_list["USDC"]
    ARBITRUM_WBTC_TOKEN_DATA = tokens_list["WBTC"]
    coin_a = ARBITRUM_USDC_TOKEN_DATA
    coin_b = ARBITRUM_WBTC_TOKEN_DATA
    coin_c = ARBITRUM_WETH_TOKEN_DATA
    USDC = Contract(coin_a["address"])
    WBTC = Contract(coin_b["address"])
    WETH = Contract(coin_c["address"])
except Exception as e:
    print(e)

def main():
    init_logger()
    # iterating over start amounts to determine maximum profit
    start_amount = 1000
    thresholds = [5000, 10000, 50000, 100000]
    threshold_index = 0
    open_oppurtunity = False

    while True:

        if open_oppurtunity == True:
            if threshold_index < len(thresholds):
                start_amount = thresholds[threshold_index]
                threshold_index += 1
            else:
                start_amount *= 2
        elif open_oppurtunity == False:
            start_amount = 1000
            threshold_index = 0
            pass
        
        try:
            time.sleep(1)
            raw_tx_a_to_b = exchange.get_swap(coin_a["address"], coin_b["address"],(start_amount),0.1,decimal=coin_a["decimals"],disableEstimate ="true",includeTokensInfo="true",includeProtocols="true",compatibility="true",parts=100,mainRouteParts=50,complexityLevel=3)
            a_to_b_amount = int(raw_tx_a_to_b['toAmount'])/10**coin_b["decimals"]
        except Exception as e:
                logging.info(f"HTTPError: {e}")
                time.sleep(1)
                continue
        try:
            time.sleep(1)
            raw_tx_b_to_c = exchange.get_swap(coin_b["address"], coin_c["address"],(a_to_b_amount),0.1,decimal=coin_b["decimals"],disableEstimate ="true",includeTokensInfo="true",includeProtocols="true",compatibility="true",parts=100,mainRouteParts=50,complexityLevel=3)
            b_to_c_amount = int(raw_tx_b_to_c['toAmount'])/10**coin_c["decimals"]
        except Exception as e:
                logging.info(f"HTTPError: {e}")
                time.sleep(1)
                continue
        try:
            time.sleep(1)
            raw_tx_c_to_a = exchange.get_swap(coin_c["address"], coin_a["address"],(b_to_c_amount),0.1,decimal=coin_c["decimals"],disableEstimate ="true",includeTokensInfo="true",includeProtocols="true",compatibility="true",parts=100,mainRouteParts=50,complexityLevel=3)
            c_to_a_amount =int(raw_tx_c_to_a['toAmount'])/10**coin_a["decimals"]
        except Exception as e:
                logging.info(f"HTTPError: {e}")
                time.sleep(1)
                continue

        if c_to_a_amount > start_amount:
            open_oppurtunity = True

            calldata1 = raw_tx_a_to_b["tx"]["data"]
            calldata2 = raw_tx_b_to_c["tx"]["data"]
            calldata3 = raw_tx_c_to_a["tx"]["data"]

            # TODO do the trade !!!!!
            try:
                logging.info(f"{ int(start_amount*10**coin_a['decimals'])}")
                logging.info(f"{ int(a_to_b_amount*10**coin_b['decimals'])}")
                logging.info(f"{ int(b_to_c_amount*10**coin_c['decimals'])}")
                rcpt = fl_contract._flashLoan.call([coin_a['address']],[start_amount*10**coin_a["decimals"]],calldata1,calldata2,calldata3,int(a_to_b_amount*10**coin_b["decimals"]),int(b_to_c_amount*10**coin_c["decimals"]),sender=jimbo,show_trace=True)
            except ape.exceptions.ContractLogicError as e:






                
                logging.info(f"tx reverted with ({e}) ")
                time.sleep(1)
                continue
            print(rcpt)
            trace = rcpt.show_trace()
            logging.info(trace)
            logging.info(rcpt)

            logging.info(f"({start_amount} USDC) -> ({a_to_b_amount} WBTC)")
            logging.info(f"({a_to_b_amount} WBTC) -> ({b_to_c_amount} WETH)")
            logging.info(f"({b_to_c_amount} WETH) -> ({c_to_a_amount} USDC)")
            # estimated_profit = c_to_a_amount - start_amount 
            # true_profit = 
            continue
        else:
            open_oppurtunity = False
        
        print(f"""
                ({start_amount }                     USDC) -> ({a_to_b_amount}          WBTC)
                ({a_to_b_amount  }               WBTC) -> ({b_to_c_amount}   WETH)
                ({b_to_c_amount  }       WETH) -> ({c_to_a_amount}          USDC)""")














#################################### FUNCTIONS #########################################################
def get_balance(token):
    time.sleep(1)
    balance = helper.get_ERC20_balance(token["address"],decimal=token["decimals"])
    return balance

def get_and_print_coin_balances(tokens):
    for token in tokens:
        token_balance = get_balance(token)
        time.sleep(1)
        token_symbol = token["symbol"]
        token_name = token["name"]
        print(f"{datetime.datetime.now()}: {token_balance} {token_symbol} ({token_name})")

def init_logger():
    # Adds a StreamHandler to print in console and write in log file
    logging.getLogger().addHandler(logging.StreamHandler())
    # Logger configuration
    logFormatter = logging.Formatter("%(asctime)s  [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    # Logger formatter and set Log-Level to INFO
    fileHandler = logging.FileHandler("{0}/logs/xBOT_1_{1}.log".format(BOT_ROOT_PATH, f"{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}"))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    rootLogger.setLevel(logging.INFO)
    # Amount Logger
    rootLogger = logging.getLogger()

main()