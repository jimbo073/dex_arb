"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
const multicall_1 = require("@1inch/multicall");
const web3_1 = require("web3");
const readline = __importStar(require("readline"));
const provider = new multicall_1.Web3ProviderConnector(new web3_1.Web3('https://floral-crimson-patina.arbitrum-mainnet.quiknode.pro/347691b7280c64c57237f77f7c8988972dde604d/'));
const contractAddress = '0x11DEE30E710B8d4a8630392781Cc3c0046365d4c';
const gasLimitService = new multicall_1.GasLimitService(provider, contractAddress);
const multiCallService = new multicall_1.MultiCallService(provider, contractAddress);
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false,
});
const balanceOfGasUsage = 30000;
process.on('unhandledRejection', (reason, promise) => {
    console.error(JSON.stringify({ error: `Unhandled Rejection: ${reason}` }));
    process.exit(1);
});
const processRequest = (pairs) => __awaiter(void 0, void 0, void 0, function* () {
    const oracle_contract_abi = require('./oracle_contract_abi.json');
    try {
        const tokenPairs = JSON.parse(pairs);
        if (!Array.isArray(tokenPairs)) {
            throw new Error('Die Eingabedaten sind kein Array. Erwartet wird ein Array von Token-Paaren.');
        }
        // Erstellen der Anfragen
        const requests = tokenPairs.map((pair, index) => ({
            to: '0x0AdDd25a91563696D8567Df78D5A01C9a991F9B8',
            data: provider.contractEncodeABI(oracle_contract_abi, null, 'getRate', [
                pair.src_token,
                pair.dst_token,
                true,
            ]),
            gas: balanceOfGasUsage,
            dst_token: pair.dst_token,
            price: null,
            index: index,
        }));
        const params = {
            maxChunkSize: 20,
            retriesLimit: 10,
            blockNumber: 'latest',
            gasBuffer: 50000,
        };
        // Berechnen Sie das Gaslimit und führen Sie den Multicall aus
        const gasLimit = yield gasLimitService.calculateGasLimit();
        const response = yield multiCallService.callByGasLimit(requests, gasLimit, params);
        // Neue Liste zur Speicherung von Preis, src_token und dst_token mit explizitem Typ
        const resultList = [];
        response.forEach((hexValue, i) => {
            try {
                const decodedValue = parseInt(hexValue, 16);
                requests[i].price = decodedValue; // Preis dem entsprechenden Anfrageobjekt hinzufügen
                resultList.push({
                    price: decodedValue,
                    from_token: tokenPairs[i].src_token,
                    to_token: tokenPairs[i].dst_token,
                });
            }
            catch (error) {
                console.error(`Fehler beim Dekodieren des Wertes ${hexValue}:`, error);
                resultList.push({
                    price: null, // Wenn Dekodierung fehlschlägt, bleibt der Preis null
                    from_token: tokenPairs[i].src_token,
                    to_token: tokenPairs[i].dst_token,
                });
            }
        });
        // Loggen Sie die neue Liste
        process.stdout.write(JSON.stringify(resultList) + '\n');
    }
    catch (error) {
        console.error(JSON.stringify({
            error: error instanceof Error ? `Verarbeitungsfehler: ${error.message}` : 'Ein unbekannter Fehler ist aufgetreten',
        }));
        process.stdout.write(JSON.stringify({
            error: error instanceof Error ? `Verarbeitungsfehler: ${error.message}` : 'Ein unbekannter Fehler ist aufgetreten',
        }) + '\n');
    }
});
// Setze die readline-Schnittstelle, um kontinuierlich auf Eingaben zu warten
rl.on('line', (input) => {
    processRequest(input).catch((error) => {
        console.error(`Fehler bei der Verarbeitung: ${error}`);
    });
});
