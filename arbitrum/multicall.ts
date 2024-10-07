import {
    AbiItem,
    GasLimitService,
    MultiCallRequestWithGas,
    MultiCallService,
    MultiCallWithGasParams,
    Web3ProviderConnector,
  } from '@1inch/multicall';
  import { Web3 } from 'web3';
  import * as readline from 'readline';
  
  const provider = new Web3ProviderConnector(
    new Web3(
      'https://floral-crimson-patina.arbitrum-mainnet.quiknode.pro/347691b7280c64c57237f77f7c8988972dde604d/'
    )
  );
  const contractAddress = '0x11DEE30E710B8d4a8630392781Cc3c0046365d4c';
  const gasLimitService = new GasLimitService(provider, contractAddress);
  const multiCallService = new MultiCallService(provider, contractAddress);
  
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
  
  const processRequest = async (pairs: string) => {
    const oracle_contract_abi: AbiItem[] = require('./oracle_contract_abi.json');
    try {
      const tokenPairs = JSON.parse(pairs);
  
      if (!Array.isArray(tokenPairs)) {
        throw new Error('Die Eingabedaten sind kein Array. Erwartet wird ein Array von Token-Paaren.');
      }
  
      // Erstellen der Anfragen
      const requests: (MultiCallRequestWithGas & {
        dst_token: string;
        price: null | number;
        index: number;
      })[] = tokenPairs.map((pair, index) => ({
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
  
      const params: MultiCallWithGasParams = {
        maxChunkSize: 20,
        retriesLimit: 10,
        blockNumber: 'latest',
        gasBuffer: 50_000,
      };
  
      // Berechnen Sie das Gaslimit und führen Sie den Multicall aus
      const gasLimit = await gasLimitService.calculateGasLimit();
      const response = await multiCallService.callByGasLimit(requests, gasLimit, params);
  
      // Neue Liste zur Speicherung von Preis, src_token und dst_token mit explizitem Typ
      const resultList: { price: number | null; from_token: string; to_token: string }[] = [];
  
      response.forEach((hexValue, i) => {
        try {
          const decodedValue = parseInt(hexValue, 16);
          requests[i].price = decodedValue; // Preis dem entsprechenden Anfrageobjekt hinzufügen
          resultList.push({
            price: decodedValue,
            from_token: tokenPairs[i].src_token,
            to_token: tokenPairs[i].dst_token,
          });
        } catch (error) {
          console.error(`Fehler beim Dekodieren des Wertes ${hexValue}:`, error);
          resultList.push({
            price: null, // Wenn Dekodierung fehlschlägt, bleibt der Preis null
            from_token: tokenPairs[i].src_token,
            to_token: tokenPairs[i].dst_token,
          });
        }
      });
  
      // Loggen Sie die neue Liste
      process.stdout.write(JSON.stringify(JSON.parse(resultList)));
    } catch (error) {
      console.error(
        JSON.stringify({
          error: error instanceof Error ? `Verarbeitungsfehler: ${error.message}` : 'Ein unbekannter Fehler ist aufgetreten',
        })
      );
      process.stdout.write(
        JSON.stringify({
          error: error instanceof Error ? `Verarbeitungsfehler: ${error.message}` : 'Ein unbekannter Fehler ist aufgetreten',
        }) + '\n'
      );
    }
  };
  
  // Setze die readline-Schnittstelle, um kontinuierlich auf Eingaben zu warten
  rl.on('line', (input) => {
    processRequest(input).catch((error) => {
      console.error(`Fehler bei der Verarbeitung: ${error}`);
    });
  });
  