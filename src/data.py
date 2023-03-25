import json
import os
import pandas_ta as ta
import ccxt.async_support as ccxt
import pandas as pd
import logging
import asyncio
from log import setup_logging
from typing import List

class DataCollector:
    setup_logging()
    def __init__(self, exchanges, symbols, timeframe):
        self.exchanges = exchanges
        self.symbols = symbols
        self.timeframe = timeframe
        self.logger = logging.getLogger(__name__)

    def set_exchanges(self, exchanges):
        self.logger.info(f"Changing exchanges from {self.exchanges} to {exchanges}.")
        self.exchanges = exchanges
    
    def set_symbols(self, symbols):
        self.logger.info(f"Changing symbols from {self.symbols} to {symbols}.")
        self.symbols = symbols
        
    def save_candles_to_file(self, exchange, symbol, timeframe, candles: pd.DataFrame):
        symbol = symbol.replace("/", "_")
        directory = f"data/exchange/{exchange}"
        os.makedirs(directory, exist_ok=True)
        self.logger.info(f"Saving {symbol}_{timeframe}...")
        candles.to_csv(f"{directory}/{symbol}_{timeframe}.csv", index=True)

    def load_candles_from_file(self, exchange, symbol, timeframe):
        filename = (
            f"data/exchange/{exchange}/{symbol.replace('/', '_')}_{timeframe}.csv"
        )
        if os.path.exists(filename):
            self.logger.info(f"Loaded: {exchange.upper()}:{symbol.replace('/', '_')}_{timeframe} from file.")
            df = pd.read_csv(filename)
            df['dates'] = pd.to_datetime(df['dates'])
            return df
        else:
            # TODO: Make columns use self.indicators to apply the users specified TA features or use this as default
            columns = ["dates", "opens", "highs", "lows", "closes", "volumes", "sma_5", "sma_20", "ema_12", "ema_26", "macd", "rsi"]
            return pd.DataFrame(columns=columns)
    
    async def fetch_candles_for_symbol(self, exchange, symbol, timeframe, since, limit, dataframe, max_retries):
        """
        This asynchronous function fetches historical OHLCV (Open, High, Low, Close, Volume) candlestick data from 
        multiple cryptocurrency exchanges for specified symbols and timeframes. The data can be returned as a pandas 
        DataFrame or a dictionary. The function also supports retries in case of errors during data fetching. Technical
        indicators like RSI, MACD, MOM, etc are also calculaated and returned.

        Arguments:
            exchanges (List[str]): List of exchange names as strings.
            symbols (List[str]): List of asset symbols as strings (e.g. 'BTC/USD').
            timeframe (str): Timeframe of the candles (e.g. '1m', '1h', '1d').
            since (str): ISO8601 formatted string representing the starting date for fetching candles.
            limit (int): Maximum number of candles to fetch in one request.
            dataframe (bool): Whether to return the data as a pandas DataFrame (True) or as a dictionary (False).
            max_retries (int, optional): Maximum number of retries in case of errors during data fetching. Defaults to 3.

        Returns:
            dict or pd.DataFrame: A dictionary or pandas DataFrame containing the historical OHLCV data, grouped by exchange and symbol.

        Example usage:
            data = await fetch_candles(
                exchanges=['binance', 'coinbase'],
                symbols=['BTC/USD', 'ETH/USD'],
                timeframe='1h',
                since='2021-01-01T00:00:00Z',
                limit=100,
                dataframe=True
            )
        """
        api = getattr(ccxt, exchange)()
        if not api.has['fetchOHLCV']:
            self.logger.info(f"{exchange.upper()} does not have fetch OHLCV.")
            return None

        # Load cached candle history which includes TA indicators
        candles = self.load_candles_from_file(exchange, symbol, timeframe)
        new_candles = []

        timeframe_duration_in_seconds = api.parse_timeframe(timeframe)
        timedelta = limit * timeframe_duration_in_seconds * 1000
        now = api.milliseconds()
        fetch_since = (
            api.parse8601(since)
            if candles.empty
            else int(candles['dates'].iloc[-1].timestamp() * 1000)
        )

        # Fetch candles
        while True:
            new_candle_batch = None
            for num_retries in range(max_retries):
                try:
                    new_candle_batch = await api.fetch_ohlcv(
                        symbol, timeframe, since=fetch_since, limit=limit
                    )
                except ccxt.ExchangeError as e:
                    print(e)
                    await asyncio.sleep(1)
                if new_candle_batch is not None:
                    break
            if new_candle_batch is None:
                await api.close()
                return None

            new_candles += new_candle_batch

            if len(new_candle_batch):
                last_time = new_candle_batch[-1][0] + timeframe_duration_in_seconds * 1000
                self.logger.info(len(new_candle_batch), "candles from", api.iso8601(new_candle_batch[0][0]), "to", api.iso8601(new_candle_batch[-1][0]))
            else:
                last_time = fetch_since + timedelta
                self.logger.info("no candles")

            if last_time >= now:
                break

            fetch_since = last_time

        # Combine old and new candles
        if not candles.empty:
            candles = candles.iloc[:-1]

        # This dataframe's dates will contain timestamps in ms from the exchange
        new_candles_df = pd.DataFrame(new_candles, columns=["dates", "opens", "highs", "lows", "closes", "volumes"])
        new_candles_df["dates"] = pd.to_datetime(new_candles_df['dates'], unit='ms')
        candles = pd.concat([candles, new_candles_df]).reset_index(drop=True)

        # Calculate technical indicators
        updated_candles = self.calculate_ta(candles)

        # Clean the data
        # updated_candles = self.clean(updated_candles)

        # Save candles with technical indicators
        self.save_candles_to_file(exchange, symbol, timeframe, updated_candles)

        await api.close()
        
        return (exchange, symbol, updated_candles)
    
    def clean(self, candles):
        # TODO: This function can clean missing candle values by pulling data from another exchange. 
        date_range = pd.date_range(start=candles.index.min(), end=candles.index.max())

    def timeframe_to_freq(self, timeframe):
        pass
    
    def calculate_ta(self, data):
        self.logger.info(f"Adding technical analysis indicators.")
        data.set_index('dates', inplace=True)

        # create technical indicators
        data['sma_5'] = data['closes'].rolling(window=5).mean().round(3)
        data['sma_20'] = data['closes'].rolling(window=20).mean().round(3)
        data['ema_12'] = data['closes'].ewm(span=12, adjust=False).mean().round(3)
        data['ema_26'] = data['closes'].ewm(span=26, adjust=False).mean().round(3)
        data['macd'] = (data['ema_12'] - data['ema_26']).round(3)
        data['rsi'] = ta.rsi(data['closes'], timeperiod=14).round(3)

        return data

    async def fetch_candles(self, exchanges: List[str], symbols: List[str], timeframe: str, since: str, limit: int, dataframe: bool, max_retries=3):
        tasks = [
            self.fetch_candles_for_symbol(exchange, symbol, timeframe, since, limit, dataframe, max_retries)
            for exchange in exchanges
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks)
        candles = {}
        for exchange, symbol, result in results:
            if exchange not in candles:
                candles[exchange] = {}
            candles[exchange][symbol] = result

        return candles

exchanges = ['coinbasepro']
symbols = ['BTC/USD']
timeframe = '1d'
collector = DataCollector(exchanges, symbols, timeframe)

loop = asyncio.get_event_loop()
data = loop.run_until_complete(collector.fetch_candles(exchanges, symbols, timeframe, "2017-01-01 00:00:00", 1000, True))
print(data)

# btc = pd.read_csv("data/exchange/coinbasepro/BTC_USD_1d.csv")
# btc.columns = ["dates","opens","highs","lows","closes","volumes"]

