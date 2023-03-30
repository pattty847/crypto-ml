import ccxt.pro as ccxtpro
import asyncio

async def watch_exchange(exchange_name, symbol, aggregated_data, lock):
    exchange_class = getattr(ccxtpro, exchange_name)
    exchange = exchange_class({
        "enableRateLimit": True,
        "newUpdates": True
    })

    base_volume = 0.0
    usd_volume = 0.0
    cvd = 0.0

    while True:
        try:
            # Watch the trades for the specified symbol
            trades = await exchange.watchTrades(symbol)

            # Calculate volume and CVD for new trades
            for trade in trades:
                trade_volume_base = trade['amount']
                trade_volume_usd = trade['amount'] * trade['price']
                base_volume += trade_volume_base
                usd_volume += trade_volume_usd
                cvd += trade_volume_base if trade['side'] == 'buy' else -trade_volume_base

            # Update and print the aggregated data
            async with lock:
                aggregated_data[exchange_name] = {
                    'base_volume': base_volume,
                    'usd_volume': usd_volume,
                    'cumulative_volume_delta': cvd
                }
                print_aggregated_data(aggregated_data)

        except KeyboardInterrupt:
            print(f"KeyboardInterrupt detected. Closing {exchange_name}...")
            await exchange.close()
            break
    print(f"{exchange_name} closed.")

def print_aggregated_data(aggregated_data):
    for exchange_name, data in aggregated_data.items():
        print(f"Exchange: {exchange_name}")
        print("Base Volume:", data['base_volume'])
        print("USD Volume:", data['usd_volume'])
        print("Cumulative Volume Delta (CVD):", data['cumulative_volume_delta'])
        print()

async def main():
    symbol = 'BTC/USDT'
    limit = None
    exchange_names = ['kucoin', 'coinbasepro', 'kraken']
    aggregated_data = {}
    lock = asyncio.Lock()

    # Create a list of coroutines for watching each exchange
    coroutines = [watch_exchange(exchange_name, symbol, limit, aggregated_data, lock) for exchange_name in exchange_names]

    # Run the coroutines concurrently
    await asyncio.gather(*coroutines)

# Run the main function
asyncio.run(main())
