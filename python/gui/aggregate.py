import logging
import threading
import ccxt.pro as ccxtpro
import asyncio
import functools

class CryptoData:
    def __init__(self):
        self.symbol = 'BTC/USDT'
        self.exchange_names = ['kucoin', 'coinbasepro', 'kraken', 'bybit', 'bittrex', 'gateio']
        self.aggregated_data = {"aggregated_volume": 0.0}
        self.lock = asyncio.Lock()
        self.window = None
        self.loop = asyncio.new_event_loop()
        self.active_exchanges = {}
        self.exchange_changes_queue = asyncio.Queue()
        self.thread = threading.Thread(target=self.start_loop)
        self.logger = logging.getLogger(__name__)
        self.started = False

    async def watch_exchange(self, exchange_name):
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
                trades = await exchange.watchTrades(self.symbol)

                # Calculate volume and CVD for new trades
                for trade in trades:
                    trade_volume_base = trade['amount']
                    trade_volume_usd = trade['amount'] * trade['price']
                    base_volume += trade_volume_base
                    usd_volume += trade_volume_usd
                    cvd += trade_volume_usd if trade['side'] == 'buy' else -trade_volume_usd

                    # Update total volume across all exchanges
                    async with self.lock:
                        self.aggregated_data['aggregated_volume'] += trade_volume_usd

                # Update and self.logger.info the aggregated data
                async with self.lock:
                    self.aggregated_data[exchange_name] = {
                        'base_volume': base_volume,
                        'usd_volume': usd_volume,
                        'cumulative_volume_delta': cvd
                    }
                    self.logger.info(self.aggregated_data)
                    self.window.update_aggregated_data(self.aggregated_data)

            except KeyboardInterrupt:
                self.logger.info(f"KeyboardInterrupt detected. Closing {exchange_name}...")
                break

        await exchange.close()
        self.logger.info(f"{exchange_name} closed.")

    def trigger_update_exchanges(self, new_exchanges):
        # This function is called when the user clicks the "Start/Refresh" button and
        # self.started == True
        # We schedule a task that will start or update the watched exchanges
        self.loop.call_soon_threadsafe(lambda: asyncio.create_task(self.update_exchanges_coroutine(new_exchanges)))

    async def update_exchanges_coroutine(self, new_exchanges):
        # This will add a newly added exchange to the watch exchanges loop
        for exchange_name in new_exchanges:
            if exchange_name not in self.active_exchanges:
                await self.update_exchange(exchange_name, True)

        # This will remove an exchange that was previously being watched in the loop
        for exchange_name in self.active_exchanges:
            if exchange_name not in new_exchanges:
                await self.update_exchange(exchange_name, False)

    async def update_exchange(self, exchange_name, active):
        # This schedules the task by storing it in the queue, the "manage_exchanges" function will watch for changes to the queue
        await self.exchange_changes_queue.put((exchange_name, active))

    async def main(self):
        # Add the initial exchanges to the queue
        for exchange_name in self.exchange_names:
            await self.exchange_changes_queue.put((exchange_name, True))

        # Start the manage_exchanges method
        manager_task = asyncio.create_task(self.manage_exchanges())
        await manager_task

    async def manage_exchanges(self):
        while True:
            exchange_name, active = await self.exchange_changes_queue.get()

            # This is where we update the active exchanges we are watching
            if active:
                if exchange_name not in self.active_exchanges:
                    coroutine = self.watch_exchange(exchange_name)
                    task = asyncio.create_task(coroutine)
                    self.active_exchanges[exchange_name] = task
            else:
                if exchange_name in self.active_exchanges:
                    task = self.active_exchanges.pop(exchange_name)
                    task.cancel()

                    # Wait for the canceled task to complete
                    await asyncio.gather(task, return_exceptions=True)

    def start_thread(self, window, exchanges=None):
        self.logger.info("Starting.")
        self.window = window
        self.exchange_names = exchanges if exchanges is not None else self.exchange_names
        self.thread.start()
        self.started = True

    def start_loop(self):
        asyncio.set_event_loop(self.loop)

        # Pass the window and exchanges to the main method
        self.loop.create_task(self.main())
        self.loop.run_forever()

    async def close_exchanges(self):
        for task in self.active_exchanges.values():
            task.cancel()
        for _, exchange in self.active_exchanges.items():
            await exchange.close()

    def on_close(self):
        if self.thread.is_alive():
            self.logger.info("Stopping threads.")
            self.loop.stop()
            self.thread.join()
            self.loop.run_until_complete(self.close_exchanges())