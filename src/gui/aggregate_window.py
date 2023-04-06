import dearpygui.dearpygui as dpg
import ccxt
import logging
import dearpygui.demo as demo

class Window:
    def __init__(self, label, parent, aggregate) -> None:
        self.tag = dpg.add_child_window(label=label, tag=dpg.generate_uuid(), parent=parent, menubar=True)
        self.exchange_items = {}
        self.active_exchanges = []
        self.aggr = aggregate
        self.logger = logging.getLogger(__name__)
        self.viewport_size = (dpg.get_viewport_client_width(), dpg.get_viewport_client_width())

    def build(self):
        self.logger.info("Building main window.")
        self.build_menu_bar()

        with dpg.group(parent=self.tag):
            with dpg.group(horizontal=True, show=False, tag="total_volume"):
                # Add text widget for displaying total volume
                self.total_volume_label = dpg.add_text("Total Volume Across All Exchanges: ")
                self.total_volume_value = dpg.add_text("0")
            
            with dpg.group(tag="watch_trades", show=False):
                pass

            with dpg.child_window(pos=(self.viewport_size[0] - 300, 24), tag="active_exchanges", show=False):
                dpg.add_listbox([], tag="active_exchanges_listbox", show=False, width=-1, num_items=10)
    
    def build_menu_bar(self):
        with dpg.menu_bar(parent=self.tag):
            dpg.add_menu_item(label="Demo", callback=demo.show_demo)
            dpg.add_menu_item(label="Watch Trades", callback=self.watch_trades_callback)
            dpg.add_menu_item(label="Exchanges", callback=self.exchange_selection_callback, show=False, tag="exchanges_menu")

    def watch_trades_callback(self):
        item_tags = [
            "exchanges_menu",
            "watch_trades",
            "active_exchanges",
            "active_exchanges_listbox",
            "total_volume"
        ]

        for tag in item_tags:
            current_show = dpg.get_item_configuration(tag)['show']
            dpg.configure_item(tag, show=not current_show)

        if not current_show:
            dpg.configure_item("active_exchanges_listbox", items=self.active_exchanges)


    def exchange_selection_callback(self, s, a, u):
        self.exchange_selection_tag = dpg.generate_uuid()
        with dpg.window(modal=True, popup=True, width=300, height=300, tag=self.exchange_selection_tag, on_close=lambda:dpg.delete_item(self.exchange_selection_tag)):
            dpg.add_text("Exchanges to Watch")
            with dpg.child_window(height=-1, width=-1):
                for exchange in ccxt.exchanges:
                    dpg.add_checkbox(
                        label=exchange, 
                        callback=self.add_or_remove_exchange_callback, user_data=exchange, 
                        tag=exchange,
                        default_value=True if exchange in self.active_exchanges else False
                    )

    def add_or_remove_exchange_callback(self, sender, app_data, user_data):
        before = self.active_exchanges.copy()
        self.active_exchanges = [ex for ex in self.active_exchanges if ex != user_data] if user_data in self.active_exchanges else self.active_exchanges + [user_data]
        self.logger.info(f"Watched Exchanges - Before:{before} | After:{self.active_exchanges}")
        dpg.configure_item("active_exchanges_listbox", items=self.active_exchanges)
        self.update_watch()

    def update_watch(self):
        if not self.aggr.started:
            self.aggr.start_thread(self, self.active_exchanges)
        else:
            self.aggr.trigger_update_exchanges(self.active_exchanges)

        self.remove_unwatched_exchanges()

    def remove_unwatched_exchanges(self):
        # Delete removed exchanges
        for exchange_name in list(self.exchange_items.keys()):
            if exchange_name not in self.active_exchanges:
                dpg.delete_item(f"{self.tag}_{exchange_name}")
                del self.exchange_items[exchange_name]

    def update_aggregated_data(self, aggregated_data):
        # Update total volume value
        dpg.set_value(self.total_volume_value, str(aggregated_data['aggregated_volume']))

        # Update individual exchange data
        for exchange_name, data in aggregated_data.items():
            # Skip the total_volume key when printing individual exchange data
            if exchange_name == 'aggregated_volume':
                continue

            if exchange_name not in self.exchange_items:
                # Create new exchange item if not exists
                with dpg.child_window(tag=f"{self.tag}_{exchange_name}", parent="watch_trades", width=self.viewport_size[1] - 330, height=105):
                    with dpg.group(horizontal=True):
                        dpg.add_text(f"Exchange: {exchange_name}")
                        dpg.add_button(label="X", callback=self.add_or_remove_exchange_callback, user_data=exchange_name)
                        dpg.add_button(label="Alerts", callback=lambda: self.add_or_remove_alert_callback(exchange_name))
                    with dpg.group(horizontal=True):
                        dpg.add_text("Base Volume: ")
                        base_volume_value = dpg.add_text(str(data['base_volume']))
                    with dpg.group(horizontal=True):
                        dpg.add_text("USD Volume: ")
                        usd_volume_value = dpg.add_text(str(data['usd_volume']))
                    with dpg.group(horizontal=True):
                        dpg.add_text("Cumulative Volume Delta (CVD): ")
                        cvd_value = dpg.add_text(str(data['cumulative_volume_delta']))
                    # dpg.add_separator()

                    # Store value widgets for future updates
                    self.exchange_items[exchange_name] = {
                        'base_volume': base_volume_value,
                        'usd_volume': usd_volume_value,
                        'cumulative_volume_delta': cvd_value
                    }
            else:
                self.aggr.check_for_alert()
                # Update existing exchange item
                dpg.set_value(self.exchange_items[exchange_name]['base_volume'], format(data['base_volume'], '.4f'))
                dpg.set_value(self.exchange_items[exchange_name]['usd_volume'], format(data['usd_volume'], '.2f'))
                dpg.set_value(self.exchange_items[exchange_name]['cumulative_volume_delta'], format(data['cumulative_volume_delta'], '.2f'))

    def add_or_remove_alert_callback(self, exchange_name):
        with dpg.window(modal=True, popup=True, width=500, height=500):
            dpg.add_text(f"{exchange_name.upper()} alerts")