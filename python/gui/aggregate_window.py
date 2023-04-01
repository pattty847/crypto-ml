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

    def add_or_remove_exchange(self, sender, app_data, user_data):
        before = self.active_exchanges.copy()
        self.active_exchanges = [ex for ex in self.active_exchanges if ex != sender] if sender in self.active_exchanges else self.active_exchanges + [sender]
        self.logger.info(f"Watched Exchanges - Before:{before} | After:{self.active_exchanges}")

    def start_watching_exchanges(self, s, a, u):
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
    
    def exchange_selection(self, s, a, u):
        self.exchange_selection_tag = dpg.generate_uuid()
        with dpg.window(modal=True, popup=True, width=300, height=300, tag=self.exchange_selection_tag, on_close=lambda:dpg.delete_item(self.exchange_selection_tag)):
            dpg.add_text("Exchanges to Watch")
            with dpg.child_window(height=dpg.get_item_configuration(self.exchange_selection_tag)['height'] * 0.72):
                for exchange in ccxt.exchanges:
                    dpg.add_checkbox(
                        label=exchange, 
                        callback=self.add_or_remove_exchange, 
                        tag=exchange,
                        default_value=True if exchange in self.active_exchanges else False
                    )

            dpg.add_button(label="Start", callback=self.start_watching_exchanges)

    def build(self):
        self.logger.info("Building menu for main window.")
        with dpg.menu_bar(parent=self.tag):
            dpg.add_menu_item(label="Demo", callback=demo.show_demo)
            dpg.add_menu_item(label="Exchanges", callback=self.exchange_selection)
        
        # Add text widget for displaying total volume
        self.total_volume_label = dpg.add_text("Total Volume Across All Exchanges: ", parent=self.tag)
        self.total_volume_value = dpg.add_text("0", parent=self.tag)

        # Separator
        dpg.add_separator(parent=self.tag)

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
                with dpg.child_window(tag=f"{self.tag}_{exchange_name}", parent=self.tag, width=-1, height=105):
                    with dpg.group(horizontal=True):
                        dpg.add_text(f"Exchange: {exchange_name}")
                        dpg.add_button(label="X")
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
                # Update existing exchange item
                dpg.set_value(self.exchange_items[exchange_name]['base_volume'], format(data['base_volume'], '.4f'))
                dpg.set_value(self.exchange_items[exchange_name]['usd_volume'], format(data['usd_volume'], '.2f'))
                dpg.set_value(self.exchange_items[exchange_name]['cumulative_volume_delta'], format(data['cumulative_volume_delta'], '.2f'))