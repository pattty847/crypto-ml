import dearpygui.dearpygui as dpg

class Window:
    def __init__(self, label, tag) -> None:
        self.tag = dpg.add_window(label=label, tag=tag)
        self.exchange_items = {}

    def build(self):
        # Add text widget for displaying total volume
        self.total_volume_label = dpg.add_text("Total Volume Across All Exchanges: ", parent=self.tag)
        self.total_volume_value = dpg.add_text("0", parent=self.tag)

        # Separator
        dpg.add_separator(parent=self.tag)

        # Initial build of exchange data widgets (will be populated later)
        self.exchange_container = dpg.add_child_window(parent=self.tag)

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
                with dpg.child(parent=self.exchange_container) as exchange_item:
                    dpg.add_text(f"Exchange: {exchange_name}")
                    dpg.add_text("Base Volume: ")
                    base_volume_value = dpg.add_text(str(data['base_volume']))
                    dpg.add_text("USD Volume: ")
                    usd_volume_value = dpg.add_text(str(data['usd_volume']))
                    dpg.add_text("Cumulative Volume Delta (CVD): ")
                    cvd_value = dpg.add_text(str(data['cumulative_volume_delta']))
                    dpg.add_separator()

                    # Store value widgets for future updates
                    self.exchange_items[exchange_name] = {
                        'base_volume': base_volume_value,
                        'usd_volume': usd_volume_value,
                        'cumulative_volume_delta': cvd_value
                    }
            else:
                # Update existing exchange item
                dpg.set_value(self.exchange_items[exchange_name]['base_volume'], str(data['base_volume']))
                dpg.set_value(self.exchange_items[exchange_name]['usd_volume'], str(data['usd_volume']))
                dpg.set_value(self.exchange_items[exchange_name]['cumulative_volume_delta'], str(data['cumulative_volume_delta']))