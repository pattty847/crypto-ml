import dearpygui.dearpygui as dpg
import pandas as pd
from screeninfo import get_monitors

class Chart:

    # candles = asyncio.run(
    #     data.fetch_candles(
    #         exchange_name,
    #         symbol,
    #         timeframe,
    #         "2023-03-04 00:00:00",
    #         1000,
    #         False
    #     )
    # )

    def __init__(self, file) -> None:
        self.exchange_name = "cryptocom"
        self.symbol = "BTC/USDT"
        self.timeframe = "1m"

        self.columns = 1
        self.rows = 1
        self.num_charts = 1

        columns = ['dates', 'opens', 'highs', 'lows', 'closes', 'volumes']

        # Read the CSV file into a pandas DataFrame using the specified columns
        self.candles = pd.read_csv(file, usecols=columns)
        # Convert the 'dates' column to datetime format
        self.candles['dates'] = pd.to_datetime(self.candles['dates'])

        # Convert the datetime values to seconds since the Unix epoch
        self.candles['dates'] = self.candles['dates'].astype(int) // 10**9

        self.init()

    def get_window_position(self):
        primary_monitor = get_monitors()[0]
        width = int(primary_monitor.width * 0.8)
        height = int(primary_monitor.height * 0.8)
        x_pos = int(primary_monitor.x + (primary_monitor.width - width) / 2)
        y_pos = int(primary_monitor.y + (primary_monitor.height - height) / 2)
        return x_pos, y_pos, width, height


    def add_chart(self, s, a, u):
        self.num_charts += 1
        self.columns += 1

        # Calculate the number of rows based on the current number of charts and columns
        self.rows = (self.num_charts + self.columns - 1) // self.columns

        # Calculate the row and column ratios
        row_ratio = [100.0 / self.rows] * self.rows
        col_ratio = [100.0 / self.columns] * self.columns

        # Configure the subplot
        dpg.configure_item('subplot', columns=self.columns, rows=self.rows, row_ratios=row_ratio, column_ratios=col_ratio)

        # Add the plot
        dpg.add_plot(parent='subplot')

        print(f"# of charts: {self.num_charts} # of cols: {self.columns} # of rows: {self.rows} row ratio: {row_ratio} col ratio: {col_ratio}")


    def init(self):
        dpg.create_context()

        with dpg.window(label=self.exchange_name.upper(), tag="Primary Window"):

            dpg.add_button(label="Add Chart", callback=self.add_chart)

            with dpg.subplots(
                    rows=2,
                    columns=1,
                    label=f"{self.symbol} | {self.timeframe}",
                    row_ratios=[70, 30],
                    link_all_x=True,
                    width=-1,
                    height=-1,
                    tag='subplot'
                ):
                with dpg.plot():
                    dpg.add_plot_legend()

                    xaxis_candles = dpg.add_plot_axis(dpg.mvXAxis, time=True)

                    with dpg.plot_axis(dpg.mvYAxis, label="USD"):

                        dpg.add_candle_series(
                            self.candles["dates"].values,
                            self.candles["opens"].values,
                            self.candles["closes"].values,
                            self.candles["lows"].values,
                            self.candles["highs"].values,
                            tag="candle_series",
                            time_unit=dpg.mvTimeUnit_Min,
                        )

                        dpg.fit_axis_data(dpg.top_container_stack())
                        dpg.fit_axis_data(xaxis_candles)

                with dpg.plot():
                    dpg.add_plot_legend()
                    xaxis_vol = dpg.add_plot_axis(dpg.mvXAxis, label="Time [UTC]", time=True)

                    with dpg.plot_axis(dpg.mvYAxis, label="USD"):
                        dpg.add_bar_series(
                            self.candles['dates'].values,
                            self.candles['volumes'].values,
                            weight=dpg.mvTimeUnit_Min
                        )

                        dpg.fit_axis_data(dpg.top_container_stack())
                        dpg.fit_axis_data(xaxis_vol)

        x_pos, y_pos, width, height = self.get_window_position()
        dpg.create_viewport(title='Custom Title', width=width, height=height, x_pos=x_pos, y_pos=y_pos)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Primary Window", True)
        dpg.start_dearpygui()
        dpg.destroy_context()


# if __name__ == "__main__":
#     c = Chart(file="data/exchange/coinbasepro/BTC_USD_1d.csv")