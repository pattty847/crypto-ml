import logging
import threading
import dearpygui.dearpygui as dpg

from gui.interactive_controls import InteractiveControls
from .aggregate import CryptoData
from .aggregate_window import Window
from screeninfo import get_monitors

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | File: %(name)s | Log Type: %(levelname)s | Message: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class ViewPort:
    """
        This is a context manager to abstract the dearpygui setup and the main program setup.
    """
    def __init__(self, title):
        self.title = title
        self.tag = "root"
        self.aggr = CryptoData()
        self.thread = threading.Thread()
        self.logger = logging.getLogger(__name__)
        self.monitor = get_monitors(is_primary=True)[0]
        self.logger.info(f"Primary Monitor: {self.monitor}")

    def __enter__(self):
        self.logger.info("Setting up DearPyGUI.")
        def get_centered_window_dimensions_(monitor):
            # Calculate 70% of the monitor's width and height
            window_width = int(monitor.width * 0.7)
            window_height = int(monitor.height * 0.7)

            # Calculate the position for the centered window
            window_x = monitor.x + (monitor.width - window_width) // 2
            window_y = monitor.y + (monitor.height - window_height) // 2

            return window_x, window_y, window_width, window_height
        
        window_x, window_y, window_width, window_height = get_centered_window_dimensions_(self.monitor)
        dpg.create_context()
        dpg.add_window(tag=self.tag)
        self.add_menu_bar()
        self.add_tab_bar()
        dpg.create_viewport(title=self.title, width=window_width, height=window_height, x_pos=window_x, y_pos=window_y)        
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window(self.tag, True)
        return self
    
    def add_menu_bar(self):
        with dpg.menu_bar(parent=self.tag):
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Open", callback=self.open_file)
                dpg.add_menu_item(label="Save", callback=self.save_file)
                dpg.add_menu_item(label="Exit", callback=self.exit_app)

            with dpg.menu(label="Settings"):
                dpg.add_menu_item(label="Preferences", callback=self.open_preferences)

    def add_tab_bar(self):
        # Add a tab bar to the main window
        with dpg.tab_bar(parent=self.tag):
            # Add a tab for the candlestick chart
            # with dpg.tab(label="Candlestick Chart"):
            #     self.candle_chart = CandleChart()

            # Add a tab for interactive controls
            with dpg.tab(label="Interactive Controls"):
                self.interactive_controls = InteractiveControls()

    def run(self):
        self.logger.info("Running DearPyGUI loop.")
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info("Destroying DearPyGUI context.")
        self.aggr.on_close()
        dpg.destroy_context()