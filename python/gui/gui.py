import asyncio
import logging
import threading

import dearpygui.dearpygui as dpg
import dearpygui.demo as demo
from aggregate import CryptoData
from aggregate_window import Window
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
    def __init__(self, title, width, height, x_pos=0, y_pos=0):
        self.title = title
        self.width = width
        self.height = height
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.tag = "root"
        self.aggr = CryptoData()
        self.thread = threading.Thread()
        self.logger = logging.getLogger(__name__)
        self.monitor = get_monitors(is_primary=True)[0]
        self.logger.info(f"Primary Monitor: {self.monitor}")

    def __enter__(self):
        self.logger.info("Setting up DearPyGUI.")
        """
        The __enter__ function is called when the context manager is entered.
        It should return an object that will be assigned to the variable in the as clause of a with statement. 
        The __exit__ function is called when leaving the context manager, and it receives three arguments: 
        the exception type, exception value and traceback object.
        
        :param self: Access the class attributes and methods
        :return: The object itself
        :doc-author: Trelent
        """
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
        dpg.create_viewport(title=self.title, width=window_width, height=window_height, x_pos=window_x, y_pos=window_y)        
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window(self.tag, True)
        return self

    def run(self):
        self.logger.info("Running DearPyGUI loop.")
        """
        The run function is the main loop of DearPyGui.
        It will run until dpg.is_dearpygui_running() returns False, which can be done by calling dpg.stop_dearpygui().
        The function should be called from a thread other than the main thread.
        
        :param self: Represent the instance of the class
        :return: Nothing
        :doc-author: Trelent
        """
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info("Destroying DearPyGUI context.")
        """
        The __exit__ function is called when the context manager exits.
        The __exit__ function can accept three arguments: exc_type, exc_value and traceback.
        If an exception was raised in the with block, these arguments will be set to the exception type, value and traceback respectively. Otherwise they are all None.
        
        :param self: Refer to the instance of the class
        :param exc_type: Determine the type of exception that was raised
        :param exc_val: Store the exception value
        :param exc_tb: Get the traceback object
        :return: A boolean that determines if the exception is suppressed or not
        :doc-author: Trelent
        """
        self.aggr.on_close()
        dpg.destroy_context()

# Main entry point to program
with ViewPort(title='Custom Title', width=1200, height=800, x_pos=500, y_pos=500) as viewport:
    window = Window('win', viewport.tag, viewport.aggr).build()
    viewport.run()