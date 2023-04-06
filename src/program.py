# This is the main program class where all UI elements contained on the front page are defind
import dearpygui.dearpygui as dpg
import dearpygui.demo as demo
from gui.strategy import Strategy

# Name can be changed to the official name later.
class Program:
    def __init__(self, viewport) -> None:
        self.viewport = viewport
        self.parent = self.viewport.tag
    
    # First call for this class
    def build_ui(self):
        self.add_menu_bar()
        self.add_tab_bar()
    
    # UI elements defined below
    def add_menu_bar(self):
        with dpg.menu_bar(parent=self.parent):
            with dpg.menu(label="Settings"):
                dpg.add_menu_item(label="Demo", callback=demo.show_demo)

    def add_tab_bar(self):
        # Add a tab bar to the main window
        with dpg.tab_bar(parent=self.parent):
            # Add a tab for the candlestick chart
            # with dpg.tab(label='Candlestick Chart'):
            #     self.candle_chart = CandleChart()
            # Add a tab for interactive controls
            with dpg.tab(label='Strategy') as interactive_controls:
                self.interactive_controls = Strategy(parent=interactive_controls, viewport=self.viewport)
                
            with dpg.tab(label="Test Tab") as test_tab:
                pass
                
    # Callbacks
    def open_file(sender, app_data):
        # Implement logic to open a file
        pass

    def save_file(sender, app_data):
        # Implement logic to save a file
        pass

    def exit_app(sender, app_data):
        # Exit the application
        dpg.stop_dearpygui()

    def open_preferences(sender, app_data):
        # Implement logic to open the preferences window
        pass