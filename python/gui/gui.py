import asyncio
import dearpygui.dearpygui as dpg
import aggregate
from user_login import Window

class ViewPort:
    def __init__(self, title, width, height, x_pos=0, y_pos=0):
        self.title = title
        self.width = width
        self.height = height
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.tag = "root"
        self.objects = []

    def __enter__(self):
        dpg.create_context()
        dpg.add_window(tag=self.tag)
        self.setup_viewport()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window(self.tag, True)
        return self

    def setup_viewport(self):
        dpg.create_viewport(title=self.title, width=self.width, height=self.height, x_pos=self.x_pos, y_pos=self.y_pos)

    def build_gui(self):
        self.window = Window('win', "1")
        self.window.build()
        return self.window

    def add_object(self, obj):
        self.objects.append(obj)

    def run(self):
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()

    def __exit__(self, exc_type, exc_val, exc_tb):
        dpg.destroy_context()


with ViewPort(title='Custom Title', width=600, height=200, x_pos=500, y_pos=500) as viewport:
    window = viewport.build_gui()
    window.update_aggregated_data()
    viewport.run()