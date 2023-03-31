import dearpygui.dearpygui as dpg

class BaseGUIObject:
    def __init__(self, label, stage=True, widget_function=None, **kwargs):
        if stage:
            with dpg.stage() as self.stage:
                self.tag = widget_function(label=label, **kwargs)
        else:
            self.tag = widget_function(label=label, **kwargs)

    def submit(self, parent):
        dpg.push_container_stack(parent)
        dpg.unstage(self.stage)
        dpg.pop_container_stack()