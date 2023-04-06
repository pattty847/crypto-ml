import dearpygui.dearpygui as dpg

class StrategyBuilder:
    def __init__(self) -> None:
        self.source = None
    
    def build_ui(self):
        if dpg.does_alias_exist('strategy_builder'):
            return
        with dpg.window(label="Strategy Builder", width=700, height=700, tag='strategy_builder') as strategy_builder_window:
            with dpg.menu_bar():
                with dpg.menu(label="Add Node"):
                    dpg.add_menu_item(label="SMA", callback=self.add_sma_node, user_data=strategy_builder_window)
                    dpg.add_menu_item(label="EMA", callback=self.add_ema_node, user_data=strategy_builder_window)
                    dpg.add_menu_item(label="MACD", callback=self.add_macd_node, user_data=strategy_builder_window)
                    dpg.add_menu_item(label="CROSSOVER", callback=self.add_crossover_node, user_data=strategy_builder_window)
                    dpg.add_menu_item(label="CROSSUNDER", callback=self.add_crossunder_node, user_data=strategy_builder_window)

            with dpg.node_editor(parent=strategy_builder_window, callback=lambda sender, app_data: dpg.add_node_link(app_data[0], app_data[1], parent=sender), 
                                delink_callback=lambda sender, app_data: dpg.delete_item(app_data), minimap=True, minimap_location=dpg.mvNodeMiniMap_Location_BottomRight) as self.node_editor:

                # Define the initial nodes and attributes
                # ...
                with dpg.node(label="Node 1", pos=[10, 10]):
                    with dpg.node_attribute():
                            dpg.add_input_float(label="F1", width=150)

                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output):
                        dpg.add_input_float(label="F2", width=150)

    def add_sma_node(self, sender, user_data):
        # User data contains the parent (strategy_builder_window)
        parent = user_data
        # Add an SMA node to the node editor
        with dpg.node(label="SMA", parent=self.node_editor, pos=[10, 10]):
            # Define the attributes for the SMA node
            # ...
            with dpg.node_attribute():
                dpg.add_input_int(label="Input", default_value=25, width=150)

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_input_float(label="Output", width=150)

    def add_ema_node(self, sender, user_data):
        # User data contains the parent (strategy_builder_window)
        parent = user_data
        # Add an EMA node to the node editor
        with dpg.node(label="EMA", parent=self.node_editor, pos=[10, 10]):
            # Define the attributes for the EMA node
            # ...
            with dpg.node_attribute():
                dpg.add_input_float(label="F1", width=150)

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_input_float(label="F2", width=150)

    def add_macd_node(self, sender, user_data):
        # User data contains the parent (strategy_builder_window)
        parent = user_data
        # Add a MACD node to the node editor
        with dpg.node(label="MACD", parent=self.node_editor, pos=[10, 10]):
            # Define the attributes for the MACD node
            # ...
            with dpg.node_attribute():
                dpg.add_input_float(label="F1", width=150)

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_input_float(label="F2", width=150)

    def add_crossover_node(self, sender, user_data):
        # User data contains the parent (strategy_builder_window)
        parent = user_data
        # Add a CROSSOVER node to the node editor
        with dpg.node(label="CROSSOVER", parent=self.node_editor, pos=[10, 10]):
            # Define the attributes for the CROSSOVER node
            # ...
            with dpg.node_attribute():
                dpg.add_input_float(label="F1", width=150)

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_input_float(label="F2", width=150)

    def add_crossunder_node(self, sender, user_data):
        # User data contains the parent (strategy_builder_window)
        parent = user_data
        # Add a CROSSUNDER node to the node editor
        with dpg.node(label="CROSSUNDER", parent=self.node_editor, pos=[10, 10]):
            # Define the attributes for the CROSSUNDER node
            # ...
            with dpg.node_attribute():
                dpg.add_input_float(label="F1", width=150)

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_input_float(label="F2", width=150)
                
    def set_source(self, data):
        self.source = data