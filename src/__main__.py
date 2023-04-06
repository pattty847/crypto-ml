from gui.viewport import Viewport
from gui.aggregate_window import Window
from program import Program 

if __name__ == '__main__':
    # Main entry point to program
    with Viewport(title='Custom Title') as viewport:
        # window = Window('win', viewport.tag, viewport.aggr).build()
        
        # Define our Program class, passing it the viewport, and call build_ui()
        program = Program(viewport).build_ui()
        
        # This will run the dearpygui loop for the program (call after UI elements are built)
        viewport.run()