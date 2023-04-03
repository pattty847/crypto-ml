from gui.viewport import ViewPort
from gui.aggregate_window import Window

if __name__ == '__main__':
    # Main entry point to program
    with ViewPort(title='Custom Title') as viewport:
        # window = Window('win', viewport.tag, viewport.aggr).build()
        viewport.run()