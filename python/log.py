import logging

def setup_logging():
    """
    The setup_logging function configures the logging module to log messages
    to stdout. It is called by main() and should not be called directly.
    
    :return: A logger object
    :doc-author: Trelent
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
