import logging
import sys

def setup_logger(name: str):
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    handler.setFormatter(formatter)
