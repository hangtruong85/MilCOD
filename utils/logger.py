import logging
import os
from datetime import datetime

def setup_logger(log_dir="logs", filename="train.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)

    # clear old handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Logger initialized → {log_path}")
    return logger
