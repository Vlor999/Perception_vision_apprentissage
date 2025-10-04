from src.conv_model.conv import main
from time import time
from loguru import logger
import cProfile
import sys

logger.remove()
logger.add(sys.stderr, level="DEBUG")

if __name__ == "__main__":
    tic = time()
    main()
    tac = time()
    logger.info(f"Time to render: {tac - tic}")
