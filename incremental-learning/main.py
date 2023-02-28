"""Use this file to train auto-encoders."""

import logging

from trainer import main_train
from utils import load_args

logging.basicConfig(
    filename="run.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
)

if __name__ == "__main__":
    args = load_args()
    main_train(args)
