"""To train an autoencoder.
"""

import logging
from utils.arguments import load_args
from train.training_process import main_train

logging.basicConfig(
    filename="run.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
)

if __name__ == "__main__":
    args = load_args()
    main_train(args.data_path, args.epochs, args.lr)
