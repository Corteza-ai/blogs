"""To read the main parameters needed to train an AE.
"""

import argparse


def load_args() -> argparse.Namespace:
    """Loads necessary arguments from the command line.

    Returns:
        argparse.Namespace: It contains the user defined data_path, epochs and lr.
    """
    arguments = argparse.ArgumentParser()

    arguments.add_argument(
        "--data_path",
        type=str,
        default="./17_flowers/train",
        help="Root directory of the dataset.",
    )

    arguments.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of epochs to training the auto-encoder.",
    )

    arguments.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate for optimization."
    )

    args = arguments.parse_args()
    return args
