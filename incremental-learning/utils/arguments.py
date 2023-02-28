"""This file defines the main parameters needed to train our auto-encoder.
"""

__all__ = ["load_args"]

import argparse


def load_args() -> argparse.Namespace:
    """
    Receives from terminal a set of arguments needed for the main functions.

    Returns:
        a argparse.Namespace object
    """
    arguments = argparse.ArgumentParser()
    arguments.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Training approach: ent-to-end or incremental.",
    )

    arguments.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of epochs to training a entire network or each layer on incremental approach.",
    )

    arguments.add_argument(
        "--last-num-epochs",
        type=int,
        default=50,
        help="Number of epochs to train the las stage of the incremental approach.",
    )

    arguments.add_argument(
        "--lr", type=float, default=0.005, help="Learning rate for optimization."
    )

    args = arguments.parse_args()
    return args
