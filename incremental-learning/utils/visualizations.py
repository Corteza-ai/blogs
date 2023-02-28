"""Useful function to visualize images and losses."""

__all__ = ["plot_loss_history", "plot_multiple_images"]

from typing import List, Tuple, Union

import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np


def plot_loss_history(
    loss_1: List[float],
    label_1: str,
    loss_2: Union[List[float], None] = None,
    label_2: Union[str, None] = None,
    fname: str = "./loss_history.png",
) -> None:
    """
    Plots the history of the loss over training or validation.

    Args:
        loss_1 (List[float]): A list containing the losses of each epoch or step.
        label_1 (string): legend for the loss 1.
        loss_2 (Optional | List[float]): A list containing the losses of each epoch or step,
                                         It is needed to compared two models.
        label_2 (string): legend for the loss 2.
        fname (string): Name for the saved file.
    """

    plt.plot(loss_1, label=label_1)
    if loss_2 is not None:
        if label_2 is None:
            label_2 = "Other"
        plt.plot(loss_2, label=label_2)
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(fname)
    plt.close()


def plot_multiple_images(
    batch: np.ndarray,
    filename: str,
    rows: int = 2,
    cols: int = 5,
    figsize: Tuple[int, int] = (10, 7),
) -> None:
    """
    Plots images in a grid.

    Args:
        batch (numpy.ndarray): Images to plot.
        filename (string): file name for saving.
        rows (integer): number of rows in the grid (rows = num_images / num_cols).
        cols (integer): number of columns in the grid (cols = num_images / num_rows).
        figsize (Tuple[int, int]): Size of the figure.

    ::note: This function saves an image in your working directory.

    """
    fig = plt.figure(figsize=figsize)
    for i in range(1, batch.shape[0] + 1):
        fig.add_subplot(rows, cols, i)
        plt.imshow((batch[i - 1] * 255).astype(int))
        plt.axis("off")
    plt.savefig(filename)
    plt.close()
