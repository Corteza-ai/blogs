"""To viualize images.
"""

from typing import List, Tuple
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np


def plot_multiple_images(
    batch: np.ndarray,
    filename: str,
    rows: int = 2,
    cols: int = 5,
    figsize: Tuple[int, int] = (10, 7),
) -> None:
    """Plots a grid of images.

    Args:
        batch (np.ndarray): images to be plotted.
        filename (str): The name to save the plot.
        rows (int, optional): Number of rows in the grid. Defaults to 2.
        cols (int, optional): Number of columns in the grid. Defaults to 5.
        figsize (Tuple[int, int], optional): The size of the image. Defaults to (10, 7).
    """
    fig = plt.figure(figsize=figsize)
    for i in range(1, batch.shape[0] + 1):
        fig.add_subplot(rows, cols, i)
        plt.imshow((batch[i - 1] * 255).astype(int))
        plt.axis("off")
    plt.savefig(filename)
    plt.close()


def visualize_intermediate_reconstruction(
    recons: List[np.ndarray], iterations: int
) -> None:
    """Plots the reconstruction history of images.

    Args:
        recons (List[np.ndarray]): The reconstruction history of a
            sample of images.
        rows (int): Number of iterations. Equals to the lenght of the
            reconstruction history plus two.
    """
    arr_recons = np.vstack(recons)
    plot_multiple_images(arr_recons, "./reconstruction_history.png", iterations)
