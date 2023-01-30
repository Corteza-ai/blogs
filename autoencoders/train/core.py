"""
This module contains the main functions to run the training process.
"""

import logging
from typing import Callable, List, Tuple
import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import tqdm


def train_fn(
    model: nn.Module, criterion: Callable, optim: Optimizer, data: DataLoader
) -> float:
    """Trains a model for one epoch.

    Args:
        model (nn.Module): The pytorch model to be trained.
        criterion (Callable): The loss function.
        optim (Optimizer): The optimizer.
        data (DataLoader): The training dataset.

    Returns:
        float: The epoch loss, which is computed as the accumulated loss
            thru the epoch divided by the number of batches.
    """
    _device_ = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    total_loss = 0.0
    for imgs, _ in data:
        imgs = imgs.to(_device_)
        optim.zero_grad()
        x_hat = model(imgs)
        loss = criterion(x_hat, imgs)
        loss.backward()
        optim.step()
        total_loss += loss.item()
    return total_loss / len(data)


def train(
    model: nn.Module,
    dataset: DataLoader,
    test: Tensor,
    epochs: int = 500,
    learning_rate: float = 0.01,
) -> Tuple[List[float], List[np.ndarray]]:
    """Runs the training process for a given number of epochs.

    Args:
        model (nn.Module): The pytorch model to be trained.
        dataset (DataLoader): The training dataset.
        test (Tensor): A set of images to test the model.
        epochs (int, optional): Number of epoch to run. Defaults to 500.
        learning_rate (float, optional): Learing rate for the optimizer.
            Defaults to 0.01.

    Returns:
        Tuple[List[float], List[np.ndarray]]: the loss and reconstruction history.
            The reconstruction history is based on the set of testing images.
    """
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True
    )
    criterion = nn.MSELoss()
    losses = []
    recons = [test.detach().cpu().numpy().transpose(0, 2, 3, 1)]
    progress = tqdm.tqdm(range(epochs + 1), desc="loss: Inf")
    for epoch in progress:
        loss = train_fn(model, criterion, optimizer, dataset)
        if epoch % 50 == 0:
            model.eval()
            x_hat = model(test)
            recons.append(x_hat.detach().cpu().numpy().transpose(0, 2, 3, 1))
        logging.info("Epoch %i | Loss: %f", epoch, loss)
        progress.set_description(f"Loss: {round(loss, 4)}")
        losses.append(loss)
    return losses, recons
