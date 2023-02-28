"""Core training functions."""

__all__ = ["train_fn", "train"]

# pylint: disable=no-member

import logging
from typing import Callable, List, Optional

import torch
import tqdm  # type: ignore[import]
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from utils import plot_multiple_images


def _get_test(dataset: DataLoader, device: str) -> torch.Tensor:
    """Returns a subset of images"""
    test = iter(dataset)
    test_ = next(test)[0][:10]
    test_ = test_.to(device)
    return test_


def _save_intermediate_reconstructions(
    model: torch.nn.Module, images: torch.Tensor, epoch: int
) -> None:
    """Saves a set of images reconstructions"""
    model.eval()
    x_hat = model(images)
    img = torch.cat([images, x_hat], axis=0)  # type: ignore
    img = img.detach().cpu().numpy().transpose(0, 2, 3, 1)
    if img.shape[-1] == 3:
        plot_multiple_images(img, f"restoration_epoch{epoch}.png", 4)


def train_fn(
    model: nn.Module,
    criterion: Callable,
    optim: Optimizer,
    data: DataLoader,
    device: str = "cuda",
    feature_extractor: Optional[nn.Module] = None,
) -> float:
    """Trains a model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        criterion (torch.nn.Loss): The loss function to be minimized.
        optim (torch.optim.Optimizer): The optimizer for training.
        data (torch.utils.data.DataLoader): The data set to train the model.
        feature_extractor (Union[torch.nn.Module, None]): Used only for incremental training.

    Returns:
        torch.Tensor: containing the original images,
        torch.Tensor: containing the image reconstructons,
        float: representing the average loss of the epoch.
    """
    model.train()
    total_loss = 0.0
    for imgs, _ in data:
        imgs = imgs.to(device)
        if feature_extractor is not None:
            with torch.no_grad():
                feature_extractor.eval()
                imgs = feature_extractor(imgs)
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
    epochs: int = 500,
    learning_rate: float = 0.01,
    feature_extractor: Optional[nn.Module] = None,
) -> List[float]:
    """Runs the training process for a given number of epochs.

    Args:
        model (nn.Module): The model to train.
        dataset (DataLoader): The data set to train the model.
        epochs (int, optional): Number of epoch to run. Defaults to 500.
        learning_rate (float): The learning rate to use for optimization.
            Defaults to 0.01.
        feature_extractor (Union[nn.Module, None]): Used only for incremental training.

    Returns:
        A list of floats, each element in the list is the average loss per epoch.

    Note: this function save intermediate image reconstruction in your working directory,
            every 10 epochs.
    """
    _device_ = "cuda" if torch.cuda.is_available() else "cpu"

    # we take this images for visualization purposes
    test = _get_test(dataset, _device_)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True
    )
    criterion = nn.MSELoss()
    losses = []
    progress = tqdm.tqdm(range(epochs), desc="Loss: Inf")
    for epoch in progress:
        loss = train_fn(
            model, criterion, optimizer, dataset, _device_, feature_extractor
        )
        is_epoch_divided_by_10 = epoch % 10 == 0
        is_a_proper_model = list(model.parameters())[0].shape[1] == 3
        if is_epoch_divided_by_10 and is_a_proper_model:
            _save_intermediate_reconstructions(model, test, epoch)
        logging.info("Epoch %i | Loss: %f", epoch, loss)
        progress.set_description(f"Loss: {round(loss, 4)}")
        losses.append(loss)
    return losses
