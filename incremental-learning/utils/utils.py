"""Utilities"""

__all__ = ["freeze", "unfreeze", "list_flattened", "save_model"]

import os
from typing import Any, List, Union

import torch
from torch import nn


def freeze(model: Union[nn.Module, None]) -> None:
    """
    Sets the requires_grad attribute of parameters to False.
    """
    if model is not None:
        for params in model.parameters():
            params.requires_grad = False


def list_flattened(to_flatten: List[List[Any]]) -> List[Any]:
    """Flattened a list of lists.

    Args:
        to_flatten (List[List[Any]]): A list of lists.

    Returns:
        The flattened list.
    """
    return sum(to_flatten, [])


def save_model(model: nn.Module, outfile: str) -> None:
    """Saves a pytorch model to a file.

    Saves a pytorch model, it creates a directory called
    saved-models (if it doesn't exists) and save the model
    whithin that falder in a file called "outfile".

    Args:
        model (nn.Module): The model to save.
        outfile (str): name of the file.
    """
    _main_dir_ = "./pre-trained-models"
    if not os.path.exists(_main_dir_):
        os.mkdir(_main_dir_)
    file_name = os.path.join(_main_dir_, outfile)
    torch.save(model, file_name)


def unfreeze(model: Union[nn.Module, None]) -> None:
    """
    Sets the requires_grad attribute of parameters to True.
    """
    if model is not None:
        for params in model.parameters():
            params.requires_grad = True
