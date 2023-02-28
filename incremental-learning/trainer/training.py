"""Main functions to train a pytorch model."""

__all__ = ["main_train"]

import argparse
import copy
import logging
import random
import time
from typing import List, Union

import numpy as np
import torch
from torch import nn

from model import AutoEncoder, Decoder, Encoder
from trainer.core import train
from utils import (
    freeze,
    list_flattened,
    load_data,
    plot_loss_history,
    save_model,
    unfreeze,
)

# We set the seed for reproducibility purposes
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def end_to_end_training(epochs: int, learning_rate: float):
    """Trains a model from scratch end-to-end.

    Args:
        epochs (int): Number of epochs to train the model.
        lr (float): The learning rate to use for optimization.

    Note: This function saves a trained model into your working directory.
    """
    _device_ = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoEncoder(Encoder(), Decoder())
    dataset = load_data()
    model.to(_device_)
    start = time.time()
    logging.info("Training process starts...")
    losses = train(model, dataset, epochs, learning_rate)
    logging.info("Elapsed time: %s", time.time() - start)
    logging.info("Saving the model...")
    save_model(model, "end_to_end.pt")
    return losses


def incremental_training(
    epochs_per_layer: int, last_layer_epoch: int, learning_rate: float
) -> List[List[float]]:
    """Trains a model in an incremental approach proposed in: 10.1109/IJCNN.2018.8489123.

    Args:
        epochs_per_layer (int): Number of epochs to train at each step.
        last_layer_epochs (int): Number of epochs to train the last step.
        lr (float): The learning rate to use for optimization.

    Note: This function saves a trained model into your working directory.
    """
    _device_ = "cuda" if torch.cuda.is_available() else "cpu"
    encoders: Union[List, nn.Module] = Encoder()
    decoders: Union[List, nn.Module] = Decoder()
    encoders = [encoders.__dict__["_modules"][f"layer{i}"] for i in range(1, 6)]
    decoders = [decoders.__dict__["_modules"][f"layer{i}"] for i in range(1, 6)][::-1]
    encoder_stack: Union[nn.Module, None] = None
    decoder_stack: Union[nn.Module, None] = None
    losses = []
    dataset = load_data()
    start = time.time()
    logging.info("Training process starts...")
    for i, encoder_ in enumerate(encoders):
        freeze(encoder_stack)
        glw_trained_model = AutoEncoder(encoder_, decoders[i])
        glw_trained_model.to(_device_)
        loss = train(
            glw_trained_model,
            dataset,
            epochs=epochs_per_layer,
            learning_rate=learning_rate,
            feature_extractor=encoder_stack,
        )
        unfreeze(encoder_stack)
        losses.append(loss)
        if i > 0:
            if i == len(encoders) - 1:
                epochs_per_layer = last_layer_epoch
            if encoder_stack is not None and decoder_stack is not None:
                encoder_stack.add_module(f"layer{i+1}", glw_trained_model.encoder)
                decoder_stack = nn.Sequential(
                    glw_trained_model.decoder, *list(decoder_stack.children())
                )
                glw_trained_model = AutoEncoder(encoder_stack, decoder_stack)
                loss = train(
                    glw_trained_model,
                    dataset,
                    epochs=epochs_per_layer,
                    learning_rate=learning_rate,
                )
            losses.append(loss)
        else:
            encoder_stack = copy.deepcopy(glw_trained_model.encoder)
            decoder_stack = copy.deepcopy(glw_trained_model.decoder)
    logging.info("Elapsed time: %s", time.time() - start)
    logging.info("Saving the model...")
    save_model(glw_trained_model, "glw_trained_model.pt")
    return losses


def blog_experiments():
    """Train the model under two approaches: end-to-end and incremental.

    It use predefined hype-parameters according to the post:
    https://corteza.ai/news-blog-incremental-learning/

    Returns:
        two lists of floats with the losses over the training.
    """
    logging.info("Training with end-to-end approach starts... ")
    ete_loss = end_to_end_training(500, 0.005)
    logging.info("Training with incremental approach starts...")
    glw_losses = incremental_training(20, 50, 0.005)
    logging.info("Saving the loss history...")
    glw_losses = list_flattened(glw_losses)
    plot_loss_history(ete_loss, "End-to-End", glw_losses, "incremental training")
    logging.info("All done!")


def main_train(args: argparse.Namespace) -> None:
    """Selects the approach to train the model.

    Args:
        args (argparse.Namespace): Hyper-parameters of the model.

    Raises:
        An error ocurrs when the approach to train is not in
        [incremental, end-to-end, blog-experiments].
    """
    if args.mode == "end-to-end":
        _ = end_to_end_training(args.epochs, args.lr)
    elif args.mode == "incremental":
        _ = incremental_training(args.epochs, args.last_num_epochs, args.lr)
    elif args.mode == "blog-experiments":
        blog_experiments()
    else:
        logging.error("The %s training is not implemented", args.mode)
        raise NotImplementedError(f"{args.mode} training approach is not supported!")
