"""
Main functions to train a pytorch model.
"""

import logging
import random
import numpy as np
import torch
from model.autoencoder import AutoEncoder, Encoder, Decoder
from train.core import train
from utils.data import load_data
from utils.visualizations import visualize_intermediate_reconstruction

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def main_train(data_path: str, epochs: int, learning_rate: float):
    """Trains a pytorch model of an AE.

    Args:
        data_path (str): Root directory to the data set to train the model.
        epochs (int): Number of epochs.
        lr (float): Learning rate.
    """
    encoder = Encoder()
    decoder = Decoder()
    if torch.cuda.is_available():
        _device_ = "cuda"
        model = AutoEncoder(encoder, decoder)
    else:
        _device_ = "cpu"
        model = AutoEncoder(encoder.layer1, decoder.layer5)
    dataset = load_data(root=data_path)
    test = iter(dataset)
    test_ = next(test)[0][:5]
    test_ = test_.to(_device_)
    model.to(_device_)
    logging.info("Starting training...")
    _, recons = train(model, dataset, test_, epochs, learning_rate)
    logging.info("Training done!")
    logging.info("Saving the reconstruction history plot...")
    rows = (epochs // 50) + 2
    visualize_intermediate_reconstruction(recons, rows)
    logging.info("Done!")
    logging.info("saving the trained model to ./trained_autoencoder.pt...")
    torch.save(model, "./trained_autoencoder.pt")
    logging.info("Done!")
