"""
Use this file to test auto-encoders and compare their performance.

Example:
    $> python comparison.py

the above line runs the module and saves in your working directory a matplotlib figure.
The images at the top of the figure are the originals.
The images in the middle are the reconstruction of the end-to-end trained model.
The images at the bottom are the reconstructions of the GLW-trained model.

It is expected that the models are in the pre-trained-models directory.
"""

import os

import numpy as np
import torch

from utils import load_data, plot_multiple_images

if __name__ == "__main__":
    _DEVICE_ = "cuda" if torch.cuda.is_available() else "cpu"
    current = os.getcwd()
    dataset = iter(
        load_data(os.path.join(current, "17_flowers/validation"), shuffle=True)
    )

    end_to_end_model = torch.load(
        os.path.join(current, "pre-trained-models/end_to_end.pt"), map_location=_DEVICE_
    )

    glw_model = torch.load(
        os.path.join(current, "pre-trained-models/glw_trained_model.pt"),
        map_location=_DEVICE_,
    )

    with torch.no_grad():
        end_to_end_model.eval()
        glw_model.eval()
        images = next(dataset)[0]
        images = images.to(_DEVICE_)

        ete_results = end_to_end_model(images)
        glw_results = glw_model(images)

    results = np.vstack(
        [
            images.cpu().numpy()[:5],
            ete_results.cpu().numpy()[:5],
            glw_results.cpu().numpy()[:5],
        ]
    )

    plot_multiple_images(results.transpose(0, 2, 3, 1), "./comparison.png", 3)
