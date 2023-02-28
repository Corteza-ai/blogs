"""Use this file to save a figure with a random sample of the images
in the 17-category data set.
"""

import logging

from utils import load_data, plot_multiple_images

if __name__ == "__main__":
    logging.info("Loading data...")
    dataset = iter(load_data())
    logging.info("Saving the image: sample_image.png..")
    plot_multiple_images(
        next(dataset)[0][:20].numpy().transpose(0, 2, 3, 1), "sample_images.png", 4
    )
    logging.info("All done!")
