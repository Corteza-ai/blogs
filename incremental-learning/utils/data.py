"""This module loads the 17-category dataset."""

__all__ = ["load_data"]


import logging

import torchvision.transforms as T  # type: ignore[import]
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder  # type: ignore[import]

logging.basicConfig(
    filename="run.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
)


def load_data(root: str = "./17_flowers/train", shuffle: bool = True) -> DataLoader:
    """Loads a dataset in the form of a folder with subfolders.
    Note:
        Images are resized to 224x224 pixels. Pixels are scaled to the [0, 1] range.
    Args:
        root (str): Root directory of the dataset.
        shuffle (bool, optional): Whether to shuffle the data or not.
            Defaults to True.
    Raises:
        FileNotFoundError: An error occurs if the root directory doesn't exist.
    Returns:
        DataLoader: An iterable that retrieves batches of images.
    """
    transforms = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    try:
        dataset = ImageFolder(root=root, transform=transforms)
    except FileNotFoundError as exc:
        logging.error("No such file or directory: %s!", root)
        raise FileNotFoundError(f"Not such file: {root}!") from exc
    else:
        dataset = DataLoader(
            dataset, batch_size=64, num_workers=8, drop_last=True, shuffle=shuffle
        )
        return dataset
