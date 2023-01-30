"""Tool for loading an image folder data set.

The folder should be organized as follows:

|-- root
|--|--class_1
|--|--|-- image1.png
|--|--|_ ...
|--|--class_2
|--|--|-- image.png
|--|--|_ ...
|--|_ ...
|_
"""


import logging
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder  # type: ignore[import]
import torchvision.transforms as T  # type: ignore[import]


def load_data(root: str, shuffle: bool = True) -> DataLoader:
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
    except FileNotFoundError:
        logging.error("No such file or directory: %s!", root)
    dataset = DataLoader(
        dataset, batch_size=32, num_workers=8, drop_last=True, shuffle=shuffle
    )
    return dataset
