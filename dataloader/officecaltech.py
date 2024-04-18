"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os
from typing import Optional
from .imagelist import ImageList


class OfficeCaltech(ImageList):
    """Office+Caltech Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'A'``: amazon, \
            ``'D'``: dslr, ``'W'``:webcam and ``'C'``: caltech.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            amazon/
                images/
                    backpack/
                        *.jpg
                        ...
            dslr/
            webcam/
            caltech/
            image_list/
                amazon.txt
                dslr.txt
                webcam.txt
                caltech.txt
    """
    # directories = {
    #     "A": "amazon",
    #     "D": "dslr",
    #     "W": "webcam",
    #     "C": "caltech"
    # }
    CLASSES = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard',
               'laptop_computer', 'monitor', 'mouse', 'mug', 'projector']

    def __init__(self, root: str, task: str, download: Optional[bool] = False, **kwargs):

        data_list_file = os.path.join(root, "_image_list", task + "_list.txt")
        super(OfficeCaltech, self).__init__(root, OfficeCaltech.CLASSES, data_list_file=data_list_file, **kwargs)

    @property
    def num_classes(self):
        """Number of classes"""
        return len(self.classes)

    @classmethod
    def domains(cls):
        return list(["amazon", "caltech", "dslr", "webcam"])