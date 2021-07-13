import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class AeroscapesDataset(CustomDataset):
    """Aeroscapes dataset.

    In segmentation map annotation for AeroscapesDataset, 0 stands for background,
    which is included in 60 categories. ``reduce_zero_label`` is fixed to
    False. The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png'.

    Args:
        split (str): Split txt file for PascalContext.
    """

    CLASSES = ('background',
                'person',
                'bike',
                'car',
                'drone',
                'boat',
                'animal',
                'obstacle',
                'construction',
                'vegetation',
                'road',
                'sky')

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7]]

    def __init__(self, split, **kwargs):
        super(AeroscapesDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            split=split,
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
