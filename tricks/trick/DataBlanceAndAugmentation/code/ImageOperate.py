import os
import cv2
import albumentations as A

class Operate(object):
    """Summary of class here.

    This class is mainly used for image processing and contains a variety of methods.

    Attributes:
        path: Folders that need to be processed.
        operate_index: Method to be executed.
        is_balance: Whether to use the balancing method.
        blance_operate: A dictionary containing data balancing methods.
        augmentation_operate: A dictionary containing data augmenting methods.
        data_blance_operate: List of short names for the balancing method.
        augmentation_operate: List of short names for the augmenting method.
    """
    def __init__(self, path, operate_index, is_balance=True):
        """Init class.
        The main part of the initialization function is to generate a dictionary
        for data balancing and data augmentation methods.

        Args:
            path: Original path.
            operate_index: Method to be executed.
            is_balance: Whether to use the balancing method.
        """
        self.path = path
        self.operate_index = operate_index
        self.is_balance = is_balance
        self.blance_operate = {
            0: A.RandomRotate90(always_apply=True),
            1: A.HorizontalFlip(always_apply=True),
            2: A.Sharpen(always_apply=True),
            3: A.GaussNoise(var_limit=(100, 100), always_apply=True),
            4: A.RandomSizedCrop([90, 90], 128, 128, always_apply=True),
            5: A.Blur(always_apply=True)
        }
        self.augmentation_operate = {
            0: A.CoarseDropout(max_holes=8, max_width=16, max_height=16,min_holes=8,
                               min_width=8, min_height=8, fill_value=0, always_apply=True),
            1: A.RandomGridShuffle(grid=(4, 4), always_apply=True),
            2: A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5,
                             hue=0.5, always_apply=True),
            3: A.RandomShadow(always_apply=True)
        }
        self.data_blance_operate = ['RR90', 'HF', 'S', 'GN', 'RSC', 'B']
        self.data_augmentation_operate = ['CP', 'RGS', 'CJ', 'RS']

    def deal_image(self):
        images_path = [os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith('.jpg') or f.endswith('.png')]
        for image_path in images_path:
            for index in self.operate_index:
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.is_balance:
                    if index == 0:
                        img = self.blance_operate[0].apply(img=img, factor=1)
                    else:
                        img = self.blance_operate[index](image=img)['image']
                else:
                    img = self.augmentation_operate[index](image=img)['image']
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if self.is_balance:
                    cv2.imwrite(f"{os.path.splitext(image_path)[0]}{self.data_blance_operate[index]}.jpg", img)
                else:
                    cv2.imwrite(f"{os.path.splitext(image_path)[0]}_{self.data_augmentation_operate[index]}.jpg", img)







