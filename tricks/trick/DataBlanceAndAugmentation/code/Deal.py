import os
import matplotlib.pyplot as plt
import argparse
import shutil
import random
from decimal import Decimal, ROUND_HALF_UP
from ImageOperate import Operate

"""Please organize the data set into the following format:

Dataset
    |-class_1
    |-class_2
    ---
    |-class_n
"""

def GetFiles(path):
    """Get the file name under the folder.

    Args:
        path: Folder path.

    Return:
        a list: File path list.
    """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]
·

def is_need_data_blance(classes_num, conservative=True):
    """Whether dataset balancing is required

    Determine whether the dataset needs to be balanced,
    and calculate the balance factor of each category.

    Args:
        classes_num: Quantity of each category.
        conservative: The balance factor is calculated by count down or half up. The default value is True, which means count down.

    Returns:
        False: No data balancing.
        balance_factors: Balance factor of each category.

    Raises:
        BlanceError1: The number of categories is balanced without data balance.
        BlanceError2: The number of categories varies too much. Please modify the code to add a balance method.
    """
    num_max = max(classes_num)
    print(num_max)
    print(len(classes_num))
    balance_factors = []
    for i in range(len(classes_num)):
        if(conservative):
            balance_factors.append(num_max // classes_num[i])
        else:
            balance_factors.append(
               int(
                   Decimal(num_max / classes_num[i]).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
               )
            )
    if max(balance_factors) <= 1:
        print("BlanceError1: This Dataset don't need Datablance.")
        return False
    else:
        if max(balance_factors) > 6:
            print("BlanceError2: Data balance is beyond the range of balance.")
            return False
        else:
            return balance_factors

def CopyImage(origin_path, new_path):
    """Copy files to the specified folder.

    Args:
        origin_path: Original Folder.
        new_path: Destination Folder.

    Raises:
        PathError: The original path not exists.
    """
    if os.path.isdir(new_path):
        shutil.rmtree(new_path)
    if os.path.exists(origin_path):
        shutil.copytree(origin_path, new_path)
    else:
        print("PathError: The original path not exists.")


def DataBalanceOrAugmentation(origin_path, blance_factors, classes, new_path,
                              augmentation_level=0, is_blance=True):
    """Augmentation or balance dataset.

    Args:
        origin_path: Original path.
        blance_factors: A list contains balance factors for all categories.
        classes: Category List,like['class_1', 'class_2',...,'class_n'].
        new_path: Destination Folder.
        augmentation_level: Strength of data augmentation.
        is_blance: Whether to balance dataset.

    Raises:
        AugmentationError: More than supported methods,please add a new method.
    """
    CopyImage(origin_path, new_path)
    for index in range(len(classes)):
        if blance_factors[index] == 1 and is_blance==True:
            continue
        process_path = os.path.join(new_path, classes[index])
        if is_blance:
            operate_index = random.sample(range(0, 6), blance_factors[index] - 1)
        else:
            if augmentation_level > 4:
                print("AugmentationError: More than supported methods.")
                return
            operate_index = random.sample(range(0, 4), augmentation_level)
        o = Operate(path=process_path, operate_index=operate_index, is_balance=is_blance)
        o.deal_image()

def CountDataset(src_path, image_name="CountDataset.png", is_need_return=True):
    """Count the number of each category in the dataset.

    Count the number of each category in the dataset,
    and save the results as png format images.

    Args:
        src_path: Home directory of the dataset.

    Returns:
        classes: Category List,like['class_1', 'class_2',...,'class_n']
        classes_num: Quantity of each category.

    Raises:
        PathError: Input error, not a directory.
    """
    if not os.path.isdir(src_path):
        print("PathError: This is not a directory, please check.")
        return
    classes_num = []
    classes = os.listdir(src_path)
    for i in classes:
        path = os.path.join(src_path, i)
        classes_num.append(len(GetFiles(path)))

    plt.xlabel("Num")
    plt.ylabel("Classes")
    plt.title("Dataset statistics")
    barh = plt.barh(classes, classes_num)
    plt.bar_label(barh, label_type='edge')
    plt.tight_layout()
    plt.savefig(image_name)
    if is_need_return:
        return classes, classes_num
    else:
        return

def DataDivide(src_path, new_path, train_test_factor, train_val_factor):
    """Divide the dataset into train set,test set and val set.

    Args:
        src_path: Original path.
        new_path: Destination Folder.
        train_test_factor: Scale of train set to test set,
        train_val_factor: Scale of train set to val set.
    """
    if not os.path.isdir(new_path):
        os.mkdir(new_path)

    for dirct in ['train', 'test', 'val']:
        os.mkdir(os.path.join(new_path, dirct))

    classes = os.listdir(src_path)
    for i in classes:
        img_list = GetFiles(os.path.join(src_path, i))
        move_path_train = os.path.join(new_path, 'train', i)
        move_path_test = os.path.join(new_path, 'test', i)
        move_path_val = os.path.join(new_path, 'val', i)

        if not os.path.isdir(move_path_test):
            os.mkdir(move_path_test)
        if not os.path.isdir(move_path_val):
            os.mkdir(move_path_val)

        random.shuffle(img_list)
        length_train = int(len(img_list) * train_test_factor)
        for img_test in img_list[length_train:]:
            shutil.move(img_test, move_path_test)
        shutil.move(os.path.join(src_path, i), os.path.join(new_path, 'train'))

        img_train_list = GetFiles(move_path_train)
        random.shuffle(img_train_list)
        length_val = int(len(img_train_list) * train_val_factor)
        for img_val in img_train_list[length_val:]:
            shutil.move(img_val, move_path_val)

parser = argparse.ArgumentParser()
parser.add_argument("--src_path", type=str, default=None, help="Dataset directory to be processed.")
parser.add_argument("--conservative", type=bool, default=False, help="The balance factor is countdown or half adjust.")
parser.add_argument("--augmentation_level", type=int, default=0, help="The strength of data augmentation.")
parser.add_argument("--blance_path", type=str, default='DataBlance', help="Set balanced dataset path.")
parser.add_argument("--is_augmentation", type=bool, default=True, help="Whether to augmentation dataset.")
parser.add_argument("--augmentation_path", type=str, default='DataAugmentation', help="Set augmented dataset path.")
parser.add_argument("--is_partition", type=bool, default=True, help="Is the dataset partitioned.")
parser.add_argument("--train_test_factor", type=float, default=0.9, help="Set the ratio of train dataset to test dataset.")
parser.add_argument("--train_val_factor", type=float, default=0.8, help="Set the ratio of train dataset set to validate dataset.")
parser.add_argument("--partition_path", type=str, default='DataPartition', help="Set up a directory to store divided data")
args = parser.parse_args()

classes, classes_num = CountDataset(src_path=args.src_path)
balance_factors = is_need_data_blance(classes_num=classes_num, conservative=args.conservative)
if balance_factors == False:
    augmentation_path = args.src_path
else:
    augmentation_path = args.blance_path
    DataBalanceOrAugmentation(origin_path=args.src_path,
                              blance_factors=balance_factors,
                              classes=classes,
                              new_path=args.blance_path)
    CountDataset(src_path=args.blance_path, image_name="DatasetBlance.png", is_need_return=False)

if args.is_augmentation:
    DataBalanceOrAugmentation(origin_path=augmentation_path,
                              blance_factors=balance_factors,
                              classes=classes,
                              new_path=args.augmentation_path,
                              augmentation_level=args.augmentation_level,
                              is_blance=False)
    CountDataset(src_path=args.augmentation_path, image_name="DatasetAugmentation.png", is_need_return=False)

if balance_factors == False and args.is_augmentation == False:
    origin_path = args.src_path
if balance_factors != False and args.is_augmentation == False:
    origin_path = args.blance_path
if balance_factors != False and args.is_augmentation == True:
    origin_path = args.augmentation_path

if args.is_partition:
    DataDivide(src_path=origin_path,
               new_path=args.partition_path,
               train_test_factor=args.train_test_factor,
               train_val_factor=args.train_val_factor)
