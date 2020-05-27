import os
from random import randint
import shutil
from shutil import copy


def split_train_set(origin_path, train_path, val_path):
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    os.mkdir(train_path)
    if os.path.exists(val_path):
        shutil.rmtree(val_path)
    os.mkdir(val_path)
    files = os.listdir(origin_path)
    train_num = len(files)
    split_num = int(0.1 * train_num)
    for i in range(split_num):
        index = randint(0, train_num - 1 - i)
        file = files[index]
        files.remove(file)
        copy(origin_path + file, val_path)
    print("Successfully constructed val_set.")
    for i in range(train_num - split_num):
        file = files[i]
        copy(origin_path + file, train_path)
    print("Successfully constructed train_set.")


if __name__ == '__main__':
    split_train_set('./Detection/train_origin/', './Detection/train/', './Detection/val/')
