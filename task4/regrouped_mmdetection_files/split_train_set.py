import os
from random import randint
from shutil import move
from sys import argv


def split_train_set(origin_path, val_path, k=15):
    files = os.listdir(origin_path)
    train_num = len(files)
    split_num = int(train_num/k)
    for i in range(split_num):
        index = randint(0, train_num - 1 - i)
        file = files[index]
        files.remove(file)
        move(origin_path + file, val_path)
    print("Successfully constructed val_set.")


if __name__ == '__main__':
    if len(argv) == 1:
        split_train_set('./Detection/train/', './Detection/val/')
    elif argv[1] == '1':
        split_train_set('./Detection/val/', './Detection/train/', 1)
    else:
        split_train_set(argv[1], argv[2], int(argv[3]))
