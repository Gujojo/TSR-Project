import os
import shutil
import json
import numpy as np
from skimage import io, transform, img_as_ubyte


basicPath = "..\\..\\data\\"
trainPath = "..\\..\\data\\Classification\\Data\\Train"
newTrainPath = "..\\task3\\Train"
newTestPath = "..\\task3\\Test"

picSize = [100, 100]
maxNum = 100

if os.path.exists(newTrainPath):
    shutil.rmtree(newTrainPath)
os.mkdir(newTrainPath)
if os.path.exists(newTestPath):
    shutil.rmtree(newTestPath)
os.mkdir(newTestPath)

dirs = os.listdir(trainPath)
classList = dirs
trainDict = {}
testDict = {}
for n in range(len(dirs)):
    d = dirs[n]
    trainDir = os.path.join(newTrainPath, d)
    if not os.path.exists(trainDir):
        os.mkdir(trainDir)
    testDir = os.path.join(newTestPath, d)
    if not os.path.exists(testDir):
        os.mkdir(testDir)
    classPath = os.path.join(trainPath, d)
    images = os.listdir(classPath)
    index = np.random.randint(0, maxNum)
    for i in range(maxNum+1):
        image = images[i]
        readPath = os.path.join(classPath, image)
        pic = io.imread(readPath, as_gray=True)
        pic = transform.resize(pic, picSize)
        pic = pic / np.max(pic)
        if i == index:  # 训练文件
            writePath = os.path.join(trainDir, image)
            io.imsave(writePath, img_as_ubyte(pic))
            trainDict[writePath] = n
        else:  # 测试文件
            writePath = os.path.join(testDir, image)
            io.imsave(writePath, img_as_ubyte(pic))
            testDict[writePath] = n

with open(os.path.join(newTrainPath, "train.json"), 'w') as outFile:
    json.dump(trainDict, outFile)
print("Successfully generate train images to " + newTrainPath, len(trainDict))
with open(os.path.join(newTestPath, "test.json"), 'w') as outFile:
    json.dump(testDict, outFile)
print("Successfully generate test images to " + newTrainPath, len(testDict))


