import os
import shutil
import json
import numpy as np
from skimage import io, transform, img_as_ubyte


basicPath = "..\\..\\data\\"
trainPath = os.path.join(basicPath, "Classification\\DataFewShot\\Train")
testPath = os.path.join(basicPath, "Classification\\DataFewShot\\Test")
labelPath = os.path.join(basicPath, "Classification\\DataFewShot\\test.json")
newTrainPath = "..\\task3\\Train"
newTestPath = "..\\task3\\Test\\test"

picSize = [64, 64]

if os.path.exists(newTrainPath):
    shutil.rmtree(newTrainPath)
os.mkdir(newTrainPath)
if os.path.exists(newTestPath):
    shutil.rmtree(newTestPath)
os.mkdir(newTestPath)

dirs = os.listdir(trainPath)
classList = dirs
trainDict = {}
for n in range(len(dirs)):
    d = dirs[n]
    trainDir = os.path.join(newTrainPath, d)
    if not os.path.exists(trainDir):
        os.mkdir(trainDir)
    classPath = os.path.join(trainPath, d)
    images = os.listdir(classPath)
    image = images[0]
    readPath = os.path.join(classPath, image)
    pic = io.imread(readPath, as_gray=True)
    pic = transform.resize(pic, picSize)
    pic = pic / np.max(pic)
    writePath = os.path.join(trainDir, image)
    io.imsave(writePath, img_as_ubyte(pic))
    trainDict[writePath] = d

with open(os.path.join(newTrainPath, "train.json"), 'w') as outFile:
    json.dump(trainDict, outFile)
print("Successfully generate train images to " + newTrainPath, len(trainDict))

testDict = {}
with open(labelPath, 'r') as inFile:
    labelDict = json.load(inFile)
images = os.listdir(testPath)
for i in range(len(images)):
    image = images[i]
    readPath = os.path.join(testPath, image)
    pic = io.imread(readPath, as_gray=True)
    pic = transform.resize(pic, picSize)
    pic = pic / np.max(pic)
    writePath = os.path.join(newTestPath, image)
    io.imsave(writePath, img_as_ubyte(pic))
    testDict[writePath] = labelDict[image]

with open(os.path.join(newTestPath, "test.json"), 'w') as outFile:
    json.dump(testDict, outFile)
print("Successfully generate test images to " + newTestPath, len(testDict))


