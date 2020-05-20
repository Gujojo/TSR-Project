import os
import numpy as np
import pickle
import json

from skimage.feature import hog
from skimage import io
from skimage import transform

from sklearn.svm import SVC


basicPath = "..\\..\\data\\"
trainPath = os.path.join(basicPath, "Classification\\DataFewShot\\Train")
testPath = os.path.join(basicPath, "Classification\\DataFewShot\\Test")
hogTrainPath = "..\\task3\\hog\\Train"
hogTestPath = "..\\task3\\hog\\Test"
modelPath = "..\\task3\\model.pickle"
predPath = "..\\..\\data\\Classification\\DataFewShot\\pred.json"

# 标志位
extracted = 0
trained = 0

orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (8, 8)
picSize = (64, 64)
hogLength = orientations * cells_per_block[0] * cells_per_block[1]
classList = os.listdir(trainPath)

# 提取hog特征
if not extracted:
    if not os.path.exists(hogTrainPath):
        os.mkdir(hogTrainPath)
    dirs = os.listdir(trainPath)
    index = np.zeros(len(dirs))
    for d in dirs:
        classPath = os.path.join(trainPath, d)
        images = os.listdir(classPath)
        image = images[0]
        imagePath = os.path.join(classPath, image)
        pic = io.imread(imagePath, as_gray=True)
        pic = transform.resize(pic, picSize)
        hogFeature = hog(pic, orientations=orientations,
                         pixels_per_cell=pixels_per_cell,
                         cells_per_block=cells_per_block,
                         block_norm='L2',
                         feature_vector=True)
        Hog = hogFeature
        featurePath = os.path.join(hogTrainPath, d)
        np.save(featurePath, Hog)
    print("Successfully Extract HOG features to " + hogTrainPath)

# 读取hog特征并开始训练
if not trained:
    xTrain = np.array([])
    yTrain = np.array([])
    files = os.listdir(hogTrainPath)
    cnt = 0
    for file in files:
        filePath = os.path.join(hogTrainPath, file)
        xTmp = np.load(filePath)
        xTmp = xTmp[np.newaxis, :]
        yTmp = cnt
        if xTrain.size == 0:
            xTrain = xTmp
            yTrain = yTmp
        else:
            xTrain = np.append(xTrain, xTmp, axis=0)
            yTrain = np.append(yTrain, yTmp)
        cnt += 1
    print("Number of samples: ", xTrain.shape[0])
    clf = SVC()
    clf.fit(xTrain, yTrain)
    with open(modelPath, 'wb') as outFile:
        pickle.dump(clf, outFile)
    print("Trained model is saved to " + modelPath)
else:
    clf = np.load(modelPath, allow_pickle=True)
    print("Successfully load model from " + modelPath)

# 进行测试并输出结果
images = os.listdir(testPath)
xTest = np.zeros((len(images), hogLength))
yPred = {}
cnt = 0
for image in images:
    imagePath = os.path.join(testPath, image)
    pic = io.imread(imagePath, as_gray=True)
    pic = transform.resize(pic, picSize)
    hogFeature = hog(pic, orientations=orientations,
                     pixels_per_cell=pixels_per_cell,
                     cells_per_block=cells_per_block,
                     block_norm='L2',
                     feature_vector=True)
    xTest[cnt] = hogFeature
    tmp = int(clf.predict(hogFeature.reshape(1, -1)))
    yPred[image] = classList[tmp]
    cnt += 1

if not os.path.exists(hogTestPath):
    os.mkdir(hogTestPath)
np.save(os.path.join(hogTestPath, "test"), xTest)

print("Number of tests: ", xTest.shape[0])
print("Successfully Extract HOG features to " + hogTestPath)

with open(predPath, 'w') as outFile:
    json.dump(yPred, outFile)
print("Save predictions to " + predPath)

