import os
import numpy as np
import pickle
import json

from skimage.feature import hog
from skimage import io
from skimage import transform

from sklearn.svm import SVC
from sklearn.model_selection import KFold


def CrossValidation(xTrain, yTrain):
    times = 1
    nSplit = 10
    TrainAcc = 0
    TestAcc = 0
    for t in range(times):
        state = np.random.get_state()
        np.random.shuffle(xTrain)
        np.random.set_state(state)
        np.random.shuffle(yTrain)
        kf = KFold(n_splits=nSplit)
        cnt = 0
        for train_index, test_index in kf.split(xTrain):
            x_train, x_test = xTrain[train_index], xTrain[test_index]
            y_train, y_test = yTrain[train_index], yTrain[test_index]
            clf = SVC()
            clf.fit(x_train, y_train)

            yPredict = clf.predict(x_train)
            acc = (yPredict == y_train).sum() / y_train.size
            TrainAcc += acc

            yPredict = clf.predict(x_test)
            acc = (yPredict == y_test).sum() / y_test.size
            TestAcc += acc
            print(acc)

    TrainAcc /= (nSplit * times)
    TestAcc /= (nSplit * times)
    print('Train acc is {:.4f}'.format(TrainAcc))
    print('Test acc is {:.4f}'.format(TestAcc))


basicPath = "..\\..\\data\\"
trainPath = "..\\..\\data\\Classification\\Data\\Train"
# testPath = os.path.join(basicPath, "Classification\\DataFewShot\\Test")
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
hogLength = orientations * cells_per_block[0] * cells_per_block[1]
classList = ['i2', 'i4', 'i5', 'io', 'ip', 'p11', 'p23', 'p26', 'p5', 'pl30',
             'pl40', 'pl5', 'pl50', 'pl60', 'pl80', 'pn', 'pne', 'po', 'w57']

# 提取hog特征
if not extracted:
    if not os.path.exists(hogTrainPath):
        os.mkdir(hogTrainPath)

    dirs = os.listdir(trainPath)
    for d in dirs:
        classPath = os.path.join(trainPath, d)
        images = os.listdir(classPath)
        cnt = 0
        image = images[0]
        imagePath = os.path.join(classPath, image)
        pic = io.imread(imagePath, as_gray=True)
        pic = transform.resize(pic, (64, 64))
        hogFeature = hog(pic, orientations=orientations,
                         pixels_per_cell=pixels_per_cell,
                         cells_per_block=cells_per_block,
                         block_norm='L2',
                         feature_vector=True)
        Hog = hogFeature
        cnt += 1
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
    with open(modelPath, 'rb') as inFile:
        clf = pickle.load(inFile)
    clf = np.load(modelPath, allow_pickle=True)
    print("Successfully load model from " + modelPath)

# 进行测试并输出结果
if not os.path.exists(hogTestPath):
    os.mkdir(hogTestPath)
dirs = os.listdir(trainPath)
count = 0
xTest = np.array([])
yTest = np.array([])
yPred = np.array([])
for d in dirs:
    classPath = os.path.join(trainPath, d)
    images = os.listdir(classPath)
    cnt = 0
    for image in images:
        if cnt != 0:
            imagePath = os.path.join(classPath, image)
            pic = io.imread(imagePath, as_gray=True)
            pic = transform.resize(pic, (64, 64))
            hogFeature = hog(pic, orientations=orientations,
                             pixels_per_cell=pixels_per_cell,
                             cells_per_block=cells_per_block,
                             block_norm='L2',
                             feature_vector=True)
            tmp = int(clf.predict(hogFeature.reshape(1, -1)))
            if xTest.size == 0:
                xTest = hogFeature
                yTest = count
                yPred = tmp
            else:
                xTest = np.append(xTest, hogFeature, axis=0)
                yTest = np.append(yTest, count)
                yPred = np.append(yPred, tmp)
        cnt += 1
    count += 1
np.save(os.path.join(hogTestPath, "test"), xTest)

print("Number of tests: ", xTest.shape[0])
print("Successfully Extract HOG features to " + hogTestPath)

# with open(predPath, 'w') as outFile:
#     json.dump(yPred, outFile)
# print("Save predictions to " + predPath)

count = (yTest == yPred)
acc = np.sum(count) / count.size
print("accuracy: ", acc)

# CrossValidation(xTrain, yTrain)
