import os
import numpy as np
import pickle
import json

from skimage.feature import hog
from skimage import io
from skimage import transform

from sklearn.ensemble import RandomForestClassifier
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
        for train_index, test_index in kf.split(xTrain):
            x_train, x_test = xTrain[train_index], xTrain[test_index]
            y_train, y_test = yTrain[train_index], yTrain[test_index]
            clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
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
trainPath = os.path.join(basicPath, "Classification\\Data\\Train")
testPath = os.path.join(basicPath, "Classification\\Data\\Test")
hogTrainPath = "..\\task1\\hog\\Train"
hogTestPath = "..\\task1\\hog\\Test"
modelPath = "..\\task1\\model.pickle"
predPath = "..\\..\\data\\Classification\\Data\\pred.json"

# 标志位
extracted = 1
trained = 1

orientations = 9
pixels_per_cell = [8, 8]
cells_per_block = [8, 8]
picSize = np.multiply(pixels_per_cell, cells_per_block)
hogLength = orientations * cells_per_block[0] * cells_per_block[1]
# classList = ['i2', 'i4', 'i5', 'io', 'ip', 'p11', 'p23', 'p26', 'p5', 'pl30',
#              'pl40', 'pl5', 'pl50', 'pl60', 'pl80', 'pn', 'pne', 'po', 'w57']
classList = os.listdir(trainPath)

# 提取hog特征
if not extracted:
    if not os.path.exists(hogTrainPath):
        os.mkdir(hogTrainPath)

    dirs = os.listdir(trainPath)
    for d in dirs:
        classPath = os.path.join(trainPath, d)
        images = os.listdir(classPath)
        Hog = np.zeros((len(images), hogLength))
        cnt = 0
        for image in images:
            imagePath = os.path.join(classPath, image)
            pic = io.imread(imagePath, as_gray=True)
            pic = transform.resize(pic, picSize)
            hogFeature = hog(pic, orientations=orientations,
                             pixels_per_cell=pixels_per_cell,
                             cells_per_block=cells_per_block,
                             block_norm='L2',
                             feature_vector=True)
            Hog[cnt] = hogFeature
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
        yTmp = cnt * np.ones(xTmp.shape[0])
        if xTrain.size == 0:
            xTrain = xTmp
            yTrain = yTmp
        else:
            xTrain = np.append(xTrain, xTmp, axis=0)
            yTrain = np.append(yTrain, yTmp)
        cnt += 1
    print("Number of samples: ", xTrain.shape[0])
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    clf.fit(xTrain, yTrain)
    with open(modelPath, 'wb') as outFile:
        pickle.dump(clf, outFile)
    print("Trained model is saved to " + modelPath)

    CrossValidation(xTrain, yTrain)

else:
    # with open(modelPath, 'rb') as inFile:
    #     clf = pickle.load(inFile)
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
np.save(os.path.join(hogTestPath, "test"), xTest)
print("Number of tests: ", xTest.shape[0])
print("Successfully Extract HOG features to " + hogTestPath)

with open(predPath, 'w') as outFile:
    json.dump(yPred, outFile)
print("Save predictions to " + predPath)


