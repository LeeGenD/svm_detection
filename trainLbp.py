#! usr/bin/python
#coding=utf-8
#lbp+svm model training

import cv2
import numpy as np
from sklearn.externals import joblib
from myUtils import globSplit, printList, minInList, maxInList
from myUtils import precisionRecallCurve
from skimage import feature as skif
from sklearn.svm import SVC

POS_PATH = "/Users/LeeGenD/work/code/xs/images/chaoyang/Positive_train/*.jpg"
NEG_PATH = "/Users/LeeGenD/work/code/xs/images/chaoyang/Negative_train/*.jpg"
POS_TEST_PATH = "/Users/LeeGenD/work/code/xs/images/chaoyang/Positive_test/*.jpg"
NEG_TEST_PATH = "/Users/LeeGenD/work/code/xs/images/chaoyang/Negative_test/*.jpg"
CACHE_NAME = "./backup/lbp.svm.joblib.pkl"

def getLbpData(image, hist_size=256, lbp_radius=1, lbp_point=8):
    image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_CUBIC)
    # 使用LBP方法提取图像的纹理特征.
    lbp = skif.local_binary_pattern(image, lbp_point, lbp_radius, 'default')
    # 统计图像的直方图
    max_bins = int(lbp.max() + 1)
    # hist size:256
    hist, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
    return [hist]

def detector():
    trainPosList, _ = globSplit(POS_PATH, 2000)
    trainNegList, _ = globSplit(NEG_PATH, 6000)
    testPosList, _ = globSplit(POS_TEST_PATH, 2000)
    testNegList, _ = globSplit(NEG_TEST_PATH, 2000)

    # 训练集
    traindata, trainlabels = [],[]
    for jpg_file in trainPosList:
        print jpg_file
        traindata.extend(getLbpData(cv2.imread(jpg_file, 0)))
        trainlabels.append(1)

    for jpg_file in trainNegList:
        print jpg_file
        traindata.extend(getLbpData(cv2.imread(jpg_file, 0)))
        trainlabels.append(-1)

    # 测试集
    testdata, testlabels = [],[]
    for jpg_file in testPosList:
        print jpg_file
        testdata.extend(getLbpData(cv2.imread(jpg_file, 0)))
        testlabels.append(1)

    for jpg_file in testNegList:
        print jpg_file
        testdata.extend(getLbpData(cv2.imread(jpg_file, 0)))
        testlabels.append(-1)

    clf = SVC(probability=True, kernel="linear", C=1000)
    clf.fit(np.array(traindata), np.array(trainlabels))
    _ = joblib.dump(clf, CACHE_NAME, compress=9)

    print("Training set score: %f" % clf.score(np.array(traindata), np.array(trainlabels)))
    print("Test set score: %f" % clf.score(np.array(testdata), np.array(testlabels)))
    y_true = np.array(testlabels)
    predictResult = clf.predict_proba(np.array(testdata))
    predictProbaList = []
    for predictItem in predictResult:
        predictProbaList.append(predictItem[1])

    precisionRecallCurve(np.array(testlabels), predictProbaList)

    print "train pos:%d neg:%d" % (len(trainPosList), len(trainNegList))
    print "test pos:%d neg:%d" % (len(testPosList), len(testNegList))


if __name__ == '__main__':
    detector()
