#! usr/bin/python
#coding=utf-8
#sift+svm model training

import cv2
import numpy as np
from sklearn.externals import joblib
from myUtils import globSplit, printList, minInList, maxInList
from myUtils import precisionRecallCurve
from skimage import feature as skif
from sklearn.svm import SVC

POS_PATH = "./images/Positive_train/*.jpg"
NEG_PATH = "./images/Negative_train/*.jpg"
POS_TEST_PATH = "./images/Positive_test/*.jpg"
NEG_TEST_PATH = "//images/Negative_test/*.jpg"
CACHE_NAME = "./backup/sift.svm.joblib.pkl"
SPLIT_NUMBER = 6000
#cv2.IMREAD_GRAYSCALE 以灰度模式读入图像
READ_TYPE = cv2.IMREAD_COLOR #读入一副彩色图像。图像的透明度会被忽略， 
NONE_TYPE = type(None)# fix some error case



def get_flann_matcher():
  flann_params = dict(algorithm = 1, trees = 5)
  return cv2.FlannBasedMatcher(flann_params, {})

def get_bow_extractor(extract, match):
  """
  获取bow提取器
  """
  return cv2.BOWImgDescriptorExtractor(extract, match)

def get_extract_detect():
  """
  返回两个sift特征提取器
  """
  return cv2.xfeatures2d.SIFT_create(), cv2.xfeatures2d.SIFT_create()

def extract_sift(fn, extractor, detector):
  """
  提取sift特征
  """
  im = cv2.imread(fn,READ_TYPE)
  return extractor.compute(im, detector.detect(im))[1]
    
def bow_features(img, extractor_bow, detector):
  """
  提取bow特征
  """
  return extractor_bow.compute(img, detector.detect(img))

def detector():
  detect, extract = get_extract_detect()
  matcher = get_flann_matcher()
  extract_bow = get_bow_extractor(extract, matcher)
  print "building BOWKMeansTrainer..."
  bow_kmeans_trainer = cv2.BOWKMeansTrainer(36)
  trainPosList, _ = globSplit(POS_PATH, SPLIT_NUMBER)
  trainNegList, _ = globSplit(NEG_PATH, SPLIT_NUMBER)
  testPosList, _ = globSplit(POS_TEST_PATH, SPLIT_NUMBER)
  testNegList, _ = globSplit(NEG_TEST_PATH, SPLIT_NUMBER)

  print "adding features to trainer"
  # 给bow添加词汇信息
  for jpg_file in trainPosList:
    print jpg_file
    bow_kmeans_trainer.add(extract_sift(jpg_file, extract, detect))

  for jpg_file in trainNegList:
    print jpg_file
    siftData = extract_sift(jpg_file, extract, detect)
    if type(siftData) != NONE_TYPE:
      bow_kmeans_trainer.add(siftData)
  
  vocabulary = bow_kmeans_trainer.cluster()
  extract_bow.setVocabulary(vocabulary)

  # 训练集
  traindata, trainlabels = [],[]
  for jpg_file in trainPosList:
    print jpg_file
    traindata.extend(bow_features(cv2.imread(jpg_file, READ_TYPE), extract_bow, detect))
    trainlabels.append(1)

  for jpg_file in trainNegList:
    print jpg_file
    bowData = bow_features(cv2.imread(jpg_file, READ_TYPE), extract_bow, detect)
    if type(bowData) != NONE_TYPE:
      traindata.extend(bowData)
      trainlabels.append(-1)

  # 测试集
  testdata, testlabels = [],[]
  for jpg_file in testPosList:
    print jpg_file
    testdata.extend(bow_features(cv2.imread(jpg_file, READ_TYPE), extract_bow, detect))
    testlabels.append(1)
  
  for jpg_file in testNegList:
    print jpg_file
    bowData = bow_features(cv2.imread(jpg_file, READ_TYPE), extract_bow, detect)
    if type(bowData) != NONE_TYPE:
      testdata.extend(bowData)
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

if __name__ =='__main__':
  detector()