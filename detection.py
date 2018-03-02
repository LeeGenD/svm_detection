#! usr/bin/python
#coding=utf-8
import time
import cv2
import numpy as np
from sklearn.externals import joblib
from skimage import feature as skif
from myUtils import globSplit, getSelectiveSelectRect, NMS, calArea

POS_PATH = "./images/Positive_train/*.jpg"
NEG_PATH = "./images/Negative_train/*.jpg"
detectorList = [
  {
    "cache": "./backup/lbp.svm.joblib.pkl",
    "name": "lbp",
    "color": (0,0,255)
  },
  {
    "cache": "./backup/sift.svm.joblib.pkl",
    "name": "sift",
    "color": (255,0,0)
  }
]

DEST_SIZE = (150, 150)
SPLIT_NUMBER = 2000
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

def getLbpData(image, hist_size=256, lbp_radius=1, lbp_point=8):
    image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用LBP方法提取图像的纹理特征.
    lbp = skif.local_binary_pattern(image, lbp_point, lbp_radius, 'default')
    # 统计图像的直方图
    max_bins = int(lbp.max() + 1)
    # hist size:256
    hist, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
    return [hist]

def showReuslt(im, ssList, color):
    img = im.copy()
    for ssArea in ssList:
      cv2.putText(img, str(round(ssArea[4], 3)), (ssArea[0], ssArea[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
      cv2.rectangle(img, (ssArea[0], ssArea[1]), (ssArea[2], ssArea[3]), color)

    cv2.imshow('result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detector():
  detect, extract = get_extract_detect()
  matcher = get_flann_matcher()
  extract_bow = get_bow_extractor(extract, matcher)

  print "building BOWKMeansTrainer..."
  bow_kmeans_trainer = cv2.BOWKMeansTrainer(36)
  trainPosList, _ = globSplit(POS_PATH, SPLIT_NUMBER)
  trainNegList, _ = globSplit(NEG_PATH, SPLIT_NUMBER)

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
  
  # 生成词汇，并设置词汇
  vocabulary = bow_kmeans_trainer.cluster()
  extract_bow.setVocabulary(vocabulary)
  print "adding to train data"

  imageList = ['./images/demo.jpg']
  print imageList
  for i in range(len(imageList)):
    imagePath = imageList[i]
    im = cv2.imread(imagePath)
    ssList = getSelectiveSelectRect(im)
    ssList = NMS(ssList, 0.2)
    for detectorItem in detectorList:
      print detectorItem
      ssListNew = []
      testdata = []
      mlp = joblib.load(detectorItem['cache'])
      for ssArea in ssList:
        crop_img = im[ssArea[1]:ssArea[3], ssArea[0]:ssArea[2]]
        crop_img = cv2.resize(crop_img, DEST_SIZE, interpolation=cv2.INTER_CUBIC)
        featureData = None
        if detectorItem['name'] == 'sift':
          featureData = bow_features(crop_img, extract_bow, detect)
        elif detectorItem['name'] == 'lbp':
          featureData = getLbpData(crop_img)
        if type(featureData) != NONE_TYPE:
          testdata.extend(featureData)
          ssListNew.append(ssArea)
      predict_proba = mlp.predict_proba(np.array(testdata))
      predictData = []
      for j in range(len(ssListNew)):
        ssRect = ssListNew[j]
        if predict_proba[j][1] > 0.5:
          ssListNew[j][4] = predict_proba[j][1]
          predictData.append(ssListNew[j])
      showReuslt(im, predictData, detectorItem['color'])

if __name__ =='__main__':
  detector()