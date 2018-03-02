#! usr/bin/python
#coding=utf-8
import glob
import selectivesearch
from PIL import Image
import cv2
import  xml.dom.minidom
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

COLOR = (55,255,155)
COLOR_SS = (0,0,255)

def globSplit(path, splitNumber):
  """
  读取path中的文件，并分为两个列表
  """
  count = 0
  list1 = []
  list2 = []
  for jpg_file in glob.glob(path):
    if count < splitNumber:
      list1.append(jpg_file)
    elif count < splitNumber * 2:
      list2.append(jpg_file)
    else:
      break
    count += 1
  return list1, list2

def predictFromProba(probaList, posProba=0.5, posTAG=1, negTag=-1):
  """
  根据设置的可能性(probability)阈值返回分类列表
  大于posProba则标记为正样本，否则为负样本
  """
  labelList = []
  for proba in probaList:
    #proba[1]为正样本概率
    if proba[1] >= posProba:
      labelList.append(posTAG)
    else:
      labelList.append(negTag)
  return labelList

def printList(l):
    newStr = ''
    for i in l:
      newStr += str(i) + ' '
    print newStr

def pointInArea(point, area):
    x1 = area[0]
    y1 = area[1]
    x2 = area[2]
    y2 = area[3]

    x = point[0]
    y = point[1]

    if x >= x1 and x <= x2 and y >= y1 and y <= y2:
        return True
    return False

def calArea(area):
    x1 = area[0]
    y1 = area[1]
    x2 = area[2]
    y2 = area[3]
    return (x1 - x2) * (y1 - y2)

def CrossLine(left, right, y, top, bottom, x):
    # 判断一根横线和一根竖线是否交叉
    # 横线有三个参数：left, right和y
    # 竖线有三个参数：top, bottom和x
    return (top < y) and (bottom > y) and (left < x) and (right > x)

def IOU(rect1, rect2):
    x11 = rect1[0]
    y11 = rect1[1]
    x12 = rect1[2]
    y12 = rect1[3]

    x21 = rect2[0]
    y21 = rect2[1]
    x22 = rect2[2]
    y22 = rect2[3]

    if pointInArea((x11, y11), rect2) == False and pointInArea((x12, y12), rect2) == False \
        and pointInArea((x11, y12), rect2) == False and pointInArea((x12, y11), rect2) == False \
        and pointInArea((x21, y21), rect1) == False \
        and not CrossLine(x11, x12, y11, y21, y22, x21) and not CrossLine(x21, x22, y21, y11, y12, x11):
        return 0
    
    xList = [x11, x12, x21, x22]
    yList = [y11, y12, y21, y22]
    xList.sort()
    yList.sort()
    areaMiddle = calArea([xList[1], yList[1], xList[2], yList[2]])
    area1 = calArea(rect1)
    area2 = calArea(rect2)
    return float(areaMiddle) / (area1 + area2 - areaMiddle)

def NMS(rectList, threshold=.5):
    rectList = sorted(rectList, key=lambda rectList: rectList[4],
            reverse=True)
    i = 0
    while i < len(rectList):
        j = i + 1
        while j < len(rectList):
            iou = IOU(rectList[i], rectList[j])
            if iou > threshold:
                del rectList[j]
            else:
                j += 1
        i += 1
    return rectList
        
    print rectList

def getSelectiveSelectRect(im):
    shape = im.shape
    if shape[0] > shape[1]:
        maxScale = shape[0]
    else:
        maxScale = shape[1]
    img_lbl, regions = selectivesearch.selective_search(im, scale=maxScale, sigma=0.7, min_size=400)
    rectList = []
    originMaxSize = (shape[0] - 2) * (shape[1] - 2)
    for i in range(len(regions)):
        rect = regions[i]['rect']
        rect = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3], 0]
        size = calArea(rect)
        if size < 400:
            continue
        if size >= originMaxSize:
            continue
        rectList.append(rect)
    return rectList

def showImgWithSS(im, area, ssList):
    img = im.copy()
    cv2.rectangle(img, (area[0], area[1]), (area[2], area[3]),COLOR, 3)
    for ssArea in ssList:
        if len(ssArea) > 4:
            cv2.putText(img, str(round(ssArea[4], 3)), (ssArea[0], ssArea[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_SS, 2)
        cv2.rectangle(img, (ssArea[0], ssArea[1]), (ssArea[2], ssArea[3]),COLOR_SS) 

    cv2.imwrite('showImgWithSS.jpg', img)
    cv2.imshow('showImgWithSS', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
def minInList(li):
    minValue = 999
    for item in li:
        if item < minValue:
            minValue = item
    return minValue

def maxInList(li):
    maxValue = -999
    for item in li:
        if item > maxValue:
            maxValue = item
    return maxValue

def precisionRecallCurve(testlabels, predictResult):
    chartPrecision, chartRecall, _ = precision_recall_curve(testlabels, predictResult)
    average_precision = average_precision_score(np.array(testlabels), predictResult)
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision))
    plt.figure()
    plt.step(chartRecall, chartPrecision, color='b', alpha=0.2, where='post')
    plt.fill_between(chartRecall, chartPrecision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
        .format(average_precision))
    plt.show()
