# Introduce
a program which use sift or lbp conbine with svm to do object detection

# Files
images/: train and test image sets

myUtils.py: some common method

trainLbp.py: train lbp+svm model

trainSift.py: train sift+svm model

detection.py: use model which's trainned by trainLbp.py/trainSift.py to do object detecion

# Usage
```sh
python trainLbp.py
python trainSift.py
python detection.py
```