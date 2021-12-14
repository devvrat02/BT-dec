import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
path = os.listdir('brain_tumor/Training/')
classes = {'no_tumor':0, 'pituitary_tumor':1,'glioma_tumor':2,'meningioma_tumor':3}
import cv2
X = []
Y = []
for cls in classes:
    pth = 'brain_tumor/Training/'+cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j, 0)
        img = cv2.resize(img, (200,200))
        X.append(img)
        Y.append(classes[cls])

X = np.array(X)
Y = np.array(Y)

X_updated = X.reshape(len(X), -1)

xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10,
                                               test_size=.20)

xtrain = xtrain/255
xtest = xtest/255

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
dec = {0:'No Tumor', 1:'Pituitary Tumor', 2:'Glioma tumor',3:'Meningioma tumor'}
import warnings
warnings.filterwarnings('ignore')

lg = LogisticRegression(C=0.1)
lg.fit(xtrain, ytrain)

sv = SVC()
sv.fit(xtrain, ytrain)
pred = sv.predict(xtest)


#input and test main.py
jpg='brain_tumor/Testing/no_tumor/image(2).jpg'

img = cv2.imread(jpg,0)
img1 = cv2.resize(img, (200,200))
img1 = img1.reshape(1,-1)/255
p = sv.predict(img1)
plt.title(dec[p[0]])
plt.imshow(img, cmap='gray')
plt.axis('off')