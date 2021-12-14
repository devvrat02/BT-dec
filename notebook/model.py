import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import os
path = os.listdir('brain_tumor/Training/')
classes = {'no_tumor', 'pituitary_tumor','glioma_tumor','meningioma_tumor'}

import cv2
X = []
Y = []
for cls in classes:
    pth = 'brain_tumor/Training/'+cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j, 0)
        img = cv2.resize(img, (200,200))
        X.append(img)
        Y.append(cls)

X = np.array(X)
Y = np.array(Y)

X_updated = X.reshape(len(X), -1)

xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10,
                                               test_size=.20)

xtrain = xtrain/255
xtest = xtest/255


sv = SVC()
sv.fit(xtrain, ytrain)


pickle.dump(sv, open("model.pkl", "wb"))


#input and test main.py
# jpg='brain_tumor/Testing/no_tumor/image(2).jpg'

# img = cv2.imread(jpg,0)
# img1 = cv2.resize(img, (200,200))
# img1 = img1.reshape(1,-1)/255
# p = sv.predict(img1)
# plt.title(p[0])
# plt.imshow(img, cmap='gray')
