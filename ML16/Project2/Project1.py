# -*- coding: utf-8 -*-
"""
Machine Learning Project1
"""
from __future__ import division
import os
import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use('Agg')

from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

curr_path = os.getcwd()
data_path = 'Images/set_train'     # training folder name

n_train = 278   # number of trainining samples to be used for dimensionality reduction

# all images have identical boundaries (checked)
rmin = 18
rmax = 154
cmin = 18
cmax = 188
zmin = 7
zmax = 151
offset = 0
n = 3329280     # size of the truncated image vector
#n=6443008
n_bins=500
#n_bins=500
X = np.zeros((n_train,n_bins))



for t in range(0,n_train):
    #t=0
    file_name = "{0}_{1}.nii".format('train',t+1)
    file_path = os.path.join(curr_path,data_path, file_name)
    img = nib.load(file_path).get_data()
    img = np.sum(img, axis=3)     # to remove the '4th' dimension which is basically intesity
    # crop the image with given offset
    img = img[rmin-offset:rmax+offset,cmin-offset:cmax+offset,zmin-offset:zmax+offset]
    # print(img[100,100,100])
    imgV = img.reshape(n,)
    hist, bins = np.histogram(imgV,bins=n_bins, range=[1,1700])
    X[t,:] = hist
    print('Generating histogram for image ',t,' done.')    



# Load csv file into numpy array
age_train=np.genfromtxt(os.path.join(curr_path,'targets.csv'), delimiter=',')



# Regression mode
#reg = LassoCV(max_iter=10000)
#reg = LassoCV()
reg = MLPClassifier()
#reg = RidgeCV()
#reg = ElasticNetCV()
#reg = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)})
#reg = SVR()
#reg.fit(X_train_pca,age_train)
reg.fit(X,age_train)
#print(reg.alpha_)
print("Data fitted with CV Ridge Regression")


n_test=138


# Prediction array
prediction = np.zeros((138,2))
prediction[:,0]=np.arange(1,139)


data_path = 'Images/set_test'
for t in range(0,n_test):
    file_name = "{0}_{1}.nii".format('test',t+1)
    file_path = os.path.join(curr_path,data_path, file_name)
    img = nib.load(file_path).get_data()
    img = np.sum(img, axis=3)     # to remove the '4th' dimension which is basically intesity
    # crop the image with given offset
    img = img[rmin-offset:rmax+offset,cmin-offset:cmax+offset,zmin-offset:zmax+offset]
    imgV = img.reshape(n,)
    hist, bins = np.histogram(imgV,bins=n_bins, range=[1,1700])
    prediction[t,1] = reg.predict(hist)
    print('Generating histogram for image ',t,' done.')
    print("Age prediction for image %d completed"%(t+1))


with open("prediction.csv","wb") as f:
    f.write(b'ID,Prediction\n')
    np.savetxt(f, prediction.astype(int), fmt='%i', delimiter=",")
