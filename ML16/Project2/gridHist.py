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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

curr_path = os.getcwd()
data_path = 'Images/set_train'     # training folder name

n_train = 278   # number of trainining samples to be used for dimensionality reduction
from sklearn.linear_model import RandomForestClassifier
# all images have identical boundaries (checked)
rmin = 18
rmax = 154 #136=8*17
cmin = 18 
cmax = 188 #170=10*17
zmin = 7
zmax = 151 #144=9*16
offset = 0
n = 3329280     # size of the truncated image vector
#n=6443008
n_bins=500
#n_bins=500

ngX=8
ngY=10
ngZ=9
gX=17
gY=17
gZ=16
X = np.zeros((n_train,45*gX*gY*gZ))
for t in range(0,n_train):
    print('Generating histogram for image ',t,' done.')
    file_name = "{0}_{1}.nii".format('train',t+1)
    file_path = os.path.join(curr_path,data_path, file_name)
    img = nib.load(file_path).get_data()
    img = np.sum(img, axis=3)     # to remove the '4th' dimension which is basically intesity
    # crop the image with given offset
    img = img[rmin-offset:rmax+offset,cmin-offset:cmax+offset,zmin-offset:zmax+offset]
    for i in range(0, gX):
		for j in range(0, gY):
		    for k in range(0, gZ):
				block = img[i*ngX:(i+1)*ngX,j*ngY:(j+1)*ngY,k*ngZ:(k+1)*ngZ]
				hist,bins = np.histogram(block,bins=45,range=[1,1700])
				X[t, (i*gY*gZ+j*gZ+k)*45:(i*gY*gZ+j*gZ+k+1)*45] = hist

#for t in range(0,n_train):
#    #t=0
#    file_name = "{0}_{1}.nii".format('train',t+1)
#    file_path = os.path.join(curr_path,data_path, file_name)
#    img = nib.load(file_path).get_data()
#    img = np.sum(img, axis=3)     # to remove the '4th' dimension which is basically intesity
#    # crop the image with given offset
#    img = img[rmin-offset:rmax+offset,cmin-offset:cmax+offset,zmin-offset:zmax+offset]
#    # print(img[100,100,100])
#    imgV = img.reshape(n,)
#    hist, bins = np.histogram(imgV,bins=n_bins, range=[1,1700])
#    X[t,:] = hist
#    print('Generating histogram for image ',t,' done.')    

# Load csv file into numpy array
age_train=np.genfromtxt(os.path.join(curr_path,'targets.csv'), delimiter=',')



# Regression mode
#reg = LassoCV(max_iter=10000)
#reg = LassoCV()
#reg=RandomForestClassifier(n_estimators=125)
reg = LogisticRegression()
#reg = MLPClassifier()
#reg=RandomForestClassifier(n_estimators=100)
#reg = GradientBoostingClassifier()
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
prediction = np.zeros((138,3))
prediction[:,0]=np.arange(1,139)

#h = np.zeros
data_path = 'Images/set_test'
for t in range(0,n_test):
    file_name = "{0}_{1}.nii".format('test',t+1)
    file_path = os.path.join(curr_path,data_path, file_name)
    img = nib.load(file_path).get_data()
    img = np.sum(img, axis=3)     # to remove the '4th' dimension which is basically intesity
    # crop the image with given offset
    img = img[rmin-offset:rmax+offset,cmin-offset:cmax+offset,zmin-offset:zmax+offset]
    h=np.zeros(45*gX*gY*gZ)
    for i in range(0, gX):
		for j in range(0, gY):
		    for k in range(0, gZ):
				block = img[i*ngX:(i+1)*ngX,j*ngY:(j+1)*ngY,k*ngZ:(k+1)*ngZ]
				hist,bins = np.histogram(block,bins=45,range=[1,1700])
				h[(i*gY*gZ+j*gZ+k)*45:(i*gY*gZ+j*gZ+k+1)*45] = hist
    print reg.predict_proba(h)
    prediction[t,1:] = reg.predict_proba(h)
    print(prediction[t,1])
    print('Generating histogram for image ',t,' done.')
    print("Age prediction for image %d completed"%(t+1))

#arr= np.ndarray((n_test,2),dtype=object)
arr=np.zeros((138,2))
arr[:,0]=prediction[:,0]
arr[:,1]=prediction[:,2]
with open("prediction.csv","wb") as f:
    f.write(b'ID,Prediction\n')
    np.savetxt(f, arr,fmt='%d,%f', delimiter=",")
