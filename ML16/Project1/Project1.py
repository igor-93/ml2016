# -*- coding: utf-8 -*-
"""
Machine Learning Project1
"""
from __future__ import division
import os
import numpy as np
import nibabel as nib

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

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
X = np.zeros((n_train,n))

for t in range(0,n_train):
    file_name = "{0}_{1}.nii".format('train',t+1)
    file_path = os.path.join(curr_path,data_path, file_name)
    img = nib.load(file_path).get_data()
    img = np.sum(img, axis=3)     # to remove the '4th' dimension which is basically intesity
    # crop the image with given offset
    img = img[rmin-offset:rmax+offset,cmin-offset:cmax+offset,zmin-offset:zmax+offset]
    X[t,:] = img.reshape(n,)
    print('file ' , file_name, ' read successfully')
    
n_features = 25  # number of low dimensional features we want
print("Extracting the top %d eigenimages from %d images" % (n_features,n_train))
pca = PCA(n_components = n_features, svd_solver = 'randomized' , whiten = True).fit(X)
#eigenimages = pca.components_   # This is the matrix used to transform to lowdimensional feature space
X_train_pca = pca.transform(X)  # Training sets after transformation
print("PCA completed successfully ...")


# save data
# np.save('X_train_pca',X_train_pca)
# X_train_pca = np.load('X_train_pca.npy')



# Load csv file into numpy array
age_train=np.genfromtxt(os.path.join(curr_path,'targets.csv'), delimiter=',')



# Regression model
# Ridge regression with lambda optimized using generalized cross validation
# reg = RidgeCV()
#degree = 5
#reg = make_pipeline(PolynomialFeatures(degree), RidgeCV())
reg = LassoCV()
reg.fit(X_train_pca,age_train)
print("Data fitted with CV Ridge Regression")



# Prediction Error
age_train_predict = reg.predict(X_train_pca)
training_error=((age_train_predict-age_train).dot(age_train_predict-age_train))/n
print('Training Error: %f'%(training_error))



# for j in range(0,int(n_features/3)):
#    save_name = "{0}_{1}-{2}.npy".format('eigenimages',3*j+1,3*(j+1))
#    np.save(save_name,eigenimages[3*j:3*(j+1),:])

#n_test=48
n_test=138



# Prediction array
#prediction = np.zeros((48,2))
#prediction[:,0]=np.arange(1,49)
prediction = np.zeros((138,2))
prediction[:,0]=np.arange(1,139)


#data_path = 'Images/set_train'
data_path = 'Images/set_test'
#for t in range(230,278):
for t in range(0,n_test):
    #file_name = "{0}_{1}.nii".format('train',t+1)
    file_name = "{0}_{1}.nii".format('test',t+1)
    file_path = os.path.join(curr_path,data_path, file_name)
    img = nib.load(file_path).get_data()
    img = np.sum(img, axis=3)     # to remove the '4th' dimension which is basically intesity
    # crop the image with given offset
    img = img[rmin-offset:rmax+offset,cmin-offset:cmax+offset,zmin-offset:zmax+offset]
    img = img.reshape(n,)
    transformed = pca.transform(img)#eigenimages*img
    prediction[t,1] = reg.predict(transformed) # or transformed.reshape(-1,1)
    print("Age prediction for image %d completed"%(t+1))


#test_error = np.sum((prediction[:,1]-age_train[230:278])**2)/48.0
with open("prediction.csv","wb") as f:
    f.write(b'ID,Prediction\n')
    np.savetxt(f, prediction.astype(int), fmt='%i', delimiter=",")
