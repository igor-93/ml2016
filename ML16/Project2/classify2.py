from __future__ import division
import os
from os.path import isfile, join
import numpy as np
import gc
import re
import nibabel as nib
import getpass
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from feat_sel import MyFeatSelector

# help functions to sort files nicely
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)



# help function to visualize
def nda_show(nda, title=None, margin=0.05, dpi=40 ):
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda,interpolation=None)
    
    if title:
        plt.title(title)
    
    plt.show()


def spatial_hist(img, vSize):
    bins = vSize
    max_bin = 1500.0
    width = max_bin/bins
    bin_edges = np.arange(0, max_bin+width, width)
    result = []
    if vSize > 1:
        #imgCUT = np.zeros((img.shape[0]/vSize, img.shape[1]/vSize, img.shape[2]/vSize))
        #print 'imgCUT shape: ', imgCUT.shape
        for i in range(0, img.shape[0], vSize):
            for j in range(0, img.shape[1], vSize):
                for k in range(0, img.shape[2], vSize):
                    #result = np.sum(img[i:i+vSize, j:j+vSize, k:k+vSize])
                    cube = img[i:i+vSize, j:j+vSize, k:k+vSize]
                    new_size = cube.shape[0] * cube.shape[1] * cube.shape[2]
                    cube = np.reshape(cube, (1, new_size))
                    nz = np.count_nonzero(cube)
                    if nz == 0:
                        x = [0]*bins
                    else:
                        cube = cube[cube <> 0]
                        x, bin_edges = np.histogram(cube, bins=bin_edges)

                    result.extend(x)
                    #print 'result: ', i/vSize, j/vSize, k/vSize
                    #imgCUT[i/vSize][j/vSize][k/vSize] = result
    return result


def load_segmented(mode, targets, parent_data_folder, bins, cube_size):
    X = None
    y = []
    X_file = join(parent_data_folder+'Images/', 'X_'+str(bins)+'.npy')
    y_file = join(parent_data_folder+'Images/', 'y_'+str(bins)+'.npy')
    X_test_file = join(parent_data_folder+'Images/', 'X_'+str(bins)+'_test.npy')
    folder_seg = None
    folder_seg_train = join(parent_data_folder+'Images/', 'set_train_segmented')
    folder_seg_test = join(parent_data_folder+'Images/', 'set_test_segmented')
    if mode == 'train':
        folder_seg = folder_seg_train
    elif mode == 'test':
        folder_seg = folder_seg_test
    files_w = [f for f in os.listdir(folder_seg) if isfile(join(folder_seg, f)) and f.endswith('.npy') and f.startswith('w_')]
    files_g = [f for f in os.listdir(folder_seg) if isfile(join(folder_seg, f)) and f.endswith('.npy') and f.startswith('g_')]
    sort_nicely(files_w)  
    sort_nicely(files_g)    
    
    for f_w, f_g in zip(files_w, files_g):

        n_w = int(f_w.split('_')[2].split('.')[0])
        n_g = int(f_g.split('_')[2].split('.')[0])
        if n_g <> n_w:
            print 'Error: f_W and f_g are not matching: '+str(n_w)+', '+str(n_g)
            break
        else:
            print 'Iteration ('+mode+') ', n_g

        #x_w = np.load(join(folder_seg, f_w))
        x_g = np.load(join(folder_seg, f_g))

        #x_w = ave_cubes(x_w, cube_size, 'white')
        x_g = spatial_hist(x_g, cube_size)
        #x = np.hstack((x_w, x_g))

        '''new_size = x_g.shape[0] * x_g.shape[1] * x_g.shape[2]
        x_g = np.reshape(x_g, (1, new_size))
        if np.max(x_g) == 0:
            print 'Fatal ERROR'

        x_g = x_g[x_g <> 0]
        # convert the image to histogram
        x, bin_edges = np.histogram(x_g, bins=bins)#, range=(0, 2500))'''


        if X == None:
            X = x
        else:
            X = np.vstack((X,x))

        if mode == 'train':
            y_tmp = targets[n_w]
            y.append(y_tmp)

        gc.collect()

    if mode == 'train':
        print 'Saving X to ', X_file
        np.save(X_file, X)    
        print 'Saving y to ', y_file
        np.save(y_file, y)  
        return X, y
    elif mode == 'test':
        print 'Saving X_test to ', X_test_file
        np.save(X_test_file, X)   
        return X  
    


parent_data_folder = None
user = getpass.getuser()
if user == 'dinesh':
    parent_data_folder = join(os.environ['HOME'], 'Desktop/ML16/Project2/')
elif user == 'igor':
    parent_data_folder = join(os.environ['HOME'], 'ML/data_2/')    
elif user == 'pesici':
    parent_data_folder = os.environ['HOME']


targets_file = join(parent_data_folder, 'targets.csv')

targets = [0]
with open(targets_file, 'rb') as csvfile:
    for line in csvfile:
        targets.append(int(line))
 

cube_size = 8
print 'cube_size = ', cube_size

bins = 100
X = None
y = []
X_test = None
bins = 100
X_file = join(parent_data_folder, 'X_'+str(bins)+'.npy')
y_file = join(parent_data_folder, 'y_'+str(bins)+'.npy')
X_test_file = join(parent_data_folder, 'X_'+str(bins)+'_test.npy')

if isfile(X_file) and isfile(y_file) and isfile(X_test_file):
    print 'Found ', X_file,' and ', y_file
    X = np.load(X_file)
    y = np.load(y_file)
    X_test = np.load(X_test_file)
    pass

else:    
    print 'Could not find ', X_file
    print ' and/or ', y_file
    print ' and/or ', X_test_file
    X, y = load_segmented('train', targets, parent_data_folder, bins, cube_size)
    X_test = load_segmented('test', targets, parent_data_folder, bins, cube_size)   


print 'X has size: ', X.shape
try:
    print 'y has size: ', y.shape
except:
    print 'y has size: ', len(y)
print 'X_test has size: ', X_test.shape



# Dimension reduction
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn import preprocessing, decomposition
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier

reg=RandomForestClassifier(n_estimators=130);
reg.fit(X,y)
predictions=reg.predict_proba(X_test);
arr=np.zeros((138,2))
arr[:,0]=np.arange(1,139)
arr[:,1]=predictions[:,1]
with open("prediction.csv","wb") as f:
    f.write(b'ID,Prediction\n')
    np.savetxt(f, arr,fmt='%d,%f', delimiter=",")
# with open(join(parent_data_folder,"predictions.csv"),"wb") as f:
#     f.write(b'ID,Prediction\n')
#     for i, pred in enumerate(arr):
#         f.write(str(i+1)+','+str(pred)+'\n')
print 'Done.'



# # Regression model
# classifier = None
# # Remove features with too low between-subject variance e.g. nulls
# variance_threshold01 = VarianceThreshold(threshold=.1)
# variance_threshold001 = VarianceThreshold(threshold=.001)
# # Normalize the data so it has 0 mean normal variance 
# scaler = preprocessing.StandardScaler()
# min_max_scaler = preprocessing.MinMaxScaler()


# method = 'SVM'


# if method == 'my_pipe':
#     my_selector = MyFeatSelector(k=10)
#     svmLin = SVC(kernel='linear', C = 1)
    
#     classifier = Pipeline([
#             ('feat_selector', my_selector),
#             ('scaler', scaler),
#             ('est', svmLin),
#             ])

#     classifier.fit(X,y)

# if method == 'SVM':
#     print 'Running ', method
    
#     # SVM 
#     svmRBF = SVC(kernel='rbf', class_weight='balanced', probability=True)
#     svmLin = SVC(kernel='linear', class_weight='balanced')
#     rndForest = RandomForestClassifier(n_estimators=150)
#     #svmQuad = SVC(kernel='poly', degree=2)

#     ests= [svmRBF]
    
#     C_range = np.logspace(-4, 0, 5)
#     gamma_range = np.logspace(-9, 3, 13)

#     pipe = Pipeline([
#             #('variance_threshold', variance_threshold01),
#             ('scaler', scaler),
#             ('est', svmRBF),
#             ])

#     params = dict(
#         #variance_threshold=[variance_threshold01, variance_threshold001],
#         scaler=[scaler],
#         est=ests,
#         est__C=C_range,
#         est__gamma=gamma_range
#         )

#     # does cross-validation with 3-fold for each combination of kernels and Cs
#     classifier = GridSearchCV(pipe, param_grid=params, n_jobs=2, cv = 10)
#     #reg = pipe

#     classifier.fit(X, y)
#     print "Best parameters set found on development set:"
#     print(classifier.best_params_)




# predictions = classifier.predict_proba(X_test)
# train_pred = classifier.predict(X)
# error = log_loss(y, train_pred)

# print 'training error', error
# arr=np.zeros((138,2))
# arr[:,0]=predictions[:,0]
# arr[:,1]=predictions[:,2]
# with open(join(parent_data_folder,"predictions.csv"),"wb") as f:
#     f.write(b'ID,Prediction\n')
#     for i, pred in enumerate(arr):
#         f.write(str(i+1)+','+str(pred)+'\n')
# print 'Done.'
