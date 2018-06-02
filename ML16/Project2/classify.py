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
    bins = 10
    max_bin = 1000.0
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


def load_segmented(mode, targets, parent_data_folder, bins):
    X = []
    y = []
    X_file = join(parent_data_folder, 'X_'+str(bins)+'.npy')
    y_file = join(parent_data_folder, 'y_'+str(bins)+'.npy')
    X_test_file = join(parent_data_folder, 'X_'+str(bins)+'_test.npy')
    folder_seg = None
    folder_seg_train = join(parent_data_folder, 'set_train_segmented')
    folder_seg_test = join(parent_data_folder, 'set_test_segmented')
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

        x_g = np.load(join(folder_seg, f_g))

        # spatial histograms
        #x = spatial_hist(x_g, cube_size)

        new_size = x_g.shape[0] * x_g.shape[1] * x_g.shape[2]
        x_g = np.reshape(x_g, (1, new_size))
        if np.max(x_g) == 0:
            print 'Fatal ERROR'

        x_g = x_g[x_g <> 0]
        # convert the whole image to histogram
        x, bin_edges = np.histogram(x_g, bins=bins)

        if X == []:
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
if user == 'igor':
    parent_data_folder = join(os.environ['HOME'], 'ML/data_2/')
elif user == 'pesici':
    parent_data_folder = os.environ['HOME']



targets_file = join(parent_data_folder, 'targets.csv')

targets = [0]
with open(targets_file, 'rb') as csvfile:
    for line in csvfile:
        targets.append(int(line))
 

cube_size = 16
print 'cube_size = ', cube_size

bins = 100
X = None
y = []
X_test = None
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
    X, y = load_segmented('train', targets, parent_data_folder, bins)
    X_test = load_segmented('test', targets, parent_data_folder, bins)   


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

# Regression model
classifier = None
# Remove features with too low between-subject variance e.g. nulls
# Normalize the data so it has 0 mean normal variance 
scaler = preprocessing.StandardScaler()
myFS0 = MyFeatSelector(0)
myFS1 = MyFeatSelector(100)
myFS2 = MyFeatSelector(200)
myFS3 = MyFeatSelector(50)

method = 'rf'


if method == 'rf':
    print 'Running ', method
    
    # SVM 
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, class_weight='balanced')
    #svmLin = SVC(kernel='linear', class_weight='balanced')
    
    #svmQuad = SVC(kernel='poly', degree=2)

    ests= [rf]
    
    n_estimators_range = [180, 200]
    max_depth_range = [2, 4]

    pipe = Pipeline([
            ('myFS', myFS0),
            #('scaler', scaler),
            ('est', rf),
            ])

    params = dict(
        myFS=[myFS0, myFS3, myFS1],
        #scaler=[scaler],
        est=ests,
        est__n_estimators=n_estimators_range,
        est__max_depth=max_depth_range
        )

    # does cross-validation with 3-fold for each combination of kernels and Cs
    classifier = GridSearchCV(pipe, param_grid=params, n_jobs=4, cv = 20, scoring='neg_log_loss')
    #reg = pipe

    classifier.fit(X, y)
    print "Best parameters set found on development set:"
    print(classifier.best_params_)



if method == 'SVM':
    print 'Running ', method
    
    # SVM 
    svmRBF = SVC(kernel='rbf', class_weight='balanced', probability=True)
    #svmLin = SVC(kernel='linear', class_weight='balanced')
    
    #svmQuad = SVC(kernel='poly', degree=2)

    ests= [svmRBF]
    
    C_range = [1e-1, 1, 10]
    gamma_range = [1e-3, 1e-2, 1e-1]

    pipe = Pipeline([
            ('myFS', myFS0),
            ('scaler', scaler),
            ('est', svmRBF),
            ])

    params = dict(
        myFS=[myFS0, myFS3, myFS1, myFS2],
        scaler=[scaler],
        est=ests,
        est__C=C_range,
        est__gamma=gamma_range
        )

    # does cross-validation with 3-fold for each combination of kernels and Cs
    classifier = GridSearchCV(pipe, param_grid=params, n_jobs=4, cv = 20, scoring='neg_log_loss')
    #reg = pipe

    classifier.fit(X, y)
    print "Best parameters set found on development set:"
    print(classifier.best_params_)




predictions = classifier.predict_proba(X_test)
predictions = predictions[:,1]


train_pred = classifier.predict_proba(X)
error = log_loss(y, train_pred)
print 'Training error', error

with open(join(parent_data_folder,"predictions.csv"),"wb") as f:
    f.write(b'ID,Prediction\n')
    for i, pred in enumerate(predictions):
        f.write(str(i+1)+','+str(pred)+'\n')
print 'Done.'
