# semi-hiararchial classification based on the gray matter histograms
# i.e. first classify the age then use 2-class estimator for young and 4-class estimator for the old (one-vs-all)


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


def spatial_hist(img, vSize, bins):
    range_min = 1
    range_max = 1500
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
                        x, bin_edges = np.histogram(cube, bins=bins, range=(range_min, range_max))

                    result.extend(x)
    return result


def load_segmented(mode, targets, parent_data_folder, cube_size, bins):
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
            if n_g % 20 == 0:
                print 'Iteration ('+mode+') ', n_g

        x_g = np.load(join(folder_seg, f_g))

        # spatial histograms
        x = spatial_hist(x_g, cube_size, bins)

        #new_size = x_g.shape[0] * x_g.shape[1] * x_g.shape[2]
        #x_g = np.reshape(x_g, (1, new_size))
        #if np.max(x_g) == 0:
        #    print 'Fatal ERROR'

        #x_g = x_g[x_g <> 0]
        # convert the whole image to histogram
        #x, bin_edges = np.histogram(x_g, bins=bins)

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
    parent_data_folder = join(os.environ['HOME'], 'ML/data_3/')
elif user == 'pesici':
    parent_data_folder = os.environ['HOME']



targets_file = join(parent_data_folder, 'targets.csv')

targets = [(8,8,8)]
with open(targets_file, 'rb') as csvfile:
    for line in csvfile:
        values = line.split(',')
        targets.append(tuple([int(values[0]), int(values[1]), int(values[2]) ]))


mapped_targets = range(len(targets))
map_to_class = {(0,0,0):0, (0,0,1):1, (0,1,1):2,
                (1,0,0):3, (1,0,1):4, (1,1,1):5, (8,8,8):888 }



for i, item in enumerate(targets):
    mapped_targets[i] = map_to_class[item]

from collections import Counter
count = Counter(mapped_targets)
print 'count: ', count




cube_size = 16
print 'cube_size = ', cube_size

bins = 80
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
    X, y = load_segmented('train', mapped_targets, parent_data_folder, cube_size, bins)
    X_test = load_segmented('test', mapped_targets, parent_data_folder, cube_size, bins)   


print 'X has size: ', X.shape
try:
    print 'y has size: ', y.shape
except:
    print 'y has size: ', len(y)
print 'X_test has size: ', X_test.shape



# Dimension reduction
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import hamming_loss

classifier = None

var_threshold = VarianceThreshold(threshold=0.1)
scaler = preprocessing.StandardScaler()
anova = SelectKBest(f_classif, k=10000)

method = 'fl'

# first classify labels 2 and 5 (young people) against 0,1,3,4 (old people)
first_lvl_targets = []
for i, lab in enumerate(y):
    if lab == 2 or lab == 5:
        first_lvl_targets.append(0)     # young
    else:
        first_lvl_targets.append(1)    



if method == 'fl':
    print 'Running ', method
    
    pca = PCA(n_components=277)
    svm1 = SVC(kernel='linear', C=1)

    ests = [svm1]
    

    pipe = Pipeline([
            ('varTh', var_threshold),
            ('scaler', scaler),
            ('anova', anova),
            ('pca', pca),
            ('est', svm1),
            ])

    params = dict(
        varTh=[var_threshold],
        scaler=[scaler],
        anova=[anova],
        pca=[pca],
        est=ests,
        est__C=[1e-1, 1]
        )

    # does cross-validation with 10-fold for each combination of kernels and Cs
    classifier1 = GridSearchCV(pipe, param_grid=params, n_jobs=2, cv = 10, scoring='accuracy')
    #reg = pipe

    classifier1.fit(X, first_lvl_targets)
    print "Best parameters set found on development set:"
    print(classifier1.best_params_)


predictions1 = classifier1.predict(X_test)
train_pred1 = classifier1.predict(X)
error1 = hamming_loss(first_lvl_targets, train_pred1)
print 'Training error 1', error1

# distinguish 2 cases:
#       1. 2 class classification for young male vs female
#       2. 4 class classification for old people


X21 = []      # young people
y21 = []
X22 = []      # old people
y22 = []
for i, lab in enumerate(first_lvl_targets):
    if lab == 0:       # young people
        X21.append(X[i,:])
        if y[i] == 2:
            y21.append(0)
        elif y[i] == 5:
            y21.append(1)   
        else:
            print 'Error 1 in labeling...' , y[i]
    else:           # old people
        X22.append(X[i,:])
        if y[i] == 0:
            y22.append(0)
        elif y[i] == 1:
            y22.append(1) 
        elif y[i] == 3:
            y22.append(2)  
        elif y[i] == 4:
            y22.append(3)           
        else:
            print 'Error 2 in labeling...' , y[i]


# classify young people (male vs female)
if method == 'fl':
    print 'Running ', method
    
    pca = PCA(n_components=144)
    svm1 = SVC(kernel='linear')

    ests = [svm1]
    

    pipe = Pipeline([
            ('varTh', var_threshold),
            ('scaler', scaler),
            ('anova', anova),
            ('pca', pca),
            ('est', svm1),
            ])

    params = dict(
        varTh=[var_threshold],
        scaler=[scaler],
        anova=[anova],
        pca=[pca],
        est=ests,
        est__C=[1e-2, 1e-1, 1]
        )

    # does cross-validation with 10-fold for each combination of kernels and Cs
    classifier21 = GridSearchCV(pipe, param_grid=params, n_jobs=2, cv = 10, scoring='accuracy')
    #reg = pipe

    classifier21.fit(X21, y21)
    print "Best parameters set found on development set:"
    print(classifier21.best_params_)



predictions21 = classifier21.predict(X_test)
train_pred21 = classifier21.predict(X21)
error21 = hamming_loss(y21, train_pred21)
print 'Training error 21', error21



if method == 'fl':
    print 'Running ', method
    
    pca = PCA(n_components=120)

    rf1 = OneVsRestClassifier(RandomForestClassifier(n_estimators=150, max_features='log2', class_weight='balanced'))
    rf2 = OneVsRestClassifier(RandomForestClassifier(n_estimators=250, max_features='log2', class_weight='balanced'))

    ests= [rf1, rf2]
    

    pipe = Pipeline([
            ('varTh', var_threshold),
            ('scaler', scaler),
            ('anova', anova),
            ('pca', pca),
            ('est', svm1),
            ])

    params = dict(
        varTh=[var_threshold],
        scaler=[scaler],
        anova=[anova],
        pca=[pca],
        est=ests,
        )

    # does cross-validation with 10-fold for each combination of kernels and Cs
    classifier22 = GridSearchCV(pipe, param_grid=params, n_jobs=2, cv = 10, scoring='accuracy')
    #reg = pipe

    classifier22.fit(X22, y22)
    print "Best parameters set found on development set:"
    print(classifier22.best_params_)


predictions22 = classifier22.predict(X_test)
train_pred22 = classifier22.predict(X22)
error22 = hamming_loss(y22, train_pred22)
print 'Training error 22', error22



# merge all the predictions
predictions = []
for p1, p21, p22 in zip(predictions1, predictions21, predictions22):
    if p1 == 0:         # young
        if p21 == 0:    # male
            predictions.append(2)
        else:
            predictions.append(5)
    else:               # old
        if p22 == 0:
            predictions.append(0)
        elif p22 == 1:
            predictions.append(1)
        elif p22 == 2:
            predictions.append(3)
        else:
            predictions.append(4)


with open(join(parent_data_folder,"predictions.csv"),"wb") as f:
    f.write(b'ID,Sample,Label,Predicted\n')
    line_counter = 0
    for i, pred in enumerate(predictions):
        res1 = 'True'
        res2 = 'True'
        res3 = 'True'
        if pred < 3:        # gender
            res1 = 'False'
        else:
            res1 = 'True'
        if pred % 3 == 2:   # age
            res2 = 'True'
        else:
            res2 = 'False'
        if pred % 3 == 0:   # health
            res3 = 'False'
        else:
            res3 = 'True'
        line1 = str(line_counter)+','+str(i)+',gender,'+res1+'\n'
        line_counter += 1
        line2 = str(line_counter)+','+str(i)+',age,'+res2+'\n'
        line_counter += 1
        line3 = str(line_counter)+','+str(i)+',health,'+res3+'\n'
        line_counter += 1
        f.write(line1)
        f.write(line2)
        f.write(line3)


print 'Done.'