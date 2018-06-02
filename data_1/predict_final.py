#from __future__ import division
import os
import os.path
import numpy as np
import gc
import nibabel as nib
from scipy import sparse, io
from nilearn import plotting
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.linear_model import RidgeCV, Lasso, LassoCV
import SimpleITK


def smooth_img(img, vSize):
    if vSize > 1:
        imgCUT = np.zeros((img.shape[0]/vSize, img.shape[1]/vSize, img.shape[2]/vSize))
        #print 'imgCUT shape: ', imgCUT.shape
        for i in range(0, img.shape[0], vSize):
            for j in range(0, img.shape[1], vSize):
                for k in range(0, img.shape[2], vSize):
                    #result = np.sum(img[i:i+vSize, j:j+vSize, k:k+vSize])
                    result = np.count_nonzero(img[i:i+vSize, j:j+vSize, k:k+vSize])
                    #print 'result: ', i/vSize, j/vSize, k/vSize
                    imgCUT[i/vSize][j/vSize][k/vSize] = result

    elif vSize == 1:
        img[img <> 0] = 1 
        imgCUT = img              

    new_size = imgCUT.shape[0] * imgCUT.shape[1] * imgCUT.shape[2]
    x = np.reshape(imgCUT, (1, new_size))
    
    if np.min(x) == 0 and np.max(x) == 0:
        print 'Error in smoothing: result are all zeros!!!!'
    return x



def nda_show(nda, title=None, margin=0.05, dpi=40 ):
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda,interpolation=None)
    
    if title:
        plt.title(title)
    
    plt.show()

def reg_run():

    parent_data_folder = ''
    import getpass
    user = getpass.getuser()
    if user == 'igor':
        parent_data_folder = '/home/igor/ML/data_1/'
    elif user == 'pesici':
        parent_data_folder = os.environ['HOME'] + '/'
    else:
        os.path.realpath(__file__)

    simple = False
    train_data_path = ''
    test_data_path = ''
    X_file_search = ''
    X_nd_file_search = ''
    Y_file_search = ''
    vSize = 8
    if not simple:
        train_data_path = parent_data_folder+'set_train_gray_matter_maps/'
        test_data_path = parent_data_folder+'set_test_gray_matter_maps/'
        X_file_search = 'X_compact'+str(vSize)+'.mtx'
        X_nd_file_search = 'X_compact'+str(vSize)+'.npy'
        Y_file_search = 'y'+str(vSize)+'.mtx.npy'
    else:
        train_data_path = parent_data_folder+'set_train_simple/'
        test_data_path = parent_data_folder+'set_test_simple/'
        X_file_search = 'X_simple.mtx'
        X_nd_file_search = 'X_simple.npy'
        Y_file_search = 'y_simple.mtx.npy'

    idxSlice = 85
    targets_file = parent_data_folder+'targets.csv'
    X_file = parent_data_folder+X_file_search
    X_nd_file = parent_data_folder+X_nd_file_search

    Y_file = parent_data_folder+Y_file_search
    targets = [0]
    with open(targets_file, 'rb') as csvfile:
        for line in csvfile:
            targets.append(int(line))


    iters = 300
    ys = []
    X = []
    method = 'SVR pipeline'
    if (method == 'SVR pipeline' or True) and (X_nd_file_search in os.listdir(parent_data_folder) and Y_file_search in os.listdir(parent_data_folder)):
        print 'Existing X and Y ARRAY files were found!'
        ys = np.load(Y_file)
        X = np.load(X_nd_file)

    #elif method <> 'SVR pipeline' and (X_file_search in os.listdir(parent_data_folder) and Y_file_search in os.listdir(parent_data_folder)):
    #    print 'Existing X and Y files were found!'
    #    ys = np.load(Y_file)
    #    X = io.mmread(X_file)

    else:    
        for f in os.listdir(train_data_path):
            if ('train_' in f and f.endswith('.mtx') and not simple) or ('train_' in f and f.endswith('.npy') and simple):
                iters = iters -1
                #print dirpath+f
                pic_id = int(f.split('.')[0].split('_')[1])
                #segmented_img = np.load(train_data_path+'/'+f)
                #segmented_img = None
                if not simple:
                    img = io.mmread(train_data_path+'/'+f)
                    img = img.toarray()
                    img = np.reshape(img, (176, 208, 176))
                    # sum up voxels of size 8x8x8 or 4x4x4 of img
                    
                    x = smooth_img(img, vSize)
                    if X == []:
                        X = x
                    else:
                        X = np.vstack((X,x))
                    y = targets[pic_id]
                    ys.append(y)
                else:
                    segmented_img = np.load(train_data_path+'/'+f)
                    x = segmented_img
                    if X == []:
                        X = x
                    else:
                        X = np.vstack((X,x))
                    y = targets[pic_id]
                    ys.append(y)
                #new_size = segmented_img.shape[0] *segmented_img.shape[1]*segmented_img.shape[2]
                #x = np.reshape(segmented_img, (new_size,1))

                #x = x.astype(int)
                #x = sparse.coo_matrix(x)
                
                #nda_show(segmented_img[:,:,idxSlice], title=str(y))
                gc.collect()
                print 'iters = ', iters
                if iters < 0:
                    break
        print 'Saving X...'
        if not simple:
            np.save(X_nd_file, X)
        else:
            np.save(X_nd_file, X)      
        #io.mmwrite(X_file, X)
        np.save(Y_file, ys) 

    print 'X has size: ', X.shape
    try:
        print 'ys has size: ', ys.shape
    except:
        print 'ys has size: ', len(ys)



    # Dimension reduction
    from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
    from sklearn import preprocessing, decomposition
    from sklearn.decomposition import PCA
    from sklearn.svm import SVR
    from sklearn.pipeline import Pipeline
    from sklearn.grid_search import GridSearchCV
    from sklearn.ensemble import RandomForestRegressor

    # Regression model
    reg = ''
    # Remove features with too low between-subject variance e.g. nulls
    variance_threshold = VarianceThreshold(threshold=.01)
    # Normalize the data so it has 0 mean normal variance 
    scaler = preprocessing.StandardScaler()
    min_max_scaler = preprocessing.MinMaxScaler()

    #if method == 'Lasso':
    #    print 'Running ', method
    #    reg = Lasso()
    #    reg.fit(X ,ys)
    #    print("Data fitted with Lasso Regression")
    #    w = reg.coef_
    #    np.save('W_Lasso', w)
    if method == 'LassoCV':
        print 'Running ', method
        #variance_threshold = VarianceThreshold(threshold=0.01)
        lasso = LassoCV()

        reg = Pipeline([
                ('variance_threshold', variance_threshold),
                ('lasso', lasso)])
        reg.fit(X ,ys)
        print("Data fitted with Lasso CV Regression")
        #w = reg.coef_
        #np.save('W_LassoCV', w)

    if method == 'SVR pipeline':
        print 'Running ', method
        #X = X.toarray()
        #print 'Converted sparse to dense nd array'
        
        #variance_threshold = VarianceThreshold(threshold=.01)
        # Here we use a classical univariate feature selection: removes all but the k highest scoring features    
        #feature_selection = SelectKBest(f_regression, k=2000)
        # ('feature_selection', feature_selection),
        
        # PCA
        #pca = PCA(n_components=1000)


        # SVM regression
        svrLinear = SVR(kernel='linear', C=1e-4)
        #svrPloy2 = SVR(kernel='poly', degree=2)
        #svrSigmoid = SVR(kernel='sigmoid')
        svrRBF = SVR()
        svr = svrLinear
        #rForest = RandomForestRegressor()

        regs = [svrLinear]
        Cs = [1e-3, 1e-2]
        #gammas = [1e-8, 1e-7, 1e-6]

        pipe = Pipeline([
                ('variance_threshold', variance_threshold),
                ('scaler', scaler),
                ('svr', svrLinear),
                ])

        params = dict(
            variance_threshold=[variance_threshold],
            scaler=[scaler],
            svr=regs,
            svr__C=Cs,
            )

        # does cross-validation with 3-fold for each combination of kernels and Cs
        reg = GridSearchCV(pipe, param_grid=params, n_jobs=4, cv = 5)
        #reg = pipe

        reg.fit(X, ys)
        print 'Data fitted with ', method
        print "Best parameters set found on development set:"
        print
        print(reg.best_params_)
        print 
        #w = reg.coef_
        #np.save('W_LassoCV', w)    


    prediction = []

    iters = 0
    test_files = sorted(os.listdir(test_data_path))
    smoothed_y_folder = parent_data_folder+'set_test_smooth'+str(vSize)+'/'
    for f in test_files:
        if ('test_' in f and f.endswith('.mtx') and not simple) or ('test_' in f and f.endswith('.npy') and simple):
            iters = iters +1
            pic_id = int(f.split('.')[0].split('_')[1])

            segmented_img = None
            if not simple:
                output_file = smoothed_y_folder + f+'.npy'
                if os.path.isfile(output_file):
                    print 'Compressed file %s found!!' %output_file
                    segmented_img = np.load(output_file)
                else:    
                    segmented_img = io.mmread(test_data_path+f)
                    if method == 'SVR pipeline' or method == 'LassoCV':
                        segmented_img = segmented_img.toarray()
                        segmented_img = np.reshape(segmented_img, (176, 208, 176))
                        segmented_img = smooth_img(segmented_img, vSize)
                        
                        #np.save(output_file, segmented_img)
            else:
               segmented_img = np.load(test_data_path+f)      

            res = reg.predict(segmented_img)
            prediction.append([pic_id, res[0]])
            gc.collect()
            print "Age prediction for image %d completed "%(pic_id)
            #print ''
            #print 'iters = ', iters



    with open(parent_data_folder+"predictions.csv","wb") as f:
        f.write(b'ID,Prediction\n')
        for pred in prediction:
            f.write(str(pred[0])+','+str(pred[1])+'\n')
    print 'Done.'



def segment_run(mode):
    parent_data_folder = ''
    import getpass
    user = getpass.getuser()
    if user == 'igor':
        parent_data_folder = '/home/igor/ML/data_1/'
    elif user == 'pesici':
        parent_data_folder = os.environ['HOME'] + '/'
    else:
        os.path.realpath(__file__)

    # Directory where the DICOM files are being stored (in this
    # case the 'MyHead' folder). 
    pathData = parent_data_folder+'set_'+mode+'/'
    pathOutput = parent_data_folder+'set_'+mode+'_gray_matter_maps/'

    if not os.path.exists(pathOutput):
        os.makedirs(pathOutput)


    # Z slice of the DICOM files to process. In the interest of
    # simplicity, segmentation will be limited to a single 2D
    # image but all processes are entirely applicable to the 3D image
    idxSlice = 55

    # int labels to assign to the segmented white and gray matter.
    # These need to be different integers but their values themselves
    # don't matter
    labelWhiteMatter = 2
    labelGrayMatter = 1
    labelWhiteMatter = 1
    labelGrayMatter = 2

    from os.path import isfile, join
    onlyfiles = [f for f in os.listdir(pathData) if isfile(join(pathData, f)) and f.endswith('.nii')]
    #onlyfiles = [f for f in os.listdir(pathData) if isfile(join(pathData, f)) and f.endswith('_1.nii')]

    print 'size onlyfiles: ', len(onlyfiles)
    #itkImages = []
    # to this only for part of images athe time
    for f in onlyfiles:
        full_f = join(pathData, f)
        ndImg = nib.load(full_f).get_data()
        ndImg = np.sum(ndImg, axis=3)       # sum upon the 4th axis so that we reduce the img to 3d
        print 'shape of image begin: ', ndImg.shape
        itkImg = SimpleITK.GetImageFromArray(ndImg)
        
        #itkImages.append(itkImg)
        visImg = itkImg[:,:,idxSlice]
        #sitk_show(visImg)
        
        # Smmoth the image (because it is noisy now)
        # https://itk.org/SimpleITKDoxygen/html/classitk_1_1simple_1_1CurvatureFlowImageFilter.html#details
        imgSmooth = SimpleITK.CurvatureFlow(image1=itkImg,
                                        timeStep=0.125,
                                        numberOfIterations=5)
        visImg = imgSmooth[:,:,idxSlice]
        
        
        
        #nda_show(ndaVis)
        # Segmentation with the ConnectedThreshold filter
        white_lstSeeds = [(100,75,55)]
        #visImg[118,137] = -25
        #print 'vis seed value: ', visImg[118,137]
        #sitk_show(visImg)
        imgWhiteMatter = SimpleITK.ConnectedThreshold(image1=imgSmooth, 
                                                seedList=white_lstSeeds, 
                                                lower=840, 
                                                upper=1500,
                                                replaceValue=labelWhiteMatter)
        
        # Rescale 'imgSmooth' and cast it to an integer type to match that of 'imgWhiteMatter'
        imgSmoothInt = SimpleITK.Cast(SimpleITK.RescaleIntensity(imgSmooth), imgWhiteMatter.GetPixelID())

        # Use 'LabelOverlay' to overlay 'imgSmooth' and 'imgWhiteMatter'
        #sitk_show(SimpleITK.LabelOverlay(imgSmoothInt[:,:,idxSlice], imgWhiteMatter[:,:,idxSlice]))
        
        imgWhiteMatterNoHoles = SimpleITK.VotingBinaryHoleFilling(image1=imgWhiteMatter,
                                                              radius=[2]*3,
                                                              majorityThreshold=1,
                                                              backgroundValue=0,
                                                              foregroundValue=labelWhiteMatter)
        
        # Rescale 'imgSmooth' and cast it to an integer type to match that of 'imgWhiteMatter'
        imgSmoothInt = SimpleITK.Cast(SimpleITK.RescaleIntensity(imgSmooth), imgWhiteMatterNoHoles.GetPixelID())

        # Use 'LabelOverlay' to overlay 'imgSmooth' and 'imgWhiteMatter'
        #sitk_show(SimpleITK.LabelOverlay(imgSmoothInt[:,:,idxSlice], imgWhiteMatterNoHoles[:,:,idxSlice]))
        
        # repeat for Gray matter
        gray_l_bound = 50
        gray_u_bound = 860
        lstSeeds = [(100, 70, 85), (116, 70,55), (118, 137, 55)]
        for iter_, seed in enumerate(lstSeeds):
            #print 'imgSmooth[seed]: ', imgSmooth[seed]
            if imgSmooth[seed] < gray_l_bound or imgSmooth[seed] > gray_u_bound:
                # for each of 3 dimension
                for i in range(3):
                    break_in = False
                    # for -1 and +1
                    for j in range(-5, 6):
                        new_seed = []
                        if i == 0:
                            new_seed = [seed[0]-j, seed[1], seed[2]]
                        if i == 1:
                            new_seed = [seed[0], seed[1]-j, seed[2]]
                        if i == 2:
                            new_seed = [seed[0], seed[1], seed[2]-j]    
                            
                        #print '  imgSmooth[new_seed]: ', imgSmooth[new_seed]
                        if not (imgSmooth[new_seed] < gray_l_bound or imgSmooth[new_seed] > gray_u_bound):
                            lstSeeds[iter_] = new_seed
                            break_in = True
                            #print '    breaking... imgSmooth[new_seed] = ', imgSmooth[new_seed]
                            break
                    if break_in:
                        break
                    

        imgGrayMatter = SimpleITK.ConnectedThreshold(image1=imgSmooth, 
                                                     seedList=lstSeeds, 
                                                     lower=gray_l_bound, 
                                                     upper=gray_u_bound,
                                                     replaceValue=labelGrayMatter)

        imgGrayMatterNoHoles = SimpleITK.VotingBinaryHoleFilling(image1=imgGrayMatter,
                                                                 radius=[2]*3,
                                                                 majorityThreshold=1,
                                                                 backgroundValue=0,
                                                                 foregroundValue=labelGrayMatter)

        #sitk_show(SimpleITK.LabelOverlay(imgSmoothInt[:,:,idxSlice], imgGrayMatterNoHoles[:,:,idxSlice]))
        
        imgLabels = imgWhiteMatterNoHoles | imgGrayMatterNoHoles

        #sitk_show(SimpleITK.LabelOverlay(imgSmoothInt[:,:,idxSlice], imgLabels[:,:,idxSlice]))
        
        
        # overlaps should be gray matter
        imgMask= imgWhiteMatterNoHoles/labelWhiteMatter * imgGrayMatterNoHoles/labelGrayMatter
        #imgMask= imgWhiteMatterNoHoles/labelWhiteMatter * imgGrayMatterNoHoles/labelGrayMatter
        imgWhiteMatterNoHoles -= imgMask*labelWhiteMatter

        imgLabels = imgWhiteMatterNoHoles + imgGrayMatterNoHoles

        #sitk_show(SimpleITK.LabelOverlay(imgSmoothInt[:,:,idxSlice], imgLabels[:,:,idxSlice]))
        
        endGrayMap = SimpleITK.GetArrayFromImage(imgGrayMatterNoHoles)
        #print 'shape of endGrayMap: ', endGrayMap.shape
        endImg = SimpleITK.GetArrayFromImage(imgSmoothInt)
        #print 'shape of endImg: ', endImg.shape
        #endImg = endGrayMap
        endImg[endGrayMap<>labelGrayMatter] = 0
        #nda_show(endImg[:,:,idxSlice])
        #endImg
        #print 'shape of map end: ', endImg.shape
        #print 'endImg mean: ', np.mean(endImg[np.nonzero(endImg)])
        #print 'endImg std: ', np.std(endImg[np.nonzero(endImg)])
        #print 'intensity NZ: ', np.count_nonzero(endGrayMap) / (endGrayMap.shape[0]* endGrayMap.shape[1])
        #print 'endImg min: ', np.min(endImg[np.nonzero(endImg)])
        #print 'endImg max: ', np.max(endImg)
        full_f = join(pathOutput, f)
        if np.nonzero(endImg) == 0:
            print 'We have an ERROR for ', full_f
        #np.save(full_f, endImg)
        #print 'endImg saved to: ', full_f

        # truncate image, resize it to row vector and save it as sparse
        #endImg = endImg[rmin-offset:rmax+offset,cmin-offset:cmax+offset,zmin-offset:zmax+offset]
        print 'old shape: '
        print 'shape[0]: ', endImg.shape[0]
        print 'shape[1]: ', endImg.shape[1]
        print 'shape[2]: ', endImg.shape[2]
        new_size = endImg.shape[0] *endImg.shape[1]*endImg.shape[2]
        x = np.reshape(endImg, (1, new_size))
        x = x.astype(int)
        x = sparse.coo_matrix(x)
        print 'shape of image end: ', x.shape
        io.mmwrite(full_f, x)
        print 'endImg saved to: ', full_f
        gc.collect()


segment_run('test')
segment_run('train')
reg_run()
