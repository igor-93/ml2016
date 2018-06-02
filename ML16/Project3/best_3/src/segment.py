import os
import numpy as np
import gc
import re
import inspect
import SimpleITK
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import sparse, io
from os.path import isfile, join
#%pylab inline

#mode = 'train'

# help function for visulazing
def sitk_show(img, title=None, margin=0.05, dpi=40 ):
    nda = SimpleITK.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda,extent=extent,interpolation=None)
    
    if title:
        plt.title(title)
    
    plt.show()
    
def nda_show(nda, title=None, margin=0.05, dpi=40 ):
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda,interpolation=None)
    
    if title:
        plt.title(title)
    
    plt.show()    

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


def run_segment(mode, parent_data_folder):
    #curr_folder = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 
    #parent_data_folder = join(curr_folder, 'data/')
    #parent_data_folder = ''
    #import getpass
    #user = getpass.getuser()
    #if user == 'dinesh':
    #    parent_data_folder = join(os.environ['HOME'], 'Desktop/ML16/Project2/Images')
    #elif user == 'pesici':
    #    parent_data_folder = os.environ['HOME'] 

    pathData = join(parent_data_folder, 'set_'+mode+'/')
    pathOutput = join(parent_data_folder, 'set_'+mode+'_segmented/')
    print 'pathData: ', pathData
    print 'pathOutput: ', pathOutput

    # Z slice of the DICOM files to process. In the interest of
    # simplicity, segmentation will be limited to a single 2D
    # image but all processes are entirely applicable to the 3D image
    idxSlice = 55

    # int labels to assign to the segmented white and gray matter.
    # These need to be different integers but their values themselves
    # don't matter
    labelWhiteMatter = 2
    labelGrayMatter = 1

    onlyfiles = [f for f in os.listdir(pathData) if isfile(join(pathData, f)) and f.endswith('.nii')]
    #onlyfiles = [f for f in os.listdir(pathData) if isfile(join(pathData, f)) and f.endswith('_1.nii')]
    sort_nicely(onlyfiles)  

    print 'Files found: ', len(onlyfiles)
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
        
        white_l_bound = 800
        white_u_bound = 4500
        white_lstSeeds = [(100,75,55)]
        for iter_, seed in enumerate(white_lstSeeds):
            print 'imgSmooth[seed]: ', imgSmooth[seed]
            if imgSmooth[seed] < white_l_bound or imgSmooth[seed] > white_u_bound:
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
                        if not (imgSmooth[new_seed] < white_l_bound or imgSmooth[new_seed] > white_u_bound):
                            white_lstSeeds[iter_] = new_seed
                            break_in = True
                            #print '    breaking... imgSmooth[new_seed] = ', imgSmooth[new_seed]
                            break
                    if break_in:
                        break
        
        #visImg[118,137] = -25
        #print 'vis seed value: ', visImg[100,75]
        #sitk_show(visImg)
        # Segmentation with the ConnectedThreshold filter
        imgWhiteMatter = SimpleITK.ConnectedThreshold(image1=imgSmooth, 
                                                seedList=white_lstSeeds, 
                                                lower=white_l_bound, 
                                                upper=white_u_bound,
                                                replaceValue=labelWhiteMatter)
        
        # Rescale 'imgSmooth' and cast it to an integer type to match that of 'imgWhiteMatter'
        imgSmoothInt = SimpleITK.Cast(SimpleITK.RescaleIntensity(imgSmooth), imgWhiteMatter.GetPixelID())

        # Use 'LabelOverlay' to overlay 'imgSmooth' and 'imgWhiteMatter'
        #sitk_show(SimpleITK.LabelOverlay(imgSmoothInt[:,:,idxSlice], imgWhiteMatter[:,:,idxSlice]))
        
        imgWhiteMatterNoHoles = SimpleITK.VotingBinaryHoleFilling(image1=imgWhiteMatter,
                                                              radius=[3]*3,
                                                              majorityThreshold=1,
                                                              backgroundValue=0,
                                                              foregroundValue=labelWhiteMatter)
        
        # Rescale 'imgSmooth' and cast it to an integer type to match that of 'imgWhiteMatter'
        imgSmoothInt = SimpleITK.Cast(SimpleITK.RescaleIntensity(imgSmooth), imgWhiteMatterNoHoles.GetPixelID())

        # Use 'LabelOverlay' to overlay 'imgSmooth' and 'imgWhiteMatter'
        #sitk_show(SimpleITK.LabelOverlay(imgSmoothInt[:,:,idxSlice], imgWhiteMatterNoHoles[:,:,idxSlice]))
        
        # repeat for Gray matter
        gray_l_bound = 30
        gray_u_bound = 860
        lstSeeds = [(100, 70, 85), (116, 70,55), (118, 137, 55)]
        for iter_, seed in enumerate(lstSeeds):

            if imgSmooth[seed] < gray_l_bound or imgSmooth[seed] > gray_u_bound:
                # for each of 3 dimension
                for i in range(3):
                    break_in = False

                    for j in range(-5, 6):
                        new_seed = []
                        if i == 0:
                            new_seed = [seed[0]-j, seed[1], seed[2]]
                        if i == 1:
                            new_seed = [seed[0], seed[1]-j, seed[2]]
                        if i == 2:
                            new_seed = [seed[0], seed[1], seed[2]-j]    
                        
                        if not (imgSmooth[new_seed] < gray_l_bound or imgSmooth[new_seed] > gray_u_bound):
                            lstSeeds[iter_] = new_seed
                            break_in = True
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
        imgWhiteMatterNoHoles -= imgMask*labelWhiteMatter

        imgLabels = imgWhiteMatterNoHoles + imgGrayMatterNoHoles

        #sitk_show(SimpleITK.LabelOverlay(imgSmoothInt[:,:,idxSlice], imgLabels[:,:,idxSlice]))
        
        endGrayMap = SimpleITK.GetArrayFromImage(imgGrayMatterNoHoles)
        endWhiteMap = SimpleITK.GetArrayFromImage(imgWhiteMatterNoHoles)
        endImgG = SimpleITK.GetArrayFromImage(imgSmooth)
        endImgW = SimpleITK.GetArrayFromImage(imgSmooth)
        endImgG[endGrayMap<>labelGrayMatter] = 0
        endImgW[endWhiteMap<>labelWhiteMatter] = 0

        #print 'White min: ', np.min(endImgW)
        #print 'White mean: ', np.mean(endImgW)
        #print 'White max: ', np.max(endImgW)
        #print 'Gray min: ', np.min(endImgG)
        #print 'Gray mean: ', np.mean(endImgG)
        #print 'Gray max: ', np.max(endImgG)
        f_white = 'w_'+f
        f_gray  = 'g_'+f 
        #endImgW = endImgW.astype(int)
        #print 'shape of endImgW: ', endImgW.shape
        #endImgW = sparse.coo_matrix(endImgW) 
        #endImgG = endImgG.astype(int)
        #endImgG = sparse.coo_matrix(endImgG)   


        full_f_white = join(pathOutput, f_white)
        full_f_gray = join(pathOutput, f_gray)

        if not os.path.exists(pathOutput):
            os.makedirs(pathOutput)   

        if np.nonzero(endImgG) == 0:
            print 'We have an ERROR for ', full_f_white
        elif np.nonzero(endImgW) == 0:
            print 'We have an ERROR for ', full_f_gray    
        else:    
            np.save(full_f_white, endImgW)
            #io.mmwrite(full_f_white, endImgW)
            print 'endImgW saved to: ', full_f_white
            np.save(full_f_gray, endImgG)
            #io.mmwrite(full_f_gray, endImgG)
            print 'endImgG saved to: ', full_f_gray

        gc.collect()
