## Authors
pesici@student.ethz.ch
acharyad@student.ethz.ch
shjain@ethz.ch

## Preprocessing Keywords
graymatter, whitematter, segmentation

## Features Keywords
graymatter, histogram, bins100, cubes

## Model Keywords
SVM, linear, cv

## Description
First we segment the braing image in 2 parts: white and gray matter. The areas that belong to both are assigen to gray matter only. After that we use the gray matter of the brain. The rest of voxels are set to zeros. We separate the image into 16x16x16 cubes and make histograms out of each cube with 100 bins. These histograms are then concatenated and they form a feature vector. Based on thess features we then classify the images as following:
1. classify the age of all images
2. classify the gender of young patiances
3. classify the gender of old patiantes
4. classify the health status of old male patiantes
5. classify the health status of old female patiantes

### The classifier is the folowing pipeline:
- Variance threshold for preprocessing
- Standard scaler
- Feature selection with ANOVA (selects 10000 best features)
- PCA reduces dimensionality to number of samples for classification - 1
- Linear SVM with C = 0.1 or 1 (based on cross validation)
