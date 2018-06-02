Team name: synapse
Members: Igor Pesic, Shobhit Jain, Dinesh Acharya



Description:

1. Segmentation:
	In this step we segment the brain in 2 parts: gray and white matter.
	We set the intensities of the white matter voxel to zero and leave the gray matter voxels as they are.
	At the end we save the image as row vector. This approach is encouraged by lots of papers that 
	have observed the changes in the gray matter with aging.


2. Regression
	We load the row vector of each segmented image and reshape it to the original size.
	We reduce the size of the image in the way that a 8x8x8 cube becomes a voxel which holds the count of 
	voxels of gray matter within the cube. This way we meassure the amout of gray matter in every part of the brain.
	The value six was chosen based on grid search. The algorith was also run with values 2, 4 and 16.
	After that we reshape the image to row vector and stack it to
	matrix X which is then given to the following pipeline:

		1. apply variance threshold with threshold=0.01. This way we remove all "zero-cubes" and any other that do not change.
		2. scale the features so that each one of them is normal distributed. This is necessary in order to run SVM in a meaningful way.
		3. run Support Vector Machine regression

	This pipeline is then validated with grid search for C parameter [0.001 and 0.01] and cross validation with kFold 
	where k=5. The chosen parameter at the end was C=0.01.

