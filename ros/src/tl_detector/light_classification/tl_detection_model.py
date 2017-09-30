import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from tl_detection import extract_features, read_image_files
import matplotlib.image as mpimg
import os
import sys
import time

import pickle
from helper import get_hog_features,bin_spatial,color_hist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

orient = 8
pix_per_cell = 8
cell_per_block = 2
hist_bins = 32	
cspace = 'YCrCb'   # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
hog_channel = 'ALL'# Can be 0, 1, 2, or "ALL"
spatial_size = (64,64)
	    
	
print('Initiating model training ..')

lights = []
notlights =[]
spatial_feat = True
hog_feat=True 
hist_feat=True
n_predict = 100

print('Reading Traffic Light Images ..')

lights=read_image_files('./images/red')
lights+=read_image_files('./images/green')

print('Light Images available ..'+str(len(lights)))
print('Reading Non Light Images ..')

notlights=read_image_files('./images/unknown')
                                                         
print('Non Light Images available ..'+str(len(notlights)))                                                                     

sample_size=len(lights)
print('Images selected for training ..'+str(sample_size))

lights = lights[0:sample_size]
notlights = notlights[0:sample_size]

light_features = extract_features(lights, cspace=cspace, orient=orient, 
                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                hog_channel=hog_channel, spatial_size=spatial_size,spatial_feat=spatial_feat, hist_bins = hist_bins,
                    hist_feat=hist_feat, hog_feat=hog_feat)

print('Lights Feature extraction complete ..')

notlight_features = extract_features(notlights, cspace=cspace, orient=orient, 
                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                hog_channel=hog_channel, spatial_size=spatial_size, spatial_feat=spatial_feat, hist_bins = hist_bins, 
                    hist_feat=hist_feat, hog_feat=hog_feat)

print('Non Light Feature extraction complete ..')

# Create an array stack of feature vectors
X = np.vstack((light_features, notlight_features)).astype(np.float64)  
print('Final Rows and Features extracted ' + str(X.shape))

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(light_features)), np.zeros(len(notlight_features))))
print('Feature scaling complete ..with outputs'+ str(y.shape))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Data Split ..')

# Use a linear SVC 
svc = LinearSVC()
svc.fit(X_train, y_train)

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))


print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])

dist_pickle = {}
dist_pickle["svc"] = svc
dist_pickle["scaler"] = X_scaler
dist_pickle["orient"] = orient
dist_pickle["pix_per_cell"] = pix_per_cell
dist_pickle["cell_per_block"] = cell_per_block
dist_pickle["spatial_size"] = spatial_size
dist_pickle["hist_bins"] = hist_bins
dist_pickle["cspace"] = cspace
dist_pickle["hog_channel"] = hog_channel        
filename = 'svc_pickle.p'
if os.path.exists(filename):
	os.remove(filename)
pickle.dump( dist_pickle,open( filename, "wb" ))





