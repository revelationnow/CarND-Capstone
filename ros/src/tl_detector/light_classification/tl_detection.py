import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
import time

import pickle
from helper import get_hog_features,bin_spatial,color_hist
from filter_false import filter_false


orient = 8
pix_per_cell = 8
cell_per_block = 2
spatial_size = (16,16)
hist_bins = 32	
cspace = 'YUV'   # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
hog_channel = 'ALL'# Can be 0, 1, 2, or "ALL"

	
'''
image = mpimg.imread('./test_images/test'+ str(i) +'.jpg')

f, ax1 = plt.subplots(1, 1, figsize=(10, 10))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original test ' + str(i), fontsize=35)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()


result = process_image(image)

f, ax2 = plt.subplots(1, 1, figsize=(10, 10))
f.tight_layout()
ax2.imshow(result)
plt.savefig('./test_images/Test Pipeline'+str(time.time())+'.png')
ax2.set_title('Final Result with Car Detections', fontsize=35 )
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
'''

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel ='ALL',spatial_size=(32.32), hist_bins = 16,
                        spatial_feat=True, hist_feat=True, hog_feat=True,ystart=400,ystop=656,scale=1):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        img_features = []
        img_tosearch = mpimg.imread(file)

        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(img_tosearch)
        

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            #4) Append features to list
            img_features.append(spatial_features)
        #5) Compute histogram features if flag is set
        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            #6) Append features to list
            img_features.append(hist_features)
        if hist_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            hog_features = []
            if hog_channel == 'ALL':
                ch1 = feature_image[:,:,0]
                ch2 = feature_image[:,:,1]
                ch3 = feature_image[:,:,2]
                hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)  
                hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
                hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
                 #Extract HOG for this patch
                hog_feat1 = hog1.ravel() 
                hog_feat2 = hog2.ravel() 
                hog_feat3 = hog3.ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                ch = feature_image[:,:,hog_channel]
                hog = get_hog_features(ch, orient, pix_per_cell, cell_per_block, feature_vec=False)  
                hog_features = hog.ravel() 

            # Append the new feature vector to the features list
            img_features.append(hog_features)
    
        features.append(np.concatenate(img_features))
    # Return list of feature vectors
    return features


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_lights(img, ystart, ystop,  scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,cspace,hog_channel):
    found_lights =[]
    i=0
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]

    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif cspace == 'HLS':
	    feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS) 	
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img_tosearch)

    if scale != 1:
        imshape = feature_image.shape
        feature_image = cv2.resize(feature_image, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = feature_image[:,:,0]
    ch2 = feature_image[:,:,1]
    ch3 = feature_image[:,:,2]

    window = 64

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)    
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step


            xpos = xb*cells_per_step
    

            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            else:
                ch = feature_image[:,:,hog_channel]
                hog = get_hog_features(ch, orient, pix_per_cell, cell_per_block, feature_vec=False)  
                hog_features = hog.ravel() 
    
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            
            # Extract the image patch
            subimg = cv2.resize(feature_image[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            # Scale features and make a prediction

            scaler = np.hstack((spatial_features, hist_features, hog_features))
            X = np.array(scaler).reshape(1,-1)


            # Apply the scaler to X
            test_features = X_scaler.transform(X)                     
           
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)



            xbox_left = np.int(xleft*scale)
            ytop_draw = np.int(ytop*scale)
            win_draw = np.int(window*scale)
            
            #if i ==0:
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,255,0),2) 
                #i = 1    
            if test_prediction == 1:
                               
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(255,0,0),2) 
                found_lights.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                    
    #return found_cars, draw_img
    return found_lights

def read_image_files(rootdir):
    files=[] 
    for file in os.listdir(rootdir):
	if not file.startswith('.'):
	    filename = rootdir +'/'+ file
	    #image = cv2.imread(filename)
	    #image = cv2.resize(image, (64,64))    
	    #cv2.imwrite(filename,image)
	    files.append(filename)
    return files

'''
    for root, subFolders, files in os.walk(rootdir):
        for folder in subFolders:
	    for file in os.listdir(rootdir+'/'+folder):
	for file in os.listdir(rootdir):
		if not file.startswith('.'):
		    files.append(rootdir + '/' + folder +'/'+ file)
		    files.append(rootdir +'/'+ file)
	return files
'''

def process_image(img):
    dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]
    cspace = dist_pickle["cspace"]
    hog_channel = dist_pickle["hog_channel"]

    #648
    ystart = 0
    ystop = 400
    scale = 1   

    box_list1 = find_lights( img,ystart,ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,cspace,hog_channel)

    #96
    ystart = 0
    ystop = 500
    scale = 1.5       
    box_list2 = find_ligths( img,ystart,ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,cspace,hog_channel)
   
    #128 
    ystart = 0
    ystop = 600
    scale = 2       
    box_list3 = find_lights( img,ystart,ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,cspace,hog_channel)
    
    box_list = box_list1 + box_list2 + box_list3
    #global outbox  
    #cv2.putText(img,'Frame Box '+ str(len(box_list)),(400,100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1,(255,0,0),2)    
    out_img = filter_false(img,box_list)
    
    #for bbox in box_list1:
    #    out_img = cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    #cv2.putText(img,'Cache Box '+str(len(outbox )),(400,100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1,(255,0,0),2)    
    #cv2.putText(out_img,'Heat Box '+str(len(outbox)),(400,200), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1,(255,0,0),2)    
    
    return out_img


 
