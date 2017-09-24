from PIL import Image
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
import time
from keras.preprocessing.image import load_img, img_to_array
from filter_false import filter_false
from keras.models import load_model
from sklearn.metrics import accuracy_score
from tl_classifier import TLClassifier
class TLIdentifier(object):

    def __init__(self,counter,methods,model):
        self.model = load_model(model)
        self.counter = counter
        self.methods = methods

    def set_window(self,xstart,xstop,ystart,ystop):
        self.xstart = xstart
        self.xstop = xstop
        self.ystart = ystart
        self.ystop = ystop

        return

    def process_image(self,img):
        self.img = img
        self.scale = 1   

        box_list1 = box_list2 = box_list3 = []
        found1 = found2 = found3 = []
        circles1 = circles2 = circles3 = []

        if self.counter==1 or self.counter=='ALL':
            xy_window = (40,80)
            box_list1,found1,circles1 = self.find_object(xy_window)
        if self.counter==2 or self.counter=='ALL':
            xy_window = (60,120)
            box_list2,found2,circles2 = self.find_object(xy_window)

        if self.counter==3 or self.counter=='ALL':      
            xy_window = (80,160)        
            box_list3,found3,circle3 = self.find_object(xy_window)
        
        box_list = box_list1 + box_list2 + box_list3
        found = found1 + found2 + found3
        circles = circles1 + circles2 + circle3
        box = filter_false(self.img,box_list, found, circles)

        if box[0][1] == 0:
	    return False,self.img

        out_img = np.copy(self.img[box[0][1]:box[1][1], box[0][0]:box[1][0]])       
        return True,out_img

    def reshape_image(self,image):
        x = img_to_array(np.resize(image,(64,64,3)))
        return x[None,:]

    def reshape_array(self,data):
        image = Image.fromarray(data, 'RGB')
        return image

    def read_image_files(self,rootdir):
        files=[] 
        for file in os.listdir(rootdir):
            if not file.startswith('.'):
                filename = rootdir +'/'+ file
                files.append(filename)
        return files

    def resize_images(self,rootdir , targetdir):

        for filename in os.listdir(rootdir):
            if not filename.startswith('.'):
                  image = cv2.imread(rootdir+filename)
                  
                  imageresized = cv2.resize(image,(40,80), interpolation = cv2.INTER_AREA)
                  cv2.imwrite( targetdir + filename ,imageresized ) 

                
        return True

    def create_pos_n_neg(self):     
        for file_type in ['light','unknown']: 
            for img in os.listdir(file_type):

                if file_type == 'light':
                    line = file_type+'/'+img +'\n' # 1 0 0 50 50
                    with open('info.dat','a') as f:
                        f.write(line)
                elif file_type == 'unknown':
                    line = file_type+'/'+img+'\n'
                    with open('bg.txt','a') as f:
                        f.write(line)

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_object(self, xy_window=(35, 80), xy_overlap =(0.5, 0.5) , scale = 1):
        found_lights =[]
        found_arr = []
        circle_arr = []

        draw_img = np.copy(self.img)

        if scale != 1:
            imshape = img.shape
            draw_img = cv2.resize(self.img, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        if self.xstart and self.ystart and self.xstop and self.ystop :
            x_start_stop = [self.xstart,self.xstop]
            y_start_stop = [self.ystart,self.ystop]
        else:
            x_start_stop = [0,draw_img.shape[1]]
            y_start_stop = [0,draw_img.shape[0]]
            
        # Compute the span of the region to be searched
        spanx = x_start_stop[1] - x_start_stop[0]
        spany = y_start_stop[1] - y_start_stop[0]
        
        # Compute the number of pixels per step in x/y
        stepSize_x= int((1-xy_overlap[0])*xy_window[0])
        stepSize_y= int((1-xy_overlap[1])*xy_window[1])

        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((spanx-nx_buffer)/stepSize_x) 
        ny_windows = np.int((spany-ny_buffer)/stepSize_y) 
              
        for iy in range(ny_windows):
            for ix in range(nx_windows):
                found = 0
                rectangle = False 

                startx = ix*stepSize_x + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = iy*stepSize_y + y_start_stop[0]
                endy = starty + xy_window[1]

                # Extract the image patch
                subimg = self.img[starty:endy,startx:endx]

                #Check for Hough circles

                if self.methods == 1 or self.methods=='ALL':
                    circles = self.find_circles(subimg)
                    if circles is not None: 
                        found = len(circles)
                        circle_arr.append(circles)


                if  self.methods==2 or self.methods=='ALL' :
                    rectangles = self.find_rectangles(subimg)
                    if rectangles is True:
                        found += 1     


                if self.methods==3 or self.methods==-1:
                    subimg = subimg.astype(np.float32)/255
                    #CNN               
                    ret = self.model.predict(self.reshape_image(subimg))
                    if ret[0][0] > 0.5:
                        found += ret[0][0]

                if self.methods==4 or self.methods==-1:
                    # Also check Haar  as backup 
                    faces = self.find_haar(subimg)
                    if faces is not None:
			found += 1

                if self.methods==5 or self.methods==-1:
                    # Also check LBP  as backup 
                    found += self.find_lbp(subimg)

                x_start_scale = np.int(startx*scale)
                y_start_scale = np.int(starty*scale)
                x_end_scale = np.int(endx*scale)
                y_end_scale = np.int(endy*scale)

      
                if found > 0:
                   found_lights.append(((x_start_scale, y_start_scale),(x_end_scale,y_end_scale)))
                   found_arr.append(found) 

                #if found > 50: 
                   #return found_lights,found_arr,circle_arr
                                           
        return found_lights,found_arr,circle_arr


 
    def find_contours(self,image):

        bilateral_filtered_image = cv2.bilateralFilter(image, 5, 175, 175)
        edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
        _, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_list = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
            area = cv2.contourArea(contour)
            if ((len(approx) > 8) & (len(approx) < 23) & (area > 30) ):
                contour_list.append(contour)
        
        #return len(contours)        
        return cv2.drawContours(image, contour_list,  -1, (255,0,0), 2)


    def find_circles(self,image):

        src_gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )

        # Reduce the noise so we avoid false circle detection
        src_gray  = cv2.GaussianBlur(src_gray, (15,15), 2, 2 )

        # Apply the Hough Transform to find the circles
        circles =   cv2.HoughCircles( src_gray, cv2.HOUGH_GRADIENT,1,100,param1=50,param2=30,minRadius=0,maxRadius=50) 
	'''	
	if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
 	    for (x, y, r) in circles:
                cv2.imshow('Circles',cv2.circle(image,(x,y),r,(0,255,0),2))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        '''
        return circles 

    def find_rectangles(self,image):

 	gray = cv2.cvtColor( image,  cv2.COLOR_BGR2GRAY )

        #bilateral_filtered_image = cv2.bilateralFilter(gray, 1, 10, 120)       
        filtered_image = cv2.GaussianBlur(gray, (3, 3), 0)

        #edge_detected_image = cv2.threshold(bilateral_filtered_image, 60, 255, cv2.THRESH_BINARY)[1]       
        edge_detected_image = cv2.Canny(filtered_image,50, 150)

        rho = 1
	theta = np.pi/180
	threshold = 1
	min_line_length = 30
	max_line_gap = 5
	line_image = np.copy(image)*0 #creating a blank to draw lines on

	# Run Hough on edge detected image
	lines = cv2.HoughLinesP(edge_detected_image, rho, theta, threshold, np.array([]),
		                    min_line_length, max_line_gap)
	try:
            lines[0]
	except:
            return False
        h_v = w_v = []
        h = w =0 
	for line in lines:
            for x1,y1,x2,y2 in line:
		#All good lines
                if (abs(y1 - y2) > 30 and abs(x2-x1) == 0) :
                    h_v.append(((x1,y1),(x2,y2)))  
                if (abs(y1 - y2) == 0 and abs(x2-x1) > 10) :
                    w_v.append(((x1,y1),(x2,y2)))  


        # At least one intersection between verticle and horizontal lines
        for i in range(len(w_v)):
            for j in range(len(h_v)):
                if w_v[i][0] == h_v[j][0] or w_v[i][0] == h_v[j][1] or w_v[i][1] == h_v[j][0] or w_v[i][0] == h_v[j][1]:
                    h = abs(h_v[j][0][1] - w_v[j][1][1])
                    w = abs(w_v[i][0][0] - w_v[i][1][0])

                    if h > 2*w:
                        return True
	                #cv2.line(line_image,w_v[i][0],w_v[i][1],(255,0,0),10)
	                #cv2.line(line_image,h_v[j][0],h_v[j][1],(255,0,0),10)

	                # Create a "color" binary image to combine with line image
	                #color_edges = np.dstack((edge_detected_image, edge_detected_image, edge_detected_image)) 
	                #combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
	                #cv2.imshow('Rectangles',combo)
	                #cv2.waitKey(0)
	                #cv2.destroyAllWindows()


        return  False


        '''
        kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( 7, 7 ) )
	closed = cv2.morphologyEx( edge_detected_image, cv2.MORPH_CLOSE, kernel )
        _, contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #contour_list = []
         
        for contour in contours:
  	    #arc_len = cv2.arcLength( contour, True )
	    #approx = cv2.approxPolyDP( contour, 0.01 * arc_len, True )
	    if cv2.contourArea( contour ) > 0 : #and len( approx )  > 0
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
		if h > 2*w and w > 40 and h > 80:
			cv2.imshow('Rectangles',image)
			cv2.waitKey(0)
		        cv2.destroyAllWindows()

                return True
	

        '''
        '''
            #approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
            #area = cv2.contourArea(contour)
            #if ((len(approx) > 15) & (len(approx) < 25) & (area > 200)  ) or (len(approx) == 4 ):
                #contour_list.append(contour)
        
        #areas = [cv2.contourArea(c) for c in contours]
        #max_index = np.argmax(areas)
        #cnt=contours[max_index]
	#x,y,w,h = cv2.boundingRect(cnt)
        #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        #cv2.imshow('Rectangles',cv2.drawContours(image, contour_list,  -1, (255,0,0), 2))
        '''


    def find_haar(self,image):

        #light_cascade = cv2.CascadeClassifier('lightcascade_haar.xml')
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        #eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #lights = light_cascade.detectMultiScale(gray, 50, 50)
	gray = np.uint8(gray)
        lights = face_cascade.detectMultiScale(gray, 1.3, 5) 
       
        #for (x,y,w,h) in lights:
            #cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)

        return lights    

  
    def find_lbp(self,image):

        light_cascade = cv2.CascadeClassifier('lightcascade_lbp.xml')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        lights = light_cascade.detectMultiScale(gray, 50, 50)
       
        for (x,y,w,h) in lights:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)

        return image 

    def find_surf(self,image):

        surf = cv2.SURF(400)

        kp, des = surf.detectAndCompute(image,None)
        surf.hessianThreshold = 50000
        kp, des = surf.detectAndCompute(img,None)

        #return len(kp)
        return cv2.drawKeypoints(image,kp,None,(255,0,0),4) 

DEBUG = False
if DEBUG == True:  
    tl = TLIdentifier('ALL',1,'./traffic_light_identifier.h5')       
    tlc = TLClassifier('./traffic_light_classifier.h5')
#    TL.resize_images('./images/unknown/','./images/resizeunknown/')
    images = tl.read_image_files('/home/student/images')
    tl.set_window(100,700,100,600)

    for image in images:
        fimage = mpimg.imread(image)
        cv2.imshow('Looking for Traffic Light', fimage )
        cv2.waitKey(0)
        cv2.destroyAllWindows()  

        fnd,result = tl.process_image(fimage)
	
        if fnd == True:
            cv2.imshow('Search Result', result )
            cv2.waitKey(0)
            cv2.destroyAllWindows()
		
