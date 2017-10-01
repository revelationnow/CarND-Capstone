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
from styx_msgs.msg import TrafficLight
import uuid

class TLIdentifier(object):

    def __init__(self,counter,methods,model):
        self.model = load_model(model)
        self.counter = counter
        self.methods = methods
        self.color = None
        self.lower_mask = []
        self.upper_mask = []
        self.colorl = []        
        self.colorl.append(TrafficLight.RED)    
        self.colorl.append(TrafficLight.GREEN)    
        self.colorl.append(TrafficLight.YELLOW)            
        self.lower_mask.append(np.array([0,50,50],dtype = np.uint8))
        self.upper_mask.append(np.array([10,255,255],dtype = np.uint8))
        self.lower_mask.append(np.array([58,100,100],dtype = np.uint8))
        self.upper_mask.append(np.array([100,255,255],dtype = np.uint8))

        self.lower_mask.append(np.array([25,40,100],dtype = np.uint8))
        self.upper_mask.append(np.array([45,255,255],dtype = np.uint8)) 
        self.lower_mask.append(np.array([150,50,50],dtype = np.uint8))
        self.upper_mask.append(np.array([179,255,255],dtype = np.uint8))
    def set_window(self,xstart,xstop,ystart,ystop):
        self.xstart = xstart
        self.xstop = xstop
        self.ystart = ystart
        self.ystop = ystop
        #self.last_box = ((0,0),(0,0))  

        return

    def process_image(self,img):
        self.img = img 
        box_list1 = box_list2 = box_list3 = []
        found1 = found2 = found3 = []

        if self.counter==1 or self.counter=='ALL':
            xy_window = (40,80)
            box_list1,found1 = self.find_object(xy_window)

        if self.counter==2 or self.counter=='ALL':
            xy_window = (60,120)
            box_list2,found2 = self.find_object(xy_window)

        if self.counter==3 or self.counter=='ALL':      
            xy_window = (80,160)        
            box_list3,found3 = self.find_object(xy_window)
        
        box_list = box_list1 + box_list2 + box_list3
        found = found1 + found2 + found3

        #i = 0 
	#for box in box_list:
             #new_img = cv2.rectangle(img,box[0],box[1],(0,0,255),0 )
             #plt.imshow(new_img)
             #plt.title('Final')                
             #plt.show()
             #print(found[i])
             #i = i +1  

        box = filter_false(self.img,box_list, found)

        if box[0][1] == 0:
	    #self.last_box = ((0,0),(0,0))
            return False,self.img
        #self.last_box  = box
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
        onlyfiles = [f for f in os.listdir(rootdir) if os.path.isfile(os.path.join(rootdir, f))]
        for file in onlyfiles:
            if not file.startswith('.') :
                filename = rootdir + file
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
    def find_object(self, xy_window=(40, 80), xy_overlap =(0.5, 0.5) , scale = 1):
        found_lights =[]
        found_arr = []
   
        draw_img = np.copy(self.img)

        if scale != 1:
            imshape = self.img.shape
            draw_img = cv2.resize(self.img, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        # Commented for now - introduces slight errors and need to add a flag as well for light change and doesn't work in reverse !!'
        #if self.last_box[0][1] != 0:
        #    self.xstart =  self.last_box[0][0]  - xy_window[0]
        #    self.ystart =  self.last_box[0][1]  - xy_window[1]

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

        
        residual = []
        hsv_image = cv2.cvtColor(draw_img,cv2.COLOR_RGB2HSV)
        for indx in range(3):      
            mask1 = cv2.inRange(hsv_image,self.lower_mask[indx],self.upper_mask[indx])
            if indx == 0:
                mask2 = cv2.inRange(hsv_image,self.lower_mask[indx+3],self.upper_mask[indx+3])
                mask = mask1 + mask2
            else: 
                mask = mask1
            residual.append(cv2.bitwise_and(draw_img,draw_img,mask=mask))
      
        for iy in range(ny_windows):
            for ix in range(nx_windows):
                found = 0
                
                startx = ix*stepSize_x + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = iy*stepSize_y + y_start_stop[0]
                endy = starty + xy_window[1]
                 
                #Check for colors      
                if self.methods == 1 or self.methods=='ALL':
                    for indx in range(3):                             
                        residualx = residual[indx][starty:endy,startx:endx]
                        if self.find_colors(residualx): 
                            self.color = self.colorl[indx]
                            found += 1

                            # Extract the image patch
                            subimg = draw_img[starty:endy,startx:endx]
                            src_gray = cv2.cvtColor( subimg, cv2.COLOR_BGR2GRAY )

                            #Check for edges/rectangles       
		            if  self.methods==2 or self.methods=='ALL' :
		                if self.find_contours(src_gray) :
		                    found += 1 

                            #Check for Hough circles       
		            if self.methods==3 or self.methods=='ALL':
		                if self.find_circles(src_gray) :
		                    found += 1

  			    x_start_scale = np.int(startx*scale)
			    y_start_scale = np.int(starty*scale)
			    x_end_scale = np.int(endx*scale)
			    y_end_scale = np.int(endy*scale)			      
			    found_lights.append(((x_start_scale, y_start_scale),(x_end_scale,y_end_scale)))
			    found_arr.append(found) 
					
                            if found == 3: 
				return found_lights,found_arr                       
                            break

        return found_lights,found_arr


 
    def find_contours(self,src_gray):

        bilateral_filtered_image = cv2.bilateralFilter(src_gray, 1, 10, 120)       
        edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
        _, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
            area = cv2.contourArea(contour)
            if ((len(approx) > 3) & (len(approx) < 23) & (area > 30) ):
                #contour_list.append(contour)
                return True
        #return len(contours)  #return        
        #return cv2.drawContours(image, contour_list,  -1, (255,0,0), 2)
        return False

    def find_circles(self,src_gray):

        # Reduce the noise so we avoid false circle detection
        src_gray_blur  = cv2.GaussianBlur(src_gray, (15,15), 2, 2 )

        # Apply the Hough Transform to find the circles
        circles =   cv2.HoughCircles( src_gray_blur, cv2.HOUGH_GRADIENT,1,100,param1=50,param2=30,minRadius=0,maxRadius=50) 
		
	if circles is not None:
            return True    
            #circles = np.round(circles[0, :]).astype("int")
 	    #for (x, y, r) in circles:
                #plt.imshow(cv2.circle(image,(x,y),r,(0,255,0),2))
	        #plt.title('Circles')                
                #plt.show()
                #cv2.destroyAllWindows()
        
        return False 

    def find_rectangles(self,image):

        edge_detected_image = cv2.Canny(bilateral_filtered_image,50, 150)
        '''
        filtered_image = cv2.GaussianBlur(gray, (3, 3), 0)
        
        #edge_detected_image = cv2.threshold(bilateral_filtered_image, 60, 255, cv2.THRESH_BINARY)[1]       

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
 	                cv2.line(image,w_v[i][0],w_v[i][1],(255,0,0),10)
	                cv2.line(image,h_v[j][0],h_v[j][1],(255,0,0),10)

	                # Create a "color" binary image to combine with line image
	                #color_edges = np.dstack((edge_detected_image, edge_detected_image, edge_detected_image)) 
	                #combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
	                cv2.imshow('Rectangles',image)
	                cv2.waitKey(0)
	                cv2.destroyAllWindows()
                        return True


        return  False


        '''
        kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( 7, 7 ) )
	closed = cv2.morphologyEx( edge_detected_image, cv2.MORPH_CLOSE, kernel )
        _, contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #contour_list = []
         
        for contour in contours:
  	    arc_len = cv2.arcLength( contour, True )
	    approx = cv2.approxPolyDP( contour, 0.01 * arc_len, True )
	    if cv2.contourArea( contour ) > 0 and len( approx )  > 0: 
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
		if h > 2*w and w > 40 and h > 80:
			cv2.imshow('Rectangles',image)
			cv2.waitKey(0)
		        cv2.destroyAllWindows()

                        return True
	

        
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

    def find_colors(self,residual):
            #result =  np.hstack([image, residual]) 
            residual[residual > 0 ] = 100
            if np.sum(residual) > 1000 :
                xlist = []
                ylist = []
                for y in range(residual.shape[1]):
                    for x in range(residual.shape[0]):
                        if residual[x, y][0] != 0 :
                            xlist.append(x)
                            ylist.append(y)
                yspan = max(ylist) - min(ylist)
                xspan = max(xlist) - min(xlist)
                if 3 < yspan < 30  and 3 < xspan < 30 : 
                   if 0.8 < float(yspan)/xspan  < 1.2 and np.sum(residual)/xspan > 900 : 
                        #plt.imshow(result)
                        #plt.title('Colors') 
                        #plt.show()                                      
                        #plt.imsave(str(uuid.uuid4())+".jpg",result)
                        #residual[residual > 0] = 100
		        return True
            return False 



DEBUG = False
if DEBUG == True:  
    tl = TLIdentifier(1,'ALL','./traffic_light_identifier.h5')       
    tlc = TLClassifier('./traffic_light_classifier.h5')
#    TL.resize_images('./images/unknown/','./images/resizeunknown/')
    images = tl.read_image_files('/home/student/catkin_ws/src/CarND-Capstone/ros/src/tl_detector/light_classification/images/training/misses/')
    tl.set_window(0,800,100,600)
    for image in images:
        fimage = mpimg.imread(image)
        plt.imshow(fimage )
        plt.show()
 
        fnd,result = tl.process_image(fimage)
	
        if fnd == True:
            plt.imshow(result )
            plt.show()
            state = tlc.get_classification(result)
	    if state == TrafficLight.RED:
		print("red")
	    elif state == TrafficLight.GREEN:
		print("green")
	    else:
		print("yellow")

