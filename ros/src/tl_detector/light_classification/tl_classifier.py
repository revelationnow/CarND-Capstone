import numpy as np
import os
import cv2
from PIL import Image
from keras.models import load_model
from keras.layers import *
from keras.preprocessing.image import load_img, img_to_array
from styx_msgs.msg import TrafficLight

import threading as thr
import tensorflow as tf
'''
#from flask import Flask
#app = Flask(__name__)
th = thr.Thread(target=t_thread)
th.start()
th.join()

th2 = thr.Thread(target=t_thread)
th2.start()
th2.join()

th3 = thr.Thread(target=t_thread)
th3.start()
th3.join()
'''
class TLClassifier(object):

    def __init__(self,mod):
        #TODO load classifier
        self.model = load_model(mod)
        self.model._make_predict_function() 
        self.graph = tf.get_default_graph()


        return

    def get_classification(self, image):
        self.image = image    
        color1 = self.find_color()    
        color2 = self.get_prediction()

        if color1 == TrafficLight.RED or color2 == TrafficLight.RED:
            return  TrafficLight.RED
        elif color1 == TrafficLight.GREEN or color2 ==TrafficLight.GREEN:
            return  TrafficLight.GREEN
        elif color1 < TrafficLight.YELLOW and color2 > TrafficLight.YELLOW:
            return  TrafficLight.YELLOW

        return TrafficLight.UNKNOWN


    def reshape_image(self,image): 
        x = img_to_array(np.resize(image,(64,64,3)))
        return x[None,:]

    def find_color(self):

        for color in [TrafficLight.RED,TrafficLight.GREEN,TrafficLight.YELLOW]:
            if color == TrafficLight.RED:
                lower_mask = np.array([0,100,100],dtype = np.uint8)
                upper_mask = np.array([22,170,150],dtype = np.uint8)
                lower_mask1 = np.array([0,100,100],dtype = np.uint8)
                upper_mask1 = np.array([22,170,150],dtype = np.uint8)

            elif color == TrafficLight.GREEN:
                lower_mask = np.array([160,100,100],dtype = np.uint8)
                upper_mask = np.array([179,255,255],dtype = np.uint8)
            else:
                lower_mask = np.array([125,100,183],dtype = np.uint8)
                upper_mask = np.array([75,255,255],dtype = np.uint8)
        
            hsv_image = cv2.cvtColor(self.image,cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_image,lower_mask,upper_mask)
	    if 	color == TrafficLight.RED:
	            mask = cv2.inRange(hsv_image,lower_mask1,upper_mask1)

            residual = cv2.bitwise_and(self.image,self.image,mask=mask)
            intensity = cv2.mean(self.image,mask=mask)
            #print("This is color speaking")
            #print(intensity)
            if intensity > 200:
                return color  
 

        return TrafficLight.UNKNOWN 


    def get_prediction(self):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        image=self.image    

        with self.graph.as_default():
            prediction = self.model.predict(self.reshape_image(image))

        #print("This is CNN speaking")
        #print(prediction)

        if prediction[0][0] >  prediction[0][1] and prediction[0][0] >  prediction[0][2] :
                return TrafficLight.GREEN
        if prediction[0][1] >  prediction[0][2] and prediction[0][1] >  prediction[0][0] :
                return TrafficLight.RED
        if prediction[0][2] >  prediction[0][1] and prediction[0][2] >  prediction[0][0] :
                return TrafficLight.YELLOW
        return TrafficLight.UNKNOWN           


