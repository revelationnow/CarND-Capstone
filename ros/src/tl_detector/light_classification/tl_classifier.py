#!/usr/bin/env python2
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

class TLClassifier(object):

    def __init__(self,mod):
        #TODO load classifier
        self.model = load_model(mod)
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()


        return

    def get_classification(self, image):
        self.image = image
        # You can use a combination of classifiers if needed
        #color1 = self.haar_prediction()
        color2 = self.get_prediction()

        '''
        if color1 == TrafficLight.RED or color2 == TrafficLight.RED:
            return TrafficLight.RED
        elif color1 == TrafficLight.GREEN or color2 ==TrafficLight.GREEN:
            return TrafficLight.GREEN
        elif color1 < TrafficLight.YELLOW and color2 > TrafficLight.YELLOW:
            return TrafficLight.YELLOW
        '''
        return color2


    def reshape_image(self,image):
        x = img_to_array(np.resize(image,(64,64,3)))
        return x[None,:]


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
            #print("red " , prediction[0][0])
            return TrafficLight.GREEN
        if prediction[0][1] >  prediction[0][2] and prediction[0][1] >  prediction[0][0] :
            #print("green",prediction[0][1])
            return TrafficLight.RED
        if prediction[0][2] >  prediction[0][1] and prediction[0][2] >  prediction[0][0] :
            #print("yellow",prediction[0][2])
            return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN

    def haar_prediction(self,image):

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


    def lbp_prediction(self,image):

        light_cascade = cv2.CascadeClassifier('lightcascade_lbp.xml')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        lights = light_cascade.detectMultiScale(gray, 50, 50)

        for (x,y,w,h) in lights:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)

        return image

    def surf_prediction(self,image):

        surf = cv2.SURF(400)

        kp, des = surf.detectAndCompute(image,None)
        surf.hessianThreshold = 50000
        kp, des = surf.detectAndCompute(img,None)

        #return len(kp)
        return cv2.drawKeypoints(image,kp,None,(255,0,0),4)

