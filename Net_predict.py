# -# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 01:14:13 2020

@author: Ucif
"""

import numpy as np
import cv2
#import tensorflow
import keras
#import tensorflow.compat.v1 as tf
from keras import backend as K
#tf.disable_v2_behavior()


#K.clear_session()

#loads pretrained model 
network = keras.models.load_model("/home/yibrahim/new_workspace/New_Car/weights.best.hdf5", compile =False)

#define predictor function
def predictor(img,x,y,w,h):
    # a dictionary to store give the predicted output based on the key
    class_labels = {0:"no overtaking", 1:"cross", 2:"overtaking", 3:"parallel", 4:"pit"}
    
    #img = cv2.imread(img)
    #selcts the detected portion to make prediction on
    im = img[x:x+w, y:y+h]
    
    #preprocess the image
    im = cv2.resize(im, (255,140), interpolation = cv2.INTER_AREA)
    im  = cv2.threshold(im , 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    im = cv2.normalize(im,None,0.0,1.0,cv2.NORM_MINMAX, dtype = cv2.CV_32FC1)
    X_test = im.reshape(-1,im.shape[0]*im.shape[1])
    
    #make predictions
    predictions = network.predict(X_test)
    pre = np.argmax(predictions)
    
    #get the confidence which helps us discard false detections
    confidence = predictions[0,pre]
    
    #print(img)     //for debugging if python is able to read image from c++    
    if confidence > 0.8:
        label = class_labels[pre]
        print("This should work")
           
        return label
    else :
        c = "no object detected"
        return c
