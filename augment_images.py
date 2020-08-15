# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 23:50:51 2020

@author: Ucif
"""
import cv2
import numpy as np
from skimage import io 
from skimage.transform import rotate, AffineTransform, warp
import matplotlib.pyplot as plt
import random
from skimage import img_as_ubyte
import os
from skimage.util import random_noise



    
#Lets define functions for each operation
def anticlockwise_rotation(image):
    angle= random.randint(0,80)
    return rotate(image, angle)

def clockwise_rotation(image):
    angle= random.randint(0,80)
    return rotate(image, -angle)

def h_flip(image):
    return  np.fliplr(image)

def v_flip(image):
    return np.flipud(image)

def add_noise(image):
    return random_noise(image)

def blur_image(image):
    return cv2.GaussianBlur(image, (9,9),0)


def warp_shift(image): 
    xtranslate = [0,40,60,80]
    ytranslate = [0,42,62,82]
    xtranslate = random.choice(xtranslate)
    ytranslate = random.choice(ytranslate)
    transform = AffineTransform(translation=(xtranslate,ytranslate))  #chose x,y values according to your convinience
    warp_image = warp(image, transform, mode="constant")
    return warp_image

def adjust_gamma(image):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
    gamma_choice = [0.5,1.5,2.5]
    gamma = random.choice(gamma_choice)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
    return cv2.LUT(image, table)

transformations = {'rotate anticlockwise': anticlockwise_rotation,
                      'rotate clockwise': clockwise_rotation,
                      'horizontal flip': h_flip, 
                      'vertical flip': v_flip,
                   'warp shift': warp_shift,
                   'adding noise': add_noise,
                   'blurring image':blur_image,
                  # 'gamma adjust': adjust_gamma
                 } 
method = list(transformations)
images_path= r"E:\VDI_COMPETITION\SIGN_DETECTION\IMAGES\overtaking.JPG" #path to original images


augmented_path= r"E:\VDI_COMPETITION\SIGN_DETECTION\IMAGES\overtaking" # path to store aumented images

classes = (images_path.split('\\')[-1]).split('.')[0]


#for im in os.listdir(images_path):  # read image name from folder and append its path into "images" array     
#    images.append(os.path.join(images_path,im))
#images = os.path.join(images_path)
images_to_generate=1000  #you can change this value according to your requirement
i=1                        # variable to iterate till images_to_generate

#transformation_range = list(range(0,len(transformations)))
while i<=images_to_generate:    
    #image=random.choice(images)
    original_image = io.imread(images_path)
    transformed_image=None
#     print(i)
    #n = 0       #variable to iterate till number of transformation to apply
    #transformation_count = random.choice(transformation_range) #choose random number of transformation to apply on the image
    
    
    key0 = random.choice(method) #randomly choosing method to call
    transformed_image = transformations[key0](original_image)
    #apply another method to obtain variety
    key = random.choice(method) #randomly choosing method to call
    
    while key == key0  : 
        key = random.choice(method)
    transformed_image = transformations[key](transformed_image)
       
        
    new_image_path= "%s/%s_%s.jpg" %(augmented_path,classes, i)
    transformed_image = img_as_ubyte(transformed_image)  #Convert an image to unsigned byte format, with values in [0, 255].
    transformed_image=cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB) #convert image to RGB before saving it
    cv2.imwrite(new_image_path, transformed_image) # save transformed image to path
    i =i+1
