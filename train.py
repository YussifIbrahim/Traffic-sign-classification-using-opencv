# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 02:30:25 2020

@author: Ucif
"""
from skimage import io 
from skimage.filters import threshold_otsu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
import cv2
import os
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.optimizers import Adam

from keras.models import Model
from keras.layers.core import Lambda, Flatten, Dense,Dropout
import glob
import numpy as np
import time
import pickle



def preprocess(path):
    #path contains aumented images of the four image categories,each in a separate folder
    classes = os.listdir(path)
    i=0
    for x in classes:
        im_path = os.path.join(path,x)
        im_s= [file for file in glob.glob(im_path+"\*.JPG")]
        
        for y in im_s:
            im = cv2.imread(y, 0)
            im = cv2.resize(im, (255,140), interpolation = cv2.INTER_AREA)
            
            ##this prevent opencv from thresholding the cross lines in overtaking image because it has already been done
            if x!= 'No_overtaking':
                im  = cv2.threshold(im , 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            im = cv2.normalize(im,None,0.0,1.0,cv2.NORM_MINMAX, dtype = cv2.CV_32FC1)
            data = np.array([im,x])
            if i==0:
                images =data
            else:
                images = np.vstack((images,data))
            i+=1
    print(i)
    return images


images =preprocess("E:\VDI_COMPETITION\SIGN_DETECTION\IMAGES\Data")

#save the preprocessed data in a pickle file
data = {"dataset": images}
f = open("images_data", "wb")
f.write(pickle.dumps(data))
f.close()

#load the pickle data which was saved above
pick = pickle.load(open("images_data", "rb"))
images = pick["dataset"] 

#unpack the image dataset into data and label
x_data,labels = zip(*images)

#split the dataset into train and test set
X_train, X_test, ytrain, ytest = train_test_split(x_data, labels, test_size=0.05,random_state=42)


#stack the elements in the dataset 
X_train =np.vstack([X_train])
X_test =np.vstack([X_test])

#reshape the dataset to the neural network input size
X_train = X_train.reshape(-1,X_train.shape[1]*X_train.shape[2])
X_test = X_test.reshape(-1,X_test.shape[1]*X_test.shape[2])


#create one hot encoding of the labels
labels = LabelBinarizer()
train_label=labels.fit_transform(ytrain)
test_label=labels.fit_transform(ytest)

#create the neural network input and output size
no_input = X_train.shape[1]
no_output = train_label.shape[1]

#create the neural network

model=Sequential()
model.add(Dense(512, input_dim=no_input, activation='tanh'))
model.add(Dense( 256, activation='relu'))
model.add(Dense( 128, activation='relu'))
model.add(Dense(no_output,activation='softmax'))

#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])

filepath = r"E:\VDI_COMPETITION\SIGN_DETECTION\weights\weights.best.tanhnew.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,save_weights_only=False, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
#start training
history = model.fit(X_train, train_label, validation_split=0.2, epochs=100 ,batch_size=32,callbacks=callbacks_list,verbose=2)
model.save('tanhnew_model.hdf5')
