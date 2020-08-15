# Traffic-sign-detection-and-classification-using-opencv
As part of the VDI autonomous driving challenge, the car should be able to detect five different road signs. This code is able to detect traffic signs and also classify the sign 
as one of five different signs. The implementation is divided into two parts. The first part is for detection of all road signs. This portion is written in c++ to take advantage of
the speed of c++ since it would be used for autonmous driving applications which is a hard real-time system. The system also produces some false positives which need to be ignored.
This is taken care of by the second part which is written in python. The main function of this code is to classify the trafffic sign as one of five traffic signs and also reject 
false positives. 

Detection of all road signs is done by [detect_sign](detect_sign.cpp). This [python code](Net_predict.py) is embedded in the c++ code to predict the actual road sign detected and 
also ignore false positives in the process.


#Dataset
The dataset for training the model for classifying the traffic sign is generated using augmentation. The actual images are 5 and a total of 5000 images are gerated for these five 
images for training. Augmentation was done with [augment_images](augment_images.py).


#Network and training
The network used used had 3 layers and was trained for 100 epochs. This [code](augment_images.py) is used for building and training training the model using keras.


