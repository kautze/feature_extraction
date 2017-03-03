import cv2
import numpy as np
import matplotlib.pyplot as plt
#Why use both VGG16 AND VGG19, and not just one? 
#why not use the following lines? 
#from keras.applications.vgg19 import VGG19
#from keras.applications.vgg16 import VGG16
from vgg19 import VGG19
from vgg16 import VGG16
from keras.preprocessing import image 
from imagenet_utils import preprocess_input
from keras.models import Model #fully connected network 
import h5py
import os

# Enter source directory
source_dir = '/home/aritra/Documents/research/GE_project/data/tests/VHE_processed/'
num_labels = 2
storage_file = '/home/aritra/Documents/research/GE_project/data/features.h5'

# HDF5 file for storing the data and metadata
f = h5py.File(storage_file, 'w')

base_model = VGG19(weights='imagenet') #use weights pre-trained on ImageNet
model = Model(input=base_model.input, output=base_model.get_layer('fc1').output) #fc1? first fully connected layer? 

base_model = VGG16(weights='imagenet')
model1 = Model(input=base_model.input, output=base_model.get_layer('fc1').output)

# LOAD IMAGE NAMES
f = open('../data/GE_images12_new.txt', 'r')
image_names = f.read().split('\n')
f.close()
vgg19 = np.zeros((1, 4096))
vgg16 = np.zeros((1, 4096))
for i in range(len(image_names)):
    img = image.load_img(source_dir+image_names[i]+'.jpg',target_size=(224,224)) #this is image size required for VGG16 and VGG19 models, single color channel (grayscale)??
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x) #what does pre-processing image data do? 

    # VGG19
    fc1_vgg19 = model.predict(x)
    vgg19 = np.vstack((vgg19,fc1_vgg19))

    # VGG16
    fc1_vgg16 = model1.predict(x)
    vgg16 = np.vstack((vgg16,fc1_vgg16))

    vgg19 = np.delete(vgg19, 0, 0)
    vgg16 = np.delete(vgg16, 0, 0)
    d = grp.create_dataset('VHE_VGG19',data=vgg19)
    d = grp.create_dataset('VHE_VGG16',data=vgg16)

f.close()