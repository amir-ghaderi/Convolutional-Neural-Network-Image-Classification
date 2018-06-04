#Prepare Data

#load libraries
import cv2
import numpy as np
import os
from random import shuffle

#define parameters
TRAIN_DIR = "C:\\Users\\ghade\\Desktop\\CNN_master\\train"
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'dogsvscats_cnn.model'

# dog = [0,1]
# cat = [1,0]

def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat': return [1,0]
    elif word_label == 'dog':return [0,1]

def create_train_data():
    training_data = []
    count = 0
    for img in os.listdir(TRAIN_DIR):
        count += 1
        print(count)
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])

    shuffle(training_data)
    np.save('train_data.npy',training_data)
    return training_data


train_data = create_train_data()




