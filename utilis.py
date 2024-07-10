import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

def getName(filePath):
    return filePath.split('\\')[-1]

def importDataInfo(path):
    columns = ['center','left','right','steering','throttle','brake','speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'),names=columns)
    # print(getName(data['center'][0]))
    data['center'] = data['center'].apply(getName)
    # print(data.shape[0])
    return data

def balanceData(data,display=True):
    nBins = 31
    samplesPerBin = 250
    hist, bins = np.histogram(data['steering'],nBins)
    # print(bins)
    center = (bins[:-1]+bins[1:])*0.5
    if display:
        # print(center)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1,1),(samplesPerBin,samplesPerBin))
        plt.show()
    
    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data['steering'])):
            if data['steering'][i] >= bins[j] and data['steering'][i]<=bins[j+1]:
                binDataList.append(i)
        
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)
    # print(len(removeIndexList))
    data.drop(data.index[removeIndexList],inplace = True)

    # print(len(data))
    if display:
        hist, _ = np.histogram(data['steering'],nBins)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1,1),(samplesPerBin,samplesPerBin))
        plt.show()

def loadData(path,data):
    imagesPath = []
    steering = []

    for i in range(len(data)):
        indexedData = data.iloc[i]
        # print(indexedData)
        imagesPath.append(os.path.join(path, 'IMG', indexedData[0]))
        steering.append(float(indexedData[3]))
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)
    return imagesPath, steering

def augementImage(imgPath, steering):
    img = mpimg.imread(imgPath)

    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={'x':(-0.1,0.1), 'y':(-0.1,0.1)})
        img = pan.augment_image(img)

    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)

    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.4,1.0))
        img = brightness.augment_image(img)

    if np.random.rand() < 0.5:
        img = cv2.flip(img,1)
        steering=-steering
    return img,steering


def preProcessing(image):
    if isinstance(image, str):
        img = mpimg.imread(image)
    else:
        img = image
    
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

# imgRe  = preProcessing('test.jpg')
# plt.imshow(imgRe)
# plt.show()

def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize):
            index = random.randint(0,len(imagesPath)-1)
            if trainFlag:
                img , steering = augementImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering=steeringList[index]

            img=preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        
        yield (np.asarray(imgBatch), np.asarray(steeringBatch))


def createModel():
    model = Sequential()

    model.add(Convolution2D(24,(5,5),(2,2),input_shape=(66,200,3),activation='elu'))
    model.add(Convolution2D(36,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(48,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))

    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50,activation='elu'))
    model.add(Dense(10,activation='elu'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='MeanSquaredError')
    return model


    

