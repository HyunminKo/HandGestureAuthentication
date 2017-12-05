import os
import tensorflow as tf
import BoundingBoxRun
import Data_20BN

epochs = 20
batch_size = 100

#Name of data
imgList = Data_20BN.LoadTrainBatch(batch_size)
#Get location of hand
locations = BoundingBoxRun.getHandLocation(imgList)
print(locations)
#Data preprocessing

#Video classification