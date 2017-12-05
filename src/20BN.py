import scipy.misc
import random
import pandas as pd
import numpy as np
import os

dataPath = '/Users/hyunminko/20bn-datasets/20bn-jester-v1'
xs = []
ys = []
nb_classes = 0
numVideoframes = 37

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

labels_csv = pd.read_csv('/Users/hyunminko/git/HandGestureAuthentication/src/jester/jester-v1-labels.csv',names=('L'), index_col=False)
train_csv = pd.read_csv('/Users/hyunminko/git/HandGestureAuthentication/src/jester/jester-v1-train.csv',sep=";",names=('dataName','labels'), index_col=False)
valid_csv = pd.read_csv('/Users/hyunminko/git/HandGestureAuthentication/src/jester/jester-v1-validation.csv',sep=";",names=('dataName','labels'), index_col=False)
test_csv = pd.read_csv('/Users/hyunminko/git/HandGestureAuthentication/src/jester/jester-v1-test.csv',names=('T'), index_col=False)

labels ={}
i = 0
for item in labels_csv['L']:
    labels[item] = i
    i+=1
nb_classes = len(labels)

for i in range(0,len(train_csv)):
    xs.append(train_csv['dataName'][i])
    ys.append(labels[train_csv['labels'][i]])
for i in range(0,len(valid_csv)):
    xs.append(valid_csv['dataName'][i])
    ys.append(labels[valid_csv['labels'][i]])

#get number of images
num_images = len(xs)

#shuffle list of images
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)
train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        imgDirPath = dataPath+"/"+train_xs[(train_batch_pointer + i) % num_train_images]+"/"
        imgNames = os.listdir(imgDirPath)
        for imgName in imgNames:
            x_out.append(scipy.misc.imresize(scipy.misc.imread(imgDirPath+imgName), [100, 176]))
            y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    
    targets = np.array([y_out]).reshape(-1)
    one_hot_targets = np.eye(nb_classes)[targets]
    y_out = one_hot_targets
    return x_out, y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        imgDirPath = dataPath+"/"+val_xs[(train_batch_pointer + i) % num_train_images]+"/"
        imgNames = os.listdir(imgDirPath)
        for imgName in imgNames:
            x_out.append(scipy.misc.imresize(scipy.misc.imread(imgDirPath+imgName), [100, 176]))
            y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    
    targets = np.array([y_out]).reshape(-1)
    one_hot_targets = np.eye(nb_classes)[targets]
    y_out = one_hot_targets
    return x_out, y_out