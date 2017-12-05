import tensorflow as tf
import scipy.misc
import BoundingBoxModel
import cv2
import os
from subprocess import call

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

def getHandLocation(imgList):
    locationList = []
    for img in imgList:
        location = BoundingBoxModel.y.eval(feed_dict={BoundingBoxModel.x: [img], BoundingBoxModel.keep_prob: 1.0})
        locationList.append([int(location[0][0]),
                             int(location[0][1]),
                             int((location[0][0]+location[0][2])*1.2),
                             int((location[0][1]+location[0][3])*1.2)])
    return locationList