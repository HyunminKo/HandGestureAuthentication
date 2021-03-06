import tensorflow as tf
import scipy.misc
import BoundingBoxModel
import cv2
import os
from subprocess import call

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

#read data.txt
xs=[]
with open("train/train_data.txt") as f:
    for line in f:
        xs.append("train/pos/" + line.split()[0])
with open("train/train_data2.txt") as f:
    for line in f:
        dirPath = "train/pos/" + line[:-1]
        files = os.listdir(dirPath)
        for file in files:
            xs.append(dirPath+"/"+file)

for i in range(len(xs)):
    full_image = scipy.misc.imread(xs[i],mode="RGB")
    image = scipy.misc.imresize(full_image, [100, 176])
    location = BoundingBoxModel.y.eval(feed_dict={BoundingBoxModel.x: [image], BoundingBoxModel.keep_prob: 1.0})
    print(location)
    cv2.rectangle(image,(int(location[0][0]),int(location[0][1])),(int((location[0][0]+location[0][2])*1.2),int((location[0][1]+location[0][3])*1.2)),(0,255,0),3)
    cv2.imshow("image",image)
    cv2.waitKey(0)

cv2.destroyAllWindows()