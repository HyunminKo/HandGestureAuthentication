import cv2
import os

fileNames = os.listdir('./30/')
fileNames.sort()
for fileName in fileNames:
    print(fileName)
img = cv2.imread('./30/00010.jpg')
grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(grayImg,127,255,0)

image, contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
image = cv2.drawContours(img, contours, -1, (0,255,0), 3)

cv2.imshow('image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()