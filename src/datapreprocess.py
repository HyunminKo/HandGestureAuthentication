import cv2
import os

fileNames = os.listdir('./30/')
fileNames.sort()
for fileName in fileNames:
	print(fileName)
img = cv2.imread('./30/00001.jpg')

cv2.imshow('image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()