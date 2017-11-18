import numpy as np
import cv2
import os

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("x,y: ",x,y)

img_path = './DataSample/pos/'
gt_path = './DataSample/posGt/'
img_fileNames = os.listdir('./DataSample/pos/')
gt_fileNames = os.listdir('./DataSample/posGt/')
img_fileNames.sort()
gt_fileNames.sort()

cv2.namedWindow('image')
cv2.setMouseCallback('image', on_mouse)

i = 0
for filename in gt_fileNames:
    with open(gt_path+filename,"r") as f:
        while True:
            line = f.readline()
            if not line: break
            if line.startswith('%'): continue
            line_arr = line.split(" ")
            print(line_arr[1],line_arr[2],line_arr[3],line_arr[4])
            print(img_path+img_fileNames[i])
            img = cv2.imread(img_path+img_fileNames[i])
            rows, cols = img.shape[:2]
            cv2.rectangle(img,(int(line_arr[1]),int(line_arr[2])),(int(line_arr[1])+int(line_arr[3]),int(line_arr[2])+int(line_arr[4])),(0,255,0),3)
            cv2.imshow("image",img)
            i+=1
            cv2.waitKey(0)
            

cv2.destroyAllWindows()
'''            
for filename in img_fileNames:
    img = cv2.imread(img_path+filename)
    print(filename)
'''