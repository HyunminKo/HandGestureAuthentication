import numpy as np
import cv2
import os

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("x,y: ",x,y)

img_path = '../DataSample/pos/'
gt_path = '../DataSample/posGt/'
img_fileNames = os.listdir('../DataSample/pos/')
gt_fileNames = os.listdir('../DataSample/posGt/')
img_fileNames.sort()
gt_fileNames.sort()

img_fileNames.remove('.ipynb_checkpoints')
gt_fileNames.remove('.ipynb_checkpoints')
cv2.namedWindow('image')
cv2.setMouseCallback('image', on_mouse)

i = 0
for filename in gt_fileNames:
    with open(gt_path+filename,"r") as f:
        while True:
            line = f.readline()
            if not line: break
            if not line.startswith('rightHand_driver'): continue
            line_arr = line.split(" ")

            img = cv2.imread(img_path+img_fileNames[i])
            img = cv2.resize(img,(176,100),interpolation=cv2.INTER_AREA)
            rows, cols = img.shape[:2]
            line_arr[1] = round(float(line_arr[1])/3.6363,1)
            line_arr[2] = round(float(line_arr[2])/4.8,1)
            line_arr[3] = round(float(line_arr[3])/3.6363,1)
            line_arr[4] = round(float(line_arr[4])/4.8,1)
            
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