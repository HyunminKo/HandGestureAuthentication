import cv2
import os

fileDirPath = './train/pos/'
fileNames = os.listdir(fileDirPath) 

countDict={}
resolutionFileNames=[]
for filename in fileNames:
	img = cv2.imread(fileDirPath+filename)
	rows,cols = img.shape[:2]
	resolutionStr = str(cols)+"x"+str(rows)
	if resolutionStr in countDict:
		countDict[resolutionStr] = countDict[resolutionStr] + 1
	else:
		countDict[resolutionStr] = 1
	if resolutionStr == '640x480':
		resolutionFileNames.append(filename)
resolutionFileNames.sort()

#print(countDict)
with open('result_data_name.txt','w') as f:
	for item in resolutionFileNames:
		f.write(item+"\n")
	
