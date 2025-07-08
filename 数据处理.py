import cv2
import pandas
import numpy as np
import csv
import glob

# path = 'captured_images/0/0.png'
# img = cv2.imread(path,cv2.COLOR_BGR2BGRA)
# _,thresh_img = cv2.threshold(img,125,255,cv2.THRESH_BINARY)



header = ['label']

for i in range(0,784):
    header.append("pixel"+str(i))

with open('dataset.csv','a') as f:
    writer = csv.writer(f)
    writer.writerow(header)

for label in range(6):
    dirList = glob.glob('captured_images/'+str(label)+'/*.png')
    for img_path in dirList:
        im= cv2.imread(img_path)
        im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        #加上高斯模糊
        im_gray = cv2.GaussianBlur(im_gray,(15,15), 0)
        roi= cv2.resize(im_gray,(28,28), interpolation=cv2.INTER_AREA)
        data=[]
        data.append(label)
        rows,cols=roi.shape
        for i in range(rows):
            for j in range(cols):
                data.append(roi[i,j])
        with open('dataset.csv','a') as f:
            writer = csv.writer(f)
            writer.writerow(data)

                