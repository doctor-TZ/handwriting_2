import cv2
import pandas as pd
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

all_data=[]
csv_rows = []
for label in range(6):
    dirList = glob.glob('captured_images/'+str(label)+'/*.png')
    for img_path in dirList:
        im= cv2.imread(img_path)
        im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        #加上高斯模糊
        # im_gray = cv2.GaussianBlur(im_gray,(15,15), 0)
        roi= cv2.resize(im_gray,(28,28))
        roi = roi.astype(np.float32)
        all_data.append({
            'label': label,
            'pixels': roi.flatten()
        })
        csv_rows.append([label] + roi.flatten().tolist())
    with open('dataset.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_rows)  # 注意是writerows不是writerow    
        
labels = [item['label'] for item in all_data]
pixels = [item['pixels'] for item in all_data]        

np.savez('dataset.npz', labels=labels, pixels=pixels)  # 使用明确的键名
print('保存结束')



                