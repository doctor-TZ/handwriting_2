import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
# import time
from dataset_create import get_dataLoaders

data = np.load('dataset.npz')
label = data['labels']
pixels = data['pixels']

# print(label)
# print(pixels)
print(label.shape)
print(pixels.shape)

plt.figure(figsize=(5, 5)) 
for i in range(150):
    pixels2 = pixels[i,:]
    image = pixels2.reshape(28,28)
    plt.imshow(image,cmap='gray')
    plt.axis('off')
    plt.draw()
    plt.pause(0.5)
     


# label = data['labels']
# print(label.shape)
# pixels = data['pixels']
# print(pixels.shape)
# row_index = 500
# pixel_values = data_file.iloc[row_index,1:].values

# image = pixel_values.reshape(28,28)
# plt.imshow(image,cmap='gray')
# plt.axis('off')
# plt.show()

plt.figure(figsize=(5, 5)) 
data_file = pd.read_csv('dataset.csv')

for i in range(150):
    pixel_values = data_file.iloc[i,1:].values
    label = data_file.iloc[i,0]
    image = pixel_values.reshape(28,28)
    plt.imshow(image,cmap='gray')
    plt.title(f'Index: {i}, Label: {label}')
    plt.axis('off')
    plt.draw()
    plt.pause(0.3)



# for i in range(500,600):
#     pixel_values = data[1,1:]
#     label = data[i,0]
    


# train_loader,val_loader = get_dataLoaders('dataset.csv',64)

# plt.figure()
# for batch in train_loader:
#     images,label = batch
#     print(images.shape)
#     print(label.shape)

#     for i in range(images.shape[0]):
#         img = images[i].squeeze()
#         plt.imshow(img,cmap='gray')
#         plt.axis('off')
#         plt.title(f'Label:{label[i]}')
#         plt.draw()
#         plt.pause(1)
        
        
        

    
    






