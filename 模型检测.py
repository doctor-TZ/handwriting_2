import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator



#微调模型的结果不是很理想,不知道是数据集问题还是其他问题

def processImage(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {path}")
    _, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)        
    thresh_img = cv2.resize(thresh_img, (28, 28))
    digit = thresh_img.reshape(1, 28, 28, 1).astype('float32') / 255
    
    return digit

# def processImage(path):
#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise ValueError(f"无法读取图像: {path}")
    
#     # 自适应阈值（对光照不均更鲁棒）
#     thresh_img = cv2.adaptiveThreshold(
#         img, 255, 
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#         cv2.THRESH_BINARY_INV, 11, 2
#     )
    
#     # 检查是否需要反转颜色（MNIST是白底黑字）
#     if np.mean(thresh_img) > 128:  
#         thresh_img = 255 - thresh_img
    
#     thresh_img = cv2.resize(thresh_img, (28, 28))
#     digit = thresh_img.reshape(1, 28, 28, 1).astype('float32') / 255
    
#     return digit # 返回预处理前后的图像


model = load_model('mnist_model.h5')

img = processImage('test_2.png')

pred = model.predict(img)

print(f"Predicted: {np.argmax(pred)}, Confidence: {np.max(pred):.2f}")

