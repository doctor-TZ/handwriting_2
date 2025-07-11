

# model.eval()

####读取图像数据

###1)直接从图像中进行数据转换,并判断

import cv2
from torchvision import transforms
from MNIST_model import SimpleCNN
import torch

# img = cv2.imread('captured_images/0/6.png',cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img,(28,28))
# # img = img.reshape()
# # print(img.shape)

# transform = transforms.Compose([
#         transforms.ToTensor(),
#         # transforms.Normalize((0.1307,), (0.3081,))  # 使用MNIST的均值和标准差
#     ])

# img_tensor = transform(img).unsqueeze(0)


def preprocess_image(image_path):
    # 1. 读取图像为灰度图 (直接得到1通道)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. 调整大小为28x28 (MNIST标准尺寸)
    img = cv2.resize(img, (28, 28))
    
    # 3. 颜色反转（可选，MNIST是白底黑字，如果你的图片是黑底白字需要反转）
    # img = 255 - img
    
    # 4. 转换为PyTorch Tensor并归一化到[0,1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))  # 使用MNIST的均值和标准差
    ])
    
    # 5. 应用转换并添加batch维度 (变成 [1, 1, 28, 28])
    img_tensor = transform(img).unsqueeze(0)
    
    return img_tensor

model = SimpleCNN()
model.load_state_dict(torch.load('mnist_cnn_best.pth'))
model.eval()  # 设置为评估模式

# 预处理你的图片
image_path = 'captured_images/3/6.png'
input_tensor = preprocess_image(image_path)

# 如果有GPU，将数据移到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = input_tensor.to(device)
model = model.to(device)

# 进行预测
with torch.no_grad():
    output = model(input_tensor)
    prediction = output.argmax(dim=1).item()

print(f"预测结果: {prediction}")


