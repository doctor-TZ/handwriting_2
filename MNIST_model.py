import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataset_create import get_dataLoaders
import cv2

# 定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层1: 输入1通道(灰度图), 输出32通道, 3x3卷积核
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 卷积层2: 输入32通道, 输出64通道, 3x3卷积核
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 最大池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层1: 输入64*7*7(经过两次池化后图像尺寸为7x7), 输出128维
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 全连接层2: 输出10维(MNIST有10个类别)
        self.fc2 = nn.Linear(128, 6)
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(0.5)
        # ReLU激活函数
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 第一次卷积+激活+池化
        x = self.pool(self.relu(self.conv1(x)))
        # 第二次卷积+激活+池化
        x = self.pool(self.relu(self.conv2(x)))
        # 展平特征图
        x = x.view(-1, 64 * 7 * 7)
        # 全连接层1+激活+Dropout
        x = self.dropout(self.relu(self.fc1(x)))
        # 全连接层2(输出层)
        x = self.fc2(x)
        return x

# 数据预处理
# transform = transforms.Compose([
#     transforms.ToTensor(),  # 转换为Tensor并归一化到[0,1]
#     transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
# ])

# 加载数据集


# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader,test_loader = get_dataLoaders('dataset.csv',32)

# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 测试函数
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # 累加批次损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取预测结果
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    return accuracy



# 主训练循环
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# epochs = 10
# best_accuracy = 0.0

# for epoch in range(1, epochs + 1):
#     train(model, device, train_loader, optimizer, criterion, epoch)
#     accuracy = test(model, device, test_loader, criterion)
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         torch.save(model.state_dict(), 'mnist_cnn_best.pth')

# print(f'Best accuracy: {best_accuracy:.2f}%')


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
        transforms.Normalize((0.1307,), (0.3081,))  # 使用MNIST的均值和标准差
    ])
    
    # 5. 应用转换并添加batch维度 (变成 [1, 1, 28, 28])
    img_tensor = transform(img).unsqueeze(0)
    
    return img_tensor

model = SimpleCNN()
model.load_state_dict(torch.load('mnist_cnn_best.pth'))
model.eval()  # 设置为评估模式

# 预处理你的图片
image_path = 'test_pic/test_6.png'
# image_path = 'captured_images/0/15.png'
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









    






