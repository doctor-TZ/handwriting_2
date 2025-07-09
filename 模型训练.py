import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
import pandas
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset_create import get_dataLoaders



class CNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 保持尺寸
            nn.BatchNorm2d(32),              # 新增BN层
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_block = nn.Sequential(
            nn.Linear(64*7*7, 256),          # 注意：padding=1后尺寸计算变化
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 6)
        )

        
        
    def forward(self,x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        return self.fc_block(x)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_Model()
train_loader, val_loader = get_dataLoaders('dataset.csv', 64)
opt = optim.Adam(model.parameters(),lr=0.0001)
crit = nn.CrossEntropyLoss()
# scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)
epochs = 50

model.to(device=device)
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for data,target in train_loader:
        data,target = data.to(device),target.to(device)
        opt.zero_grad()
        output = model(data)
        loss = crit(output,target)
        opt.step()
        running_loss += loss.item()
    # scheduler.step()
    print(f'Epoch {epoch+1}/{epochs} Loss: {running_loss/len(train_loader):.4f}')

#test
model.eval()
correct = 0
with torch.no_grad():
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()

acc = correct / len(val_loader.dataset)
print(f"Epoch {epoch+1}: Val Acc = {acc:.4f}")
        


    



























#tensorflow 实现的模型

# import tensorflow as tf

# # 加载数据
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
# x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# # 定义模型
# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Conv2D(64, 3, activation="relu"),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dense(10, activation="softmax")
# ])

# # 训练模型
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# # 保存模型
# model.save("mnist_model.h5")
# print("模型已保存为 mnist_model.h5")
        