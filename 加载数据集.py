import torch
from torch.utils.data import Dataset
import pandas
from torch.utils.data import random_split
from torch.utils.data import DataLoader


class CustomImageDataset(Dataset):
    def __init__(self,csv,trannsformer=None):
        self.data = pandas.read_csv(csv)
        self.transform = trannsformer
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        label = self.data.iloc[idx, 0]
        pixels = self.data.iloc[idx, 1:].values.astype('float32')
        image = pixels.reshape(28, 28)
        image = image / 255.0
        if self.transform:
            image = self.transform(image)
        return image, label    
    



full_dataset = CustomImageDataset('dataset.csv')
train_ratio = 0.8
val_ratio = 0.2
train_size = int(train_ratio * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset,val_dataset = random_split(full_dataset,[train_size,val_size])

batch_size = 64

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)

print(f"训练集样本数: {len(train_dataset)}")
print(f"验证集样本数: {len(val_dataset)}")

# 查看第一个训练样本
sample_image, sample_label = next(iter(train_loader))
print(f"训练集批次形状: {sample_image.shape}")  # 应为 [batch_size, 28, 28]





        

