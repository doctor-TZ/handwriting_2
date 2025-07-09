import torch
from torch.utils.data import Dataset
import pandas
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from copy import deepcopy


class CustomImageDataset(Dataset):
    def __init__(self,csv):
        self.data = pandas.read_csv(csv)
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        label = self.data.iloc[idx, 0]
        pixels = self.data.iloc[idx, 1:].values.astype('float32')
        image = pixels.reshape(28, 28)
        image = image.astype('uint8')  
        
        return image, label    

class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)    



def create_dataset(csv_path):
    train_transforms = transforms.Compose([
    transforms.ToPILImage(),              # 将 numpy 数组转为 PIL 图像（torchvision 增强需要 PIL 格式）
    # transforms.RandomRotation(10),        # 随机旋转（-10° 到 +10°）
    # transforms.RandomHorizontalFlip(),    # 随机水平翻转（概率 50%）
    transforms.ToTensor(),                # 转为 Tensor 并自动归一化到 [0, 1]
    transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化到 [-1, 1]（可选）
    ])
    
    val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    full_dataset = CustomImageDataset(csv_path)
    train_dataset,val_dataset = random_split(full_dataset,[0.8,0.2])
    
    train_dataset = TransformSubset(train_dataset,train_transforms)
    val_dataset = TransformSubset(val_dataset, val_transforms)
    return train_dataset,val_dataset
    

def get_dataLoaders(csv_path,batchSize):
    train_dataset,val_dataset = create_dataset(csv_path=csv_path)
    train_loader = DataLoader(train_dataset,batch_size=batchSize,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=batchSize,shuffle=False)
    
    return train_loader,val_loader
    

        

