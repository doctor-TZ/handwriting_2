import torch.nn as nn
import torch
from dataset_create import get_dataLoaders


train_loader,val_loader = get_dataLoaders('dataset.csv',32)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784,256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,1)
    
    def forward(self,x):
        x = torch.flatten(x, start_dim=1)  # 保持batch维度
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleModel()
model.to(device)
opt = torch.optim.Adam(model.parameters(),lr=0.001)

criterion = torch.nn.CrossEntropyLoss()


for epoch in range(50):
    model.train()
    for images,labels in train_loader:
        images,labels = images.to(device),labels.to(device)
        opt.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        opt.step()
    print(f"Loss:{loss:.4f}")
    


# model.eval()
# with torch.no_grad():
#     for images,labels in range(val_loader):
#         images,labels = images.to(device),labels.to(device)
#         output = model(images)
#         pred = torch.argmax(output)
#         accuracy += (pred == labels).item()
        
    



