import os
import torch,torchvision
import torch.nn as nn

class CatNet(nn.Module):
    def __init__(self):
        super(CatNet,self).__init__()
        self.backbone = torchvision.models.resnet50()
        self.head = nn.Sequential(
            nn.Linear(1000,150),
            nn.ReLU(),
            nn.Linear(150,36),
            nn.ReLU(),
            nn.Linear(36,18)
        )
    def forward(self,x):
        x = self.backbone(x)
        x = self.head(x)
        return x

if __name__=='__main__':
    img = torch.ones((16,3,512,512))
    model = CatNet()
    out = model(img)
    print(out.shape)