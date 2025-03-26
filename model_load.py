import torch
from model_save import *
import torchvision
from torch import nn

# 方式1-》保存方式1，加载模型
model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2-》保存方式2，加载模型
# 加载模型结构
vgg16 = torchvision.models.vgg16(pretrained=False)
# 加载模型参数
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# print(vgg16)

# 陷阱1
# # ------需要额外加载部分------
# class Tudui(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model1 = Sequential(
#             Conv2d(3, 32, 5, padding=2),
#             MaxPool2d(2),
#             Conv2d(32, 32, 5, padding=2),
#             MaxPool2d(2),
#             Conv2d(32, 64, 5, padding=2),
#             MaxPool2d(2),
#             Flatten(),
#             Linear(1024, 64),
#             Linear(64, 10)
#         )
#
#     def forward(self, x):
#         x = self.model1(x)
#         return x
#
# # tudui = Tudui()
# # ------需要额外加载部分------
model = torch.load("tudui_method1.pth")
print(model)
# AttributeError: Can't get attribute 'Tudui' on <module '__main__' from 'C:\\Users\\lenovo\\Desktop\\learn torch\\model_load.py'>