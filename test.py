import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "imgs/airplane1.jpg"
image = Image.open(image_path)
print(image)
# 需要添加 image = image.convert('RGB')，因为
# png格式是四个通道，除了RGB三通道外，还有一个透明度通道。
# 所以，我们调用 image = image.convert('RGB')，保留其颜色通道
# 当然，如果图片本来就是三个颜色通道，经过此操作，不变。
# 加上这一步后，可以适应 png jpg 各种格式的图片。
image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.model(x)
        return x

# if the model is trained with GPU, and run on a CPU-only machine,
# please use torch.load with "map_location=torch.device("cpu")" to map your storage to the CPU.
model = torch.load("tudui_49_gpu.pth", map_location=torch.device("cpu"))
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval() # applicable for dropout and BatchNorm etc.
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(1))