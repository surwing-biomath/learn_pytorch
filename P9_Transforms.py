from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# python的用法 -》 tensor数据类型
# 通过transforms.ToTensor解决两个问题
# 1、Transforms如何使用（python）
# 2、为什么需要Tensor数据类型

# absolute path C:\Users\lenovo\Desktop\learn torch\dataset\train\ants_image\0013035.jpg
# relative path dataset/train/ants_image/0013035.jpg
img_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)


writer = SummaryWriter("logs")


# 1、Transforms如何使用（python）
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

writer.close()
