import torchvision
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

# true_weights:  Parameter containing:
# tensor([[-0.0011, -0.0027,  0.0022,  ...,  0.0066, -0.0004, -0.0021],
#         [ 0.0052,  0.0020,  0.0046...,  0.0036,  0.0021,  0.0038],
#         [ 0.0063,  0.0041, -0.0004,  ..., -0.0030,  0.0011,  0.0047]],
#        requires_grad=True)
# false_weights: Parameter containing:
# tensor([[-0.0028,  0.0055,  0.0073,  ...,  0.0108,  0.0007, -0.0117],
#         [ 0.0094, -0.0156,  0.0055..., -0.0113, -0.0247,  0.0202],
#         [ 0.0010, -0.0148,  0.0057,  ...,  0.0079,  0.0129, -0.0096]],
#        requires_grad=True)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10("data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

vgg16_true.classifier.add_module("7", nn.Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)