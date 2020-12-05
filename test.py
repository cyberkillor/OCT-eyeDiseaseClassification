import VGG
import torch
import torchvision.models as Model
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == "__main__":
    x = torch.rand(size=(8, 3, 224, 224))
    vgg = VGG.vgg16_bn(num_classes=5)
    print(vgg)
    out = vgg(x)
    print(out.size())
