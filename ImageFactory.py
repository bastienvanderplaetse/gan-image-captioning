import torch
from torchvision import transforms
from ImageEncoder import ImageEncoder

import numpy as np

class ImageFactory(object):
    def __init__(self, resize=None, crop=None):

        self.cnn_encoder = ImageEncoder(
            cnn_type="resnet50",
            pretrained=True)

        self.cnn_encoder.setup(layer="avgpool",
                          dropout=0.,
                          pool=None)

        self.output_size = self.cnn_encoder.get_output_shape()[1]
        self.cnn = self.cnn_encoder.get()
        self.cnn.cuda()


        _transforms = []
        if resize is not None:
            _transforms.append(transforms.Resize(resize))
        if crop is not None:
            _transforms.append(transforms.CenterCrop(crop))
        _transforms.append(transforms.ToTensor())
        _transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(_transforms)


    def get_features(self, img):
        img = self.transform(img).cuda()
        return self.cnn(img.unsqueeze(0)).cpu().data.numpy().squeeze().astype(np.float32)
