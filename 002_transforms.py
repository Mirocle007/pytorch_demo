# -*- coding: utf-8 -*-

"""
加载数据时对数据进行预处理， 包括可能的数据增强
"""

import random

from PIL import Image
import torchvision.transforms.functional as F
import torch
from torchvision import transforms


class Resize(transforms.Resize):
    def __init__(self, size, interpolation=Image.BILINEAR, p=1):
        super(Resize, self).__init__(size, interpolation)
        self.p = p

    def __call__(self, info):
        if random.random() > self.p:
            return info
        if isinstance(info, tuple):
            img, target = info
            boxes = target["boxes"]
        else:
            return F.resize(info, self.size, interpolation=Image.BILINEAR)
        w, h = img.size
        img = F.resize(img, self.size, interpolation=Image.BILINEAR)
        new_w, new_h = img.size
        new_boxes = boxes * torch.tensor([new_w/w, new_h/h, new_w/w, new_h/h])
        target["boxes"] = new_boxes
        info = (img, target)
        return info


class RandomRotation(transforms.RandomRotation):
    def __init__(self, degrees, resample=False, expand=False, center=None, p=1):
        super(RandomRotation, self).__init__(degrees, resample, expand, center)
        self.p = p

    def __call__(self, info):
        if random.random() > self.p:
            return info
        if isinstance(info, tuple):
            assert False, "the rotation transform for object detect hasn't been finished."
            img, target = info
            boxes = target["boxes"]
        else:
            img = info

        angle = self.get_params(self.degrees)
        img = F.rotate(img, angle, self.resample, self.expand, self.center)
        if not isinstance(info, tuple):
            return img
        # To do
        info = img, target
        return info


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlip, self).__init__(p)

    def __call__(self, info):
        if random.random() > self.p:
            return info
        if isinstance(info, tuple):
            img, target = info
            boxes = target["boxes"]
        else:
            return F.hflip(info)

        w, h = img.size
        img = F.hflip(img)
        new_boxes = torch.zeros_like(boxes)
        new_boxes[:, 0] = torch.tensor(w) - boxes[:, 2]
        new_boxes[:, 1] = boxes[:, 1]
        new_boxes[:, 2] = torch.tensor(w) - boxes[:, 0]
        new_boxes[:, 3] = boxes[:, 3]
        target["boxes"] = new_boxes
        info = img, target
        return info


class RandomVerticalFlip(transforms.RandomVerticalFlip):
    def __init__(self, p=0.5):
        super(RandomVerticalFlip, self).__init__(p)

    def __call__(self, info):
        if random.random() > self.p:
            return info
        if isinstance(info, tuple):
            img, target = info
            boxes = target["boxes"]
        else:
            return F.vflip(info)

        w, h = img.size
        img = F.vflip(img)
        new_boxes = torch.zeros_like(boxes)
        new_boxes[:, 0] = boxes[:, 0]
        new_boxes[:, 1] = torch.tensor(w) - boxes[:, 3]
        new_boxes[:, 2] = boxes[:, 2]
        new_boxes[:, 3] = torch.tensor(w) - boxes[:, 1]
        target["boxes"] = new_boxes
        info = img, target
        return info


class ColorJitter(transforms.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=1):
        super(ColorJitter, self).__init__(brightness, contrast, saturation, hue)
        self.p = p

    def __call__(self, info):
        if random.random() > self.p:
            return info
        if isinstance(info, tuple):
            img, target = info
        else:
            img = info
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        img = transform(img)
        if not isinstance(info, tuple):
            return img
        info = img, target
        return info


class RandomCrop(transforms.RandomCrop):
    def __init__(self, size, padding=None, pad_if_needed=False,
                 fill=0, padding_mode='constant', p=1):
        super(RandomCrop, self).__init__(size, padding, pad_if_needed, fill, padding_mode)
        assert not padding, "padding's transform is to do"
        assert not pad_if_needed, "padding's transform is to do"
        self.p = p

    def __call__(self, info):
        if random.random() > self.p:
            return info
        if isinstance(info, tuple):
            img, target = info
        else:
            img = info
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        print(type(img))
        i, j, h, w = self.get_params(img, self.size)
        img = F.crop(img, i, j, h, w)
        if not isinstance(info, tuple):
            return img
        boxes = torch.zeros_like(target["boxes"])
        boxes[:, [0, 2]] = target["boxes"][:, [0, 2]].clamp(j, j+w) - j
        boxes[:, [1, 3]] = target["boxes"][:, [1, 3]].clamp(i, i+h) - i
        boxes_indexes = (boxes[:, 0] < boxes[:, 2]) * (boxes[:, 1] < boxes[:, 3])
        boxes = boxes[boxes_indexes]
        if not len(boxes):
            return info
        target["boxes"] = boxes
        target["labels"] = target["labels"][boxes_indexes]
        info = img, target

        return info


class CenterCrop(transforms.CenterCrop):
    def __init__(self, size, p=1):
        super(CenterCrop, self).__init__(size)
        self.p = p

    def __call__(self, info):
        if random.random() > self.p:
            return info
        if isinstance(info, tuple):
            img, target = info
        else:
            img = info
        th, tw = self.size
        w, h = img.size
        img = F.center_crop(img, self.size)
        if not isinstance(info, tuple):
            return img
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        boxes = torch.zeros_like(target["boxes"])
        boxes[:, [0, 2]] = target["boxes"][:, [0, 2]].clamp(j, j+tw) - j
        boxes[:, [1, 3]] = target["boxes"][:, [1, 3]].clamp(i, i+th) - i
        boxes_indexes = (boxes[:, 0] < boxes[:, 2]) * (boxes[:, 1] < boxes[:, 3])
        boxes = boxes[boxes_indexes]
        if not len(boxes):
            return info
        target["boxes"] = boxes
        target["labels"] = target["labels"][boxes_indexes]
        info = img, target
        return info


class ToTensor(object):
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, info):
        if isinstance(info, tuple):
            img, target = info
            img = self.to_tensor(img)
            info = img, target
            return info
        else:
            img = info
            return self.to_tensor(img)


class Normalize(transforms.Normalize):
    def __init__(self, mean, std, inplace=False):
        super(Normalize, self).__init__(mean, std, inplace)

    def __call__(self, info):
        if isinstance(info, tuple):
            img, target = info
            img = F.normalize(img, self.mean, self.std, self.inplace)
            info = img, target
            return info
        else:
            img = info
            return F.normalize(img, self.mean, self.std, self.inplace)


if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    while True:
        inputs = np.random.randn(
            random.randint(1, 500), random.randint(1, 500), random.randint(2, 3)
        ) * 255
        resize_size = random.randint(1, 500)
        crop_size = random.randint(1, min(500, inputs.shape[0], inputs.shape[1], resize_size))
        transformses = [Resize(resize_size, p=random.random()),
                        RandomRotation(degrees=random.randint(1, 360), p=random.random()),
                        RandomHorizontalFlip(p=random.random()),
                        RandomVerticalFlip(p=random.random()),
                        ColorJitter(brightness=random.random(), contrast=random.random(),
                                    saturation=random.random()),
                        RandomCrop(size=crop_size, p=random.random()),
                        CenterCrop(size=crop_size, p=random.random()),
                        # ToTensor(),
                        # Normalize([0.485, 0.456, 0.406],
                        #           [0.229, 0.224, 0.225])
                        ]
        indexes = random.sample(range(len(transformses)), random.randint(0, len(transformses)))
        transform = transforms.Compose([transformses[i] for i in indexes])
        img = Image.fromarray(inputs.astype("uint8"))
        print(f"Transforms: {transform}")
        print(f"input size: {inputs.shape}, {img.size}")
        output = transform(img)
        if isinstance(output, tuple):
            print(f"output size: {output[0].size}")
        else:
            print(f"output size: {output.size}")
