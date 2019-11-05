# -*- coding: utf-8 -*-

import json

from PIL import Image


def load_json(source_file):
    """Load json file
    """
    print("Loading {}...".format(source_file))
    with open(source_file, 'r') as f:
        d = json.load(fp=f)
    print("Data Loaded.")
    return d


def write_json(dataset, target_file):
    """Write json files
    """
    print("Writing {}...".format(target_file))
    with open(target_file, "w") as f:
        json.dump(dataset, f)
    print("Data writed.")


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def default_loader(path):
    """Load Image from path and as a PIL Image
    """
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def find_classes_classifier(datas):
    """Get classes and class2index from classifier datas

    Args:
        datas: a list of infomation of different images

    Return:
        classes: a list of all class in the datas
        class2idx: a dict of pairs like <class: class_index>
    """
    classes = set()
    for data in datas:
        classes.add(data["category"])
    classes = list(classes)
    classes.sort()
    class2idx = {classes[i]: i for i in range(len(classes))}
    return classes, class2idx


def find_classes_detect(datas):
    classes = set(["background"])  # default 0 as background class
    for data in datas:
        for obj in data["objects"]:
            classes.add(obj["category"])
    classes = list(classes)
    classes.sort()
    class2idx = {classes[i]: i for i in range(len(classes))}
    return classes, class2idx


def collate_fn(batch):
    return tuple(zip(*batch))
