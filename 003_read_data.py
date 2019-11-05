# -*- coding: utf-8 -*-

from torch.utils.data import Dataset

from utils.util import load_json, default_loader, collate_fn
from utils.util import find_classes_classifier, find_classes_detect


class ClassifierData(Dataset):
    """Read data for a classfier trainer
    """

    def __init__(self, json_info_file, phase, transform=None):
        """Initialization of instance of the class

        Args:
            json_info_file: the json file with infomation of the dataset.
            phase: ”train", "val" or "test".
            transform: optional transform to be applied on a sample.
        """
        dataset = load_json(json_info_file)
        self.dataset = dataset[phase]
        self.transform = transform
        classes, class2idx = find_classes_classifier(self.dataset)
        self.classes = classes
        self.class2idx = class2idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]
        path = image["path"]
        category = image["category"]
        category = self.class2idx[category]
        img = default_loader(path)
        if self.transform:
            img = self.transform(img)
        return img, category, path


class DetectData(Dataset):
    """Read data for a object detector trainer
    """
    def __init__(self, json_info_file, phase, transform=None):
        """
        Args:
            json_info_file: the json file with infomation of the object-detection dataset.
            phase: ”train", "val" or "test".
            transform: optional transform to be applied on a sample.
        """
        dataset = load_json(json_info_file)
        self.dataset = dataset[phase]
        self.transform = transform
        classes, class2idx = find_classes_detect(self.dataset)
        self.classes = classes
        self.class2index = class2idx

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img_path = data["img_path"]
        img = Image.open(img_path).convert("RGB")
        objects = data["objects"]
        num_objs = len(objects)
        boxes = []
        labels = []
        for obj in objects:
            xmin = obj["xmin"]
            xmax = obj["xmax"]
            ymin = obj["ymin"]
            ymax = obj["ymax"]
            label = self.class2index[obj["category"]]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id

        if self.transform:
            img, target = self.transform((img, target))
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["area"] = area

        return img, target, img_path

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    import torch
    classifier_data = ClassifierData("prepared_data_info_sample1.json", "train")
    object_data = DetectData("prepared_data_info_sample2.json", "train")
    classifier_loader = torch.utils.data.DataLoader(classifier_data,
                                                    shuffle=True,
                                                    batch_size=2,
                                                    num_workers=2)
    object_loader = torch.utils.data.DataLoader(object_data,
                                                shuffle=True,
                                                batch_size=2,
                                                num_workers=2,
                                                collate_fn=collate_fn)
    for img, category, path in classifier_loader:
        print("img_size: ", img.size)
        print("category: ", category)
        print("path: ", path)

    for img, category, path in object_loader:
        print("img_size: ", img.size)
        print("category: ", target)
        print("path: ", path)
        # TODO
