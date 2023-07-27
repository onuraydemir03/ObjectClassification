import json
import os.path as op

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class DsData(Dataset):
    """
    Custom dataloader class
    Args:
        labels_path (str): Loading label json path, image_names are the keys of json file
    """
    def __init__(self, labels_path):
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.labels_path = labels_path
        self.labels = json.load(open(self.labels_path, "r"))
        self.keys, self.values = list(self.labels.keys()), list(self.labels.values())
        self.classes = list(set(self.values))
        self.classes.sort()
        self.label_dict = {}
        for idx, cls in enumerate(self.classes):
            self.label_dict[cls] = idx

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        label = np.zeros(len(self.classes), dtype=int)
        label[self.label_dict[self.values[item]]] = 1
        image_path = op.join(op.dirname(self.labels_path), "crops", self.keys[item])
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        return image, label, image_path
