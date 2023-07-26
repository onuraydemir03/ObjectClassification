import json
import os

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class DsData(Dataset):
    """
    Custom dataloader class
    Args:
        dataset_selector (Optional [val, train, all]): Selects the loading json & images by one key
    """
    def __init__(self, labels_path):
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.labels = json.load(open(labels_path, "r"))
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
        image_path = os.path.join("data", "crops", self.keys[item])
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        return image, label, image_path
