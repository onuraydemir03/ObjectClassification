import glob
import json
import os
import os.path as op
from random import random

import cv2
import numpy as np
import requests
from tqdm import tqdm


class Dataset:
    def __init__(self, json_path: str, data_dir: str = "./data"):
        self.metadata = json.load(open(json_path, "r"))
        self.data_dir = data_dir

        self.label_save_path = op.join(self.data_dir, "labels")
        self.image_save_path = op.join(self.data_dir, "images")
        self.crop_save_path = op.join(self.data_dir, "crops")
        self.crop_label_save_path = op.join(self.data_dir, "labels.json")
        self.train_crop_label_save_path = op.join(self.data_dir, "labels_train.json")
        self.val_crop_label_save_path = op.join(self.data_dir, "labels_val.json")

    def download_images(self):
        os.makedirs(self.image_save_path, exist_ok=True)
        os.makedirs(self.label_save_path, exist_ok=True)
        pbar = tqdm(self.metadata)
        for img_info_dict in pbar:
            image_url = img_info_dict.get('asset')
            task_id = img_info_dict['tasks'][0]['taskId']
            success = self.save_image(image_url=image_url, image_name=task_id)
            if success:
                save_abs_path = op.join(self.label_save_path, f"{task_id}.json")
                if not op.exists(save_abs_path):
                    objects = []
                    for task in img_info_dict.get('tasks'):
                        for detected_obj in task.get('objects'):
                            objects.append({
                                "bbox": [detected_obj.get('bounding-box')],
                                "classname": detected_obj.get('title')
                            })
                    json.dump(objects, open(save_abs_path, "w"), indent=1)

    def save_image(self, image_url: str, image_name: str) -> bool:
        save_abs_path = op.join(self.image_save_path, f"{image_name}.jpg")
        if not op.exists(save_abs_path):
            response = requests.get(image_url)
            if response.status_code == 200:
                with open(save_abs_path, 'wb') as handler:
                    handler.write(response.content)
                return True
            return False
        return True

    def save_crops(self):
        os.makedirs(self.crop_save_path, exist_ok=True)
        label_files = list(sorted(glob.glob(op.join(self.label_save_path, "*.json"))))
        label_dict = {}
        for label_file in tqdm(label_files):
            labels = json.load(open(label_file, "r"))
            image_path = label_file.replace('labels', 'images').replace('json', 'jpg')
            sample_id = os.path.splitext(os.path.basename(image_path))[0]
            presaved_image_names = glob.glob(op.join(self.crop_save_path, f"{sample_id}*.jpg"))
            if len(presaved_image_names) != len(labels):
                image = cv2.imread(image_path)
                for crop_no, label in enumerate(labels):
                    save_abs_path = op.join(self.crop_save_path, f"{sample_id}_{crop_no}.jpg")
                    if not op.exists(save_abs_path):
                        classname = label['classname']
                        xmin = int(label['bbox'][0]['x'])
                        ymin = int(label['bbox'][0]['y'])
                        xmax = int(xmin + label['bbox'][0]['width'])
                        ymax = int(ymin + label['bbox'][0]['height'])
                        crop = self.crop_image(image=image, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
                        try:
                            cv2.imwrite(os.path.join("data", "crops", f"{sample_id}_{crop_no}.jpg"), crop)
                            label_dict[f"{sample_id}_{crop_no}.jpg"] = classname
                        except Exception as exc:
                            print(f"Exception: {exc} in {image_path} , Bbox({xmin}, {ymin}, {xmax}, {ymax}).")
        if not op.exists(self.crop_label_save_path):
            json.dump(label_dict, open(self.crop_label_save_path, "w"), indent=1)

    def crop_image(self, image: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int) -> np.ndarray:
        left = max(0, min(xmin, xmax))
        right = max(xmin, xmax)
        top = max(0, min(ymin, ymax))
        bottom = max(ymin, ymax)
        return image[top: bottom, left: right]

    def split(self, val_rate: float):
        labels = json.load(open(self.crop_label_save_path, "r"))
        keys, values = list(labels.keys()), list(labels.values())
        classnames = list(set(values))
        train_sample_idxs, val_sample_idxs = [], []
        for cls_name in classnames:
            cls_idxs = np.argwhere(np.array(values) == cls_name).flatten().tolist()
            random.shuffle(cls_idxs)
            split_point = int(len(cls_idxs) * val_rate)
            val_sample_idxs.extend(cls_idxs[:split_point])
            train_sample_idxs.extend(cls_idxs[split_point:])

        train_keys = np.array(keys)[train_sample_idxs]
        val_keys = np.array(keys)[val_sample_idxs]

        train_values = np.array(values)[train_sample_idxs]
        val_values = np.array(values)[val_sample_idxs]

        train_labels = {k: v for k, v in zip(train_keys, train_values)}
        val_labels = {k: v for k, v in zip(val_keys, val_values)}

        json.dump(train_labels, open(self.train_crop_label_save_path, "w"), indent=1)
        json.dump(val_labels, open(self.val_crop_label_save_path, "w"), indent=1)



if __name__ == '__main__':
    dataset = Dataset(json_path="./data/raw/export.json")
    # dataset.download_images()
    # dataset.save_crops()