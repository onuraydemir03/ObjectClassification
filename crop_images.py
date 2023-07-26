import glob
import json
import os

import cv2
from tqdm import tqdm


"""
This script crops the full image into detected bboxes and saves them with index postfix
Id_Index.jpg
"""

if __name__ == '__main__':
    label_files = list(sorted(glob.glob(os.path.join("./data", "labels", "*.json"))))
    os.makedirs(os.path.join("data", "crops"), exist_ok=True)
    label_dict = {}
    for label_file in tqdm(label_files):
        labels = json.load(open(label_file, "r"))
        image_path = label_file.replace('labels', 'images').replace('json', 'jpg')
        image = cv2.imread(image_path)
        for crop_no, label in enumerate(labels):
            x1 = int(label['bbox'][0]['x'])
            y1 = int(label['bbox'][0]['y'])
            w = label['bbox'][0]['width']
            h = label['bbox'][0]['height']
            x2 = int(x1 + w)
            y2 = int(y1 + h)

            left = max(0, min(x1, x2))
            right = max(x1, x2)
            top = max(0, min(y1, y2))
            bottom = max(y1, y2)
            classname = label.get('classname')
            crop = image[top: bottom, left: right]
            sample_id = os.path.splitext(os.path.basename(image_path))[0]
            try:
                cv2.imwrite(os.path.join("data", "crops", f"{sample_id}_{crop_no}.jpg"), crop)
                label_dict[f"{sample_id}_{crop_no}.jpg"] = classname
            except Exception as exc:
                print(f"Exception in {top}:{bottom}, {left}:{right} in {image_path}")
        json.dump(label_dict, open(os.path.join("data", "labels.json"), "w"), indent=1)

    json.dump(label_dict, open(os.path.join("data", "labels.json"), "w"), indent=1)
