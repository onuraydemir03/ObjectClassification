import json
import os

import requests as requests
from tqdm import tqdm

"""
This script downloads the data from json urls, parse the entries and saves them in more usable format
"""
if __name__ == '__main__':
    dataset = json.load(open("data/raw/export.json", "r"))
    pbar = tqdm(dataset)
    for img_info_dict in pbar:
        image_url = img_info_dict.get('asset')
        response = requests.get(image_url)
        if response.status_code == 200:
            task_id = img_info_dict['tasks'][0]['taskId']
            with open(os.path.join("./data", "images", f"{task_id}.jpg"), 'wb') as handler:
                handler.write(response.content)

            parsed_dataset = {}
            for task in img_info_dict.get('tasks'):
                objects = []
                for detected_obj in task.get('objects'):
                    objects.append({
                        "bbox": [detected_obj.get('bounding-box')],
                        "classname": detected_obj.get('title')
                    })
            json.dump(objects, open(os.path.join("./data", "labels", f"{task_id}.json"), "w"), indent=1)

