import json
import os.path as op
import random

import numpy as np

"""
This script splits datasets into two sets named validation & train
Validation set 20%, Train set 80% of all samples
Splitting samples made by class & class
"""

if __name__ == '__main__':
    val_data_rate = 0.2
    labels = json.load(open(op.join("data", "labels.json"), "r"))
    keys, values = list(labels.keys()), list(labels.values())
    classnames = list(set(values))
    train_sample_idxs, val_sample_idxs = [], []
    for cls_name in classnames:
        cls_idxs = np.argwhere(np.array(values) == cls_name).flatten().tolist()
        random.shuffle(cls_idxs)
        split_point = int(len(cls_idxs) * val_data_rate)
        val_sample_idxs.extend(cls_idxs[:split_point])
        train_sample_idxs.extend(cls_idxs[split_point:])

    train_keys = np.array(keys)[train_sample_idxs]
    val_keys = np.array(keys)[val_sample_idxs]

    train_values = np.array(values)[train_sample_idxs]
    val_values = np.array(values)[val_sample_idxs]

    train_labels = {k: v for k, v in zip(train_keys, train_values)}
    val_labels = {k: v for k, v in zip(val_keys, val_values)}

    json.dump(train_labels, open(op.join("./data", "labels_train.json"), "w"), indent=1)
    json.dump(val_labels, open(op.join("./data", "labels_val.json"), "w"), indent=1)
