import os
import random

import cv2
import numpy as np
import torch
from torch import nn

"""
This script takes feature vectors and searches to most close 12 crops
Rearranges the crop images and creates one result image from merged & resized crops
"""


if __name__ == '__main__':
    os.makedirs("Results", exist_ok=True)
    feature_vectors = torch.load("FeatureVectors.pth")
    cos = nn.CosineSimilarity()
    image_paths, feats = list(feature_vectors.keys()), list(feature_vectors.values())
    idxs = np.arange(len(feats))
    random.shuffle(idxs)
    idxs = idxs[:10]

    one_dim = 200
    for test_sample_no, test_sample_idx in enumerate(idxs):
        result_image = np.zeros((2*one_dim, 8*one_dim, 3), dtype=np.uint8)
        selected_sample_img_path, selected_sample_feat = image_paths[test_sample_idx], feats[test_sample_idx]
        scores = cos(torch.unsqueeze(selected_sample_feat, 0), torch.stack(feats))
        most_similar_idxs = torch.argsort(scores, descending=True)[:12]
        selected_image = cv2.imread(selected_sample_img_path)
        selected_image = cv2.resize(selected_image, (2*one_dim, 2*one_dim))
        result_image[:2*one_dim, :2*one_dim] = selected_image
        for idx, similar_image_idx in enumerate(most_similar_idxs):
            similar_image = cv2.imread(image_paths[similar_image_idx])
            similar_image = cv2.resize(similar_image, (one_dim, one_dim))

            text = '%.2f' % scores[similar_image_idx]
            x1, y1 = 20, 20
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            similar_image = cv2.rectangle(similar_image, (x1, y1 - 20), (x1 + w, y1), (255, 0, 255), -1)
            similar_image = cv2.putText(similar_image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if idx // 6 == 0:
                result_image[: one_dim, (2 + idx)*one_dim:(3 + idx)*one_dim] = similar_image
            else:
                idx_ = idx % 6
                result_image[one_dim:, (2 + idx_)*one_dim:(3 + idx_)*one_dim] = similar_image
        cv2.imwrite(f"Results/{'%03d' % test_sample_no}.jpg", result_image)
