import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from dataset import DsData


"""
This script extracts features from trained models
It is not feeding the last fc layer of network and taking 1x512 feature vector for each crop
"""
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    all_dataset = DsData(dataset_selector="all")
    all_data_loader = DataLoader(all_dataset, batch_size=256, shuffle=False, num_workers=12)

    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 9)
    state_dict = torch.load("models_fs/resnet_18_epoch19_loss_0.59_acc_81.70.pth", map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.to(device=device)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feat_vectors = {}
    with torch.no_grad():
        pbar = tqdm(enumerate(all_data_loader), total=len(all_data_loader))
        for test_img_no, test_data in pbar:
            images, labels, image_paths = test_data
            images = images.to(device)
            outputs = feature_extractor(images)
            for image_path, feat_vector in zip(image_paths, outputs):
                feat_vectors[image_path] = feat_vector.flatten()

    torch.save(feat_vectors, "FeatureVectors.pth")
