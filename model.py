import os
import random

import cv2
import numpy as np
import torch.utils.data
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import wandb
import os.path as op


class Model:
    """
    Custom model class that takes ResNet18 and make processes with it
    This class has its own functionality
        - train
        - feature extraction
        - similarity matching
    """
    def __init__(self, scratch: bool = False, num_classes: int = 9):
        """
        Args:
            scratch (bool): Model creation will be made from 0 or pretrained
            num_classes (int): Number of classes that will be classified
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scratch = scratch
        self.num_classes = num_classes
        self.model = models.resnet18(pretrained=True)
        if not self.scratch:
            for param in self.model.parameters():
                param.requires_grad = False
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)
        self.model.to(self.device)

    def train(self,
              num_epochs: int,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              optimizer: Optimizer,
              criterion,
              output_path: str):
        """
        Trains the model & validates it & saves the checkpoints
        Logs the training & val (acc, loss) into wandb
        Args:
            num_epochs (int): Number of epochs that the train will be made
            train_dataloader (DataLoader): train set torch dataloader
            val_dataloader (DataLoader): validation set torch dataloader
            optimizer: torch optimizer
            criterion: torch loss function
            output_path (str): model save path
        """
        wandb.init(
            project="CaseDs",
            config={
                "model": "Resnet18",
                "optimizer": optimizer.__class__.__name__,
                "dataset": "custom",
                "from-scratch": self.scratch,
                "epochs": num_epochs
            }
        )
        os.makedirs(output_path, exist_ok=True)
        pbar = tqdm(range(num_epochs), total=num_epochs)
        for epoch in pbar:
            train_acc, train_loss = self._train_iter(dataloader=train_dataloader,
                                                     optimizer=optimizer,
                                                     criterion=criterion)
            val_acc, val_loss = self._validate_iter(dataloader=val_dataloader,
                                                    criterion=criterion)
            pbar.set_description_str(f"Scratch({self.scratch}), Optimizer({optimizer.__class__.__name__}), Epoch{epoch + 1}")
            pbar.set_postfix_str(f"Train / Val Acc: ({'%.02f' % train_acc}/{'%.02f' % val_acc}) & "
                                 f"Train / Val Loss: ({'%.02f' % train_loss}/{'%.02f' % val_loss})")

            wandb.log({"train_acc": train_acc, "train_loss": train_loss, "val_acc": val_acc, "val_loss": val_loss})

            model_name = f"ResNet18_E{epoch + 1}_VA_{'%.01f' % val_acc}_TA_{'%.01f' % train_acc}"
            torch.save(self.model.state_dict(), op.join(output_path, f'{model_name}.pth'))
        wandb.finish()

    def _train_iter(self, dataloader: DataLoader, optimizer: Optimizer, criterion):
        """
        Iterates train model one epoch
        Args:
            dataloader (DataLoader): train set torch dataloader
            optimizer: torch optimizer
            criterion: torch loss function
        Returns:
            epoch_acc (float): Accuracy of one epoch
            epoch_loss (float): Loss of one epoch
        """
        self.model.train()
        training_loss, correct_preds, counter = 0, 0, 0
        for idx, data in enumerate(dataloader):
            counter += 1
            images, labels, image_paths = data
            images = images.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels.float())
            training_loss += loss

            _, preds = torch.max(outputs.data, 1)
            _, lbls = torch.max(labels.data, 1)
            correct_preds += (preds == lbls).sum().item()

            loss.backward()
            optimizer.step()
        epoch_loss = training_loss / counter
        epoch_acc = 100 * (correct_preds / len(dataloader.dataset))
        return epoch_acc, epoch_loss

    def _validate_iter(self, dataloader: DataLoader, criterion):
        """
        Iterates validation model one epoch
        Args:
            dataloader (DataLoader): validation set torch dataloader
            criterion: torch loss function
        Returns:
            epoch_acc (float): Accuracy of one epoch
            epoch_loss (float): Loss of one epoch
        """
        self.model.eval()
        val_loss, correct_preds, counter = 0, 0, 0
        with torch.no_grad():
            for idx, data in enumerate(dataloader):
                counter += 1
                images, labels, image_paths = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()

                _, preds = torch.max(outputs.data, 1)
                _, lbls = torch.max(labels.data, 1)
                correct_preds += (preds == lbls).sum().item()

        epoch_loss = val_loss / counter
        epoch_acc = 100 * (correct_preds / len(dataloader.dataset))

        return epoch_acc, epoch_loss

    def extract_features(self, dataloader: DataLoader, model_path: str, save_path: str = "FeatureVectors.pth"):
        """
        Extracts features of all the crops and saves them into a torch dict file
        Args:
            dataloader (DataLoader): all dataset torch dataloader
            model_path (str): trained model path that will be used in order to extract features
            save_path (str): extracted features dictionary file save path
        """
        state_dict = torch.load(model_path, map_location=torch.device(self.device))
        self.model.load_state_dict(state_dict)
        feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
        feat_vectors = {}
        with torch.no_grad():
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            for test_img_no, test_data in pbar:
                images, labels, image_paths = test_data
                images = images.to(self.device)
                outputs = feature_extractor(images)
                for image_path, feat_vector in zip(image_paths, outputs):
                    feat_vectors[image_path] = feat_vector.flatten()
        torch.save(feat_vectors, "FeatureVectors.pth")

    def find_similars(self, save_path: str, features_path: str = "FeatureVectors.pth", number_of_tests: int = 10):
        """
        Finds the similar 12 crops for number_of_tests number of test crops, saves the result grid image
        Args:
            save_path (str): save path of the result grid images
            features_path (str): torch dict file path that holds the feature vectors for all crops
            number_of_tests (int): Number of test sample that will be processed
        """
        feature_vectors = torch.load(features_path)
        cos = nn.CosineSimilarity()
        image_paths, feats = list(feature_vectors.keys()), list(feature_vectors.values())
        idxs = np.arange(len(feats))
        random.shuffle(idxs)
        idxs = idxs[:number_of_tests]
        one_dim = 200
        for test_sample_no, test_sample_idx in enumerate(idxs):
            result_image = np.zeros((2 * one_dim, 8 * one_dim, 3), dtype=np.uint8)
            selected_sample_img_path, selected_sample_feat = image_paths[test_sample_idx], feats[test_sample_idx]
            scores = cos(torch.unsqueeze(selected_sample_feat, 0), torch.stack(feats))
            most_similar_idxs = torch.argsort(scores, descending=True)[:12]
            selected_image = cv2.imread(selected_sample_img_path)
            selected_image = cv2.resize(selected_image, (2 * one_dim, 2 * one_dim))
            result_image[:2 * one_dim, :2 * one_dim] = selected_image
            for idx, similar_image_idx in enumerate(most_similar_idxs):
                similar_image = cv2.imread(image_paths[similar_image_idx])
                similar_image = cv2.resize(similar_image, (one_dim, one_dim))

                text = '%.2f' % scores[similar_image_idx]
                x1, y1 = 20, 20
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                similar_image = cv2.rectangle(similar_image, (x1, y1 - 20), (x1 + w, y1), (255, 0, 255), -1)
                similar_image = cv2.putText(similar_image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                            (255, 255, 255), 1)
                if idx // 6 == 0:
                    result_image[: one_dim, (2 + idx) * one_dim:(3 + idx) * one_dim] = similar_image
                else:
                    idx_ = idx % 6
                    result_image[one_dim:, (2 + idx_) * one_dim:(3 + idx_) * one_dim] = similar_image
            image_name = f"{'%03d' % test_sample_no}.jpg"
            cv2.imwrite(op.join(save_path, image_name), result_image)



