import torch.utils.data
from torch import nn, Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import wandb
import os.path as op

class Model:

    def __init__(self, scratch: bool = False, num_classes: int = 9, device: str = "cuda"):
        self.scratch = scratch
        self.device = device
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
              dataloader: DataLoader,
              optimizer: Optimizer,
              criterion: Module,
              output_path: str):

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

        pbar = tqdm(range(num_epochs), total=num_epochs)
        for epoch in pbar:
            train_acc, train_loss = self._train_iter(dataloader=dataloader,
                                                     optimizer=optimizer,
                                                     criterion=criterion)
            val_acc, val_loss = self._validate_iter(dataloader=dataloader,
                                                    criterion=criterion)
            pbar.set_description_str(f"Epoch {epoch + 1}")
            pbar.set_postfix_str(f"Train / Val Acc: ({'%.02f' % train_acc}/{'%.02f' % val_acc}) & "
                                 f"Train / Val Loss: ({'%.02f' % train_loss}/{'%.02f' % val_loss})")

            wandb.log({"train_acc": train_acc, "train_loss": train_loss, "val_acc": val_acc, "val_loss": val_loss})

            model_name = f"ResNet18_E{epoch + 1}_VA_{'%.01f' % val_acc}_TA_{'%.01f' % train_acc}"
            torch.save(self.model.state_dict(), op.join(output_path, f'{model_name}.pth'))
        wandb.finish()

    def _train_iter(self, dataloader: DataLoader, optimizer: Optimizer, criterion):
        self.model.train()
        training_loss, correct_preds, counter = 0, 0, 0
        for idx, data in enumerate(dataloader):
            counter += 1
            images, labels = data
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
        self.model.eval()
        val_loss, correct_preds, counter = 0, 0, 0
        with torch.no_grad():
            for idx, data in enumerate(dataloader):
                counter += 1
                images, labels = data
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

