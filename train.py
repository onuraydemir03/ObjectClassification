import os.path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from dataloader import DsData
import wandb


def train(model, dataloader, optimizer, criterion, device):
    """

    :param model: Training model, it is ResNet18 by default
    :param dataloader: torch custom dataloader
    :param optimizer: torch optimizer, experiments are made on Adam & SGD
    :param criterion: experiments are made on CrossEntropyLoss
    :param device: device is cuda if it is visible else cpu
    :return: loss & accuracy for one epoch
    """
    model.train()
    training_loss, correct_preds, counter, iter_data_counter = 0, 0, 0, 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, data in pbar:
        counter += 1
        images, labels = data
        iter_data_counter += len(labels)
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels.float())

        training_loss += loss

        _, preds = torch.max(outputs.data, 1)
        _, lbls = torch.max(labels.data, 1)
        correct_preds += (preds == lbls).sum().item()

        loss.backward()

        optimizer.step()
        pbar.set_postfix_str(
            f"Loss: {'%.2f' % (training_loss / counter)}, Acc: {'%.2f' % (100 * (correct_preds / iter_data_counter))}")

    epoch_loss = training_loss / counter
    epoch_acc = 100 * (correct_preds / len(dataloader.dataset))

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
        :param model: Training model, it is ResNet18 by default
        :param dataloader: torch custom dataloader
        :param criterion: experiments are made on CrossEntropyLoss
        :param device: device is cuda if it is visible else cpu
        :return: loss & accuracy for one epoch
        """
    model.eval()
    val_loss, correct_preds, counter, iter_data_counter = 0, 0, 0, 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for idx, data in pbar:
            counter += 1
            images, labels = data
            iter_data_counter += len(labels)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.float())
            val_loss += loss.item()

            _, preds = torch.max(outputs.data, 1)
            _, lbls = torch.max(labels.data, 1)
            correct_preds += (preds == lbls).sum().item()

            pbar.set_postfix_str(
                f"Loss: {'%.2f' % (val_loss / counter)}, Acc: {'%.2f' % (100 * (correct_preds / iter_data_counter))}")

    epoch_loss = val_loss / counter
    epoch_acc = 100 * (correct_preds / len(dataloader.dataset))

    return epoch_loss, epoch_acc


if __name__ == '__main__':
    # Train iterated 20 epochs on cuda device
    num_epochs = 20
    device = "cuda"

    # Initializing dataloaders for train  & val set
    train_dataset = DsData(dataset_selector="train")
    val_dataset = DsData(dataset_selector="val")
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=12)

    for from_scratch in [True, False]:  # pretrained or scratch selector
        for opt in ["SGD", "Adam"]:  # optimizer selector
            wandb.init(
                project="CaseDs",
                config={
                    "model": "Resnet18",
                    "optimizer": opt,
                    "dataset": "custom",
                    "from-scratch": from_scratch,
                    "epochs": num_epochs
                }
            )
            model = models.resnet18(pretrained=True)  # load resnet18 model
            if from_scratch:
                os.makedirs('models_fs', exist_ok=True)
            else:
                os.makedirs('models', exist_ok=True)
                for param in model.parameters():
                    param.requires_grad = False

            num_features = model.fc.in_features  # extract fc layers features
            model.fc = nn.Linear(num_features, 9)  # num_of_class is 9
            model = model.to(device)
            criterion = nn.CrossEntropyLoss()  # (set loss function)
            if opt == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0, nesterov=False)
            else:
                optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

            for epoch in range(num_epochs):
                train_loss, train_acc = train(model=model,
                                              dataloader=train_loader,
                                              optimizer=optimizer,
                                              criterion=criterion,
                                              device=device)
                val_loss, val_acc = validate(model=model,
                                             dataloader=val_loader,
                                             criterion=criterion,
                                             device=device)
                ret_str = f"Epoch: {epoch}"
                ret_str += f" -> TrainLoss: {'%.2f' % train_loss}, TrainAcc: {'%.2f' % train_acc}"
                ret_str += f" , ValLoss: {'%.2f' % val_loss}, ValAcc: {'%.2f' % val_acc}"
                wandb.log({"train_acc": train_acc, "train_loss": train_loss, "val_acc": val_acc, "val_loss": val_loss})
                print(ret_str)
                if from_scratch:
                    save_path = os.path.join('models_fs',
                                             f'resnet_18_epoch{epoch}_loss_{"%.2f" % val_loss}_acc_{"%.2f" % val_acc}.pth')
                else:
                    save_path = os.path.join('models',
                                             f'resnet_18_epoch{epoch}_loss_{"%.2f" % val_loss}_acc_{"%.2f" % val_acc}.pth')
                torch.save(model.state_dict(), save_path)
            wandb.finish()
