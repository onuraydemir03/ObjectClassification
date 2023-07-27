import glob

from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam

import os.path as op
from dataset import Dataset
from model import Model

json_path = "./data/export.json"
data_dir = "./extracted_data"
dataset = Dataset(json_path=json_path, data_dir=data_dir)

trained_models_path = "./trained_models"
results_path = "./Results"


def prepare_data():
    dataset.download_images()
    dataset.save_crops()
    dataset.split(val_rate=0.2)


def train_model(num_epochs: int = 20):
    train_dataloader = dataset.create_train_dataloader()
    val_dataloader = dataset.create_val_dataloader()
    criterion = CrossEntropyLoss()
    for scratch in [True, False]:
        model = Model(scratch=scratch)
        optimizers = [
            SGD(model.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0, nesterov=False),
            Adam(model.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
        ]
        for optimizer in optimizers:
            model.train(num_epochs=num_epochs,
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        optimizer=optimizer,
                        criterion=criterion,
                        output_path=trained_models_path)

def extract_features_and_find_similars():
    all_dataloader = dataset.create_all_dataloader()
    model = Model()
    selected_model = sorted(glob.glob(op.join(trained_models_path, "*.pth")))[-1]
    model.extract_features(dataloader=all_dataloader, model_path=selected_model)
    model.find_similars(save_path=results_path, number_of_tests=5)


if __name__ == '__main__':
    # prepares the dataset
    # prepare_data()
    # train_model(num_epochs=1)
    extract_features_and_find_similars()


