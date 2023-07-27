import glob

from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam

import os.path as op
from dataset import Dataset
from model import Model

# Preset filepaths of process steps
json_path = "./data/export.json"
data_dir = "./extracted_data"
dataset = Dataset(json_path=json_path, data_dir=data_dir)

trained_models_path = "./trained_models"
results_path = "./Results"


def prepare_data(val_rate: float = .2):
    """
    Prepares the data for training
    If some are the steps are made early, it skips them
        For ex: If an image downloaded it will not be downloaded twice
    This block runs effectively
    """
    dataset.download_images()
    dataset.save_crops()
    dataset.split(val_rate=val_rate)


def train_model(num_epochs: int = 20):
    """
    This block trains & validates the model and saves the checkpoints under the selected folder
    Args:
        num_epochs (int): Epochs that train operation will be made
    """
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


def extract_features_and_find_similars(number_of_test_samples: int = 5):
    """
    This block extracts the features of all crops with trained model
    Saves them into a .pth torch dict file
    Selects randomly number_of_test_samples and draws the most similar 12 crops for each of them
    Args:
        number_of_test_samples (int): Number of the samples that the similarity search will be made on
    """
    all_dataloader = dataset.create_all_dataloader()
    model = Model()
    selected_model = sorted(glob.glob(op.join(trained_models_path, "*.pth")))[-1]
    model.extract_features(dataloader=all_dataloader, model_path=selected_model)
    model.find_similars(save_path=results_path, number_of_tests=number_of_test_samples)


if __name__ == '__main__':
    prepare_data()
    train_model(num_epochs=20)
    extract_features_and_find_similars()


