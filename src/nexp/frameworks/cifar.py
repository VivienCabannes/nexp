
import argparse
import logging
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import nexp.models.vision as vision_models
from nexp.config import (
    CHECK_DIR,
    LOG_DIR,
    cifar10_path,
)
import nexp.datasets.statistics as datastats
from nexp.utils import (
    get_unique_path,
) 
from nexp.frameworks import TrainingFramework

logging.basicConfig(
    format="{asctime} {levelname} [{name}:{lineno}] {message}",
    style='{',
    datefmt='%H:%M:%S',
    level="INFO",
    handlers=[
        # Log to stderr, which is catched by SLURM into log files
        logging.StreamHandler(),
    ],
)

class CIFAR(TrainingFramework):
    """Abstract base class for training frameworks.

    Parameters
    ----------
    args: Arguments from parser instanciated with `nexp.parser`.
    """
    def __init__(self, args: argparse.Namespace):
        self.config = args
        self.file_path = Path(__file__).resolve()
    
    def __call__(self):
        self.register_logger()
        self.register_architecture()
        self.register_dataloader()
        self.register_optimizer()

        for epoch in range(self.config.epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device),data[1].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 20 == 19:    # print every 2000 mini-batches
                    logging.info(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
                # Add checkpoints there
                # Compute training loss and accuracy only when needed.

        logging.info('Finished Training')

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                # calculate outputs by running images through the network
                outputs = self.model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        logging.info(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    def register_architecture(self):
        """Register neural network architecture."""

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        num_gpus = torch.cuda.device_count()
        logging.info(f"Device: {self.device}, {num_gpus} GPUs available")

        model_name = self.config.architecture
        model, fan_in = vision_models.headless_model(model_name)
        # model = vision_models.model(model_name)
        self.model_preprocessing = vision_models.model_preprocessing(model_name)

        self.model = nn.DataParallel(model)
        self.model.to(self.device);

    def register_dataloader(self):
        """Register dataloader for training and validation."""
        dataset_name = "cifar10"
        mean, std = datastats.compute_mean_std(dataset_name)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            self.model_preprocessing,
        ])
        trainset = torchvision.datasets.CIFAR10(
            root=cifar10_path, train=True, download=True, transform=transform
        )
        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.config.full_batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root=cifar10_path, train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.config.full_batch_size, shuffle=False, num_workers=2)

    def register_optimizer(self):
        """Register optimizer."""
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=0.9)


if __name__=="__main__":
    import nexp.parser as nparser

    parser = argparse.ArgumentParser(
        description="Training configuration",
    ) 
    nparser.decorate_parser(parser)
    parser.add_argument("--autolaunch", action="store_true")
    args = parser.parse_args()
    nparser.fill_namespace(args)

    framework = CIFAR(args)
    if args.autolaunch:
        # Launch calculations 
        framework()
    else:
        # Setup slurm launcher
        framework.launch_slurm()