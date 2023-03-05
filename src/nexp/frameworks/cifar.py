
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import nexp.models.vision as vision_models
from nexp.config import cifar10_path
import nexp.datasets.datastats as datastats
from nexp.trainer import Trainer

logger = logging.getLogger("cifar")
logger.setLevel(logging.INFO)


class CIFAR(Trainer):
    """Training frameworks for CIFAR.

    Parameters
    ----------
    args: Arguments from parser instanciated with `nexp.parser`.
    """
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.file_path = Path(__file__).resolve()
    
    def train(self):
        logger.info("training model")

        self.model.train()
        for epoch in range(self.config.epochs):  # loop over the dataset multiple times

            # running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                logger.debug("loading batch to devices")
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                logger.debug("set gradient accumulator to zero")
                self.optimizer.zero_grad()

                logger.debug("forward pass")
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                logger.debug("backward pass")
                loss.backward()
                self.optimizer.step()

            if epoch % self.config.checkpoint_frequency == self.config.checkpoint_frequency - 1:
                logger.info(f"saving model at epoch {epoch + 1}")
                self.save_checkpoint(full=True, epoch=epoch)

                # # print statistics
                # running_loss += loss.item()
                # if i % 20 == 19:    # print every 2000 mini-batches
                #     logger.info(f'epochs {epoch + 1:3d} ({i + 1:5d}) loss: {running_loss / 2000:.3f}')
                #     running_loss = 0.0

        logger.info('finished Training')
        logger.info(f'saving model ({self.bestcheck_path})')
        self.save_checkpoint(full=False, file_path=self.bestcheck_path)

    def test(self):
        logger.info('testing model')

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.testloader:

                logger.debug("loading batch to devices")
                images, labels = data[0].to(self.device), data[1].to(self.device)

                logger.debug("forward pass")
                outputs = self.model(images)

                logger.debug("loss computation")
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        logging.info(f'accuracy of the network on the 10000 test images: {100 * correct // total} %')

    def register_architecture(self):
        """Register neural network architecture.
        
        Load checkpoint if available.

        TODO
        ----
        - [ ] Add support for single GPU
        - [ ] Add support for multiple nodes
        """
        logger.debug("registering architecture")

        model_name = self.config.architecture
        model, fan_in = vision_models.headless_model(model_name)
        # model = vision_models.model(model_name)
        self.model_preprocessing = vision_models.model_preprocessing(model_name)

        self.model = nn.DataParallel(model)
        self.model.to(self.device);

        self.load_checkpoint()

    def register_dataloader(self, train: bool = True):
        """Register dataloader for training and validation.
        
        Parameters
        ----------
        train: flag to return training data loader.
        """
        logger.debug("registering dataloaders")

        dataset_name = "cifar10"
        mean, std = datastats.compute_mean_std(dataset_name, recompute=False)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            self.model_preprocessing,
        ])
        dataset = torchvision.datasets.CIFAR10(
            root=cifar10_path, train=train, download=True, transform=transform
        )
        shuffle = train
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.full_batch_size, shuffle=shuffle, num_workers=2)

        return dataloader

    def register_optimizer(self):
        """Register optimizer."""
        logger.debug("registering optimizer")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=0.9)

    def register_scheduler(self):
        """Register scheduler."""
        logger.debug("registering scheduler")
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)


if __name__=="__main__":
    import nexp.parser as nparser

    logging.basicConfig(
        format="{asctime} {levelname} [{name:10s}:{lineno:3d}] {message}",
        style='{',
        datefmt='%H:%M:%S',
        level="INFO",
        handlers=[
            # Log to stderr, which is catched by SLURM into log files
            logging.StreamHandler(),
        ],
    )

    logger.info("Parsing arguments")
    parser = argparse.ArgumentParser(
        description="Training configuration",
    ) 
    nparser.decorate_parser(parser)
    args = parser.parse_args()
    nparser.fill_namespace(args)

    framework = CIFAR(args)
    framework()
