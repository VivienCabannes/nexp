
import logging

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import nexp.models.vision as vision_models
from nexp.config import (
    cifar10_path,
)
import nexp.datasets.statistics as datastats

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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
logging.info(f"Device: {device}, {num_gpus} GPUs available")

model_name = "resnet18"
full_model = vision_models.model(model_name)
model, fan_in = vision_models.headless_model(model_name)
model_preprocessing = vision_models.model_preprocessing(model_name)

model = nn.DataParallel(model)
model.to(device);

dataset_name = "cifar10"
mean, std = datastats.compute_mean_std(dataset_name)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

batch_size = 128 * num_gpus 

trainset = torchvision.datasets.CIFAR10(
    root=cifar10_path, train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root=cifar10_path, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.to(device)

for epoch in range(1000):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device),data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 2000 mini-batches
            logging.info(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

logging.info('Finished Training')

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

logging.info(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')