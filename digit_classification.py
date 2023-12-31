import warnings

import torch
from matplotlib import MatplotlibDeprecationWarning

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Grayscale

import matplotlib.pyplot as plt

from DLM import digitLearnerModel

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

train_data = datasets.MNIST(
    root="data",
    download=False,
    train=True,
    transform=Compose([ToTensor(), Grayscale()])
)

train_dataLoader = DataLoader(train_data, 500, True)

train_batch_images, train_batch_labels = next(iter(train_dataLoader))
print(train_batch_images[0].size())

plt.imshow(train_batch_images[0].squeeze())
plt.show()
print(train_data.classes[train_batch_labels[0]])



model = digitLearnerModel([28, 28])

model.train(train_dataLoader)

model.plot_losses_with_respect_to_epochs()

torch.save(model.state_dict(), "models/digit_model")