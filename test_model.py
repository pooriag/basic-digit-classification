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

model = digitLearnerModel([28, 28])

model.load_state_dict(torch.load("models/digit_model"))

test_data = datasets.MNIST(
    root="data",
    download=False,
    train=False,
    transform=Compose([ToTensor(), Grayscale()])
)


test_dataLoader = DataLoader(test_data, 1000, True)

test_batch_images, test_batch_labels = next(iter(test_dataLoader))
i = 0
for i in range(999):
    #plt.imshow(test_batch_images[i].squeeze())
    #plt.show()
    if test_data.classes[test_batch_labels[i]][0] == (str(model.predict(test_batch_images[i]).item())):
        i += 1
        

print(i)


