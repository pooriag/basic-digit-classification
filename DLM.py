#digit learner model

import torch
from torch import nn

import matplotlib.pyplot as plt

class digitLearnerModel(nn.Module):

    def __init__(self, resolution):

        super(digitLearnerModel, self).__init__()

        self.resolution = resolution
        self.losses = []

        self.layers = nn.Sequential(
            nn.Linear(resolution[0] * resolution[1], 60),
            nn.ReLU(),
            nn.Linear(60, 15),
            nn.ReLU(),
            nn.Linear(15, 15),
            nn.ReLU(),
            nn.Linear(15, 10)

        )


    def forward(self, x):

        return self.layers(x)


    def predict(self, x):

        x_flatten = torch.flatten(x, start_dim=0)

        output = self.forward(x_flatten)
        output = output / 10
        print(output)

        return torch.argmax(output)


    def train(self, train_data_loader):

        loss_function = nn.CrossEntropyLoss()

        learnig_rate = (1e-2)

        optimizer = torch.optim.SGD(self.parameters(), lr = learnig_rate)

        epochs = 30000

        for i in range(epochs):

            x, y = next(iter(train_data_loader))

            x_flatts = torch.flatten(x, start_dim=1)

            y_output = self.forward(x_flatts)

            loss = loss_function(y_output, y)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            self.losses.append(loss.item())

            if i % 100 == 0:

                print(f"loss in {i} epoch is : {loss.item()}")


        return self.losses

    def plot_losses_with_respect_to_epochs(self):

        plt.figure()

        for i in range(0, len(self.losses)):

            plt.plot(i, self.losses[i], "ro")

        plt.show()