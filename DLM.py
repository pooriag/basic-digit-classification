#digit learner model

import torch
from torch import nn

class digitLearnerModel(nn.Module):

    def __init__(self, resolution):

        super(digitLearnerModel, self).__init__()

        self.resolution = resolution

        self.layers = nn.Sequential(
            nn.Linear(resolution[0] * resolution[1], 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )


    def forward(self, x):

        return self.layers(x)


    def predict(self, x):

        output = self.forward(self, x)

        return torch.argmax(output, 1)


    def train(self, x, y):

        loss_function = nn.CrossEntropyLoss()

        learnig_rate = 5 * (1e-2)

        optimizer = torch.optim.SGD(self.parameters(), lr = learnig_rate)

        epochs = 400

        losses = []

        for i in range(epochs):

            y_output = self.forward(x)

            loss = loss_function(y_output, y)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            losses.append(loss.item())

            if i % 100:

                print(f"loss in {i} epoch is : {loss.item()}")


            return losses