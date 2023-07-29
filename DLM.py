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

        x_flatten = torch.flatten(x, start_dim=0)

        output = self.forward(x_flatten)
        output = output / 10
        print(output)

        return torch.argmax(output)


    def train(self, train_data_loader):

        loss_function = nn.CrossEntropyLoss()

        learnig_rate = 5 * (1e-2)

        optimizer = torch.optim.SGD(self.parameters(), lr = learnig_rate)

        epochs = 3000

        losses = []

        for i in range(epochs):

            x, y = next(iter(train_data_loader))

            x_flatts = torch.flatten(x, start_dim=1)

            y_output = self.forward(x_flatts)

            loss = loss_function(y_output, y)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            losses.append(loss.item())

            if i % 100 == 0:

                print(f"loss in {i} epoch is : {loss.item()}")


        return losses