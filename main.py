import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# set if we're training or not
REBUILD_DATA = False


class DogsVSCats():
    # set the size of the image size (50x50)
    IMG_SIZE = 50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}

    training_data = []
    catcount = 0
    dogcount = 0
    errorcount = 0

    def make_training_data(self):
        # for every label/directory (cats, dogs)
        for label in self.LABELS:
            print(label)
            # for every image in each folder
            # tqdm adds a process bar
            for f in tqdm(os.listdir(label)):
                try:
                    # f = filename
                    path = os.path.join(label, f)
                    # read and convert image to greyschale
                    # why? color is not relevant
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    # resize
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    # add to training data. Convert to one hot vector using np eye
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                # sometimes it fails
                except Exception as e:
                    self.errorcount += 1

        # shuffle data
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)

        print("Cats", self.catcount)
        print("Dogs", self.dogcount)
        print("Error", self.errorcount)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # make some layers, convolutional layers
        # 1 input, 32 features, 5x5
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        # pass some random data to find the amount of inputs in linear
        # flatten the X
        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        print("Shape: ", x[0].shape)
        self.fc2 = nn.Linear(512, 2)
        print("Shape: ", x[0].shape)

    # determining the shape to flatten from convoluted to linear
    # can be replaces by .flatten method?
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        print("Shape: ", x[0].shape)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        print("Shape: ", x[0].shape)
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        print("Shape: ", x[0].shape)

        # if nothing is there yet, set linear amount
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()

# load training data. Set pickle to true
training_data = np.load("training_data.npy", allow_pickle=True)

net = Net()

plt.imshow(training_data[0][0], cmap="gray")
plt.show()



