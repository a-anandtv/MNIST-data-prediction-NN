# %%
# Fetching dataset
# wget https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
# sudo apt-get install unzip
# unzip kagglecatsanddogs_3367a.zip
########################################

import os
import cv2
import numpy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# %%
# Building dataset with a data builder class

# Flag to control rebuilds
REBUILD_DATA = False

# Relative path to datasets
PARENTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
DATASETS = os.path.join(PARENTDIR, "Datasets\kagglecatsanddogs_3367a")

class BuildData() :

    IMG_SIZE = 50
    CATS = "PetImages\Cat"
    DOGS = "PetImages\Dog"
    TESTING = "PetImages\Testing"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []

    # Countersss to count available data
    catCount = 0
    dogCount = 0

    def make_training_data(self):
        for label in self.LABELS:
            print ("Processing ", label)
            for f in tqdm(os.listdir(os.path.join(DATASETS, label))):
                if "jpg" in f:
                    try:
                        path = os.path.join(DATASETS, label, f)
                        img = cv2.imread (path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize (img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append ([numpy.array(img), numpy.eye(2)[self.LABELS[label]]])
                            
                        if label == self.CATS:
                            self.catCount += 1
                        elif label == self.DOGS:
                            self.dogCount += 1
                    
                    except Exception as e:
                        pass

        numpy.random.shuffle (self.training_data)
        numpy.save ("training_data.npy", self.training_data)    
        print ("CatCount: ", self.catCount, " DogCount: ", self.dogCount)

# Class for the NN module
class Net(nn.Module):
    # init 
    def __init__(self):
        # Calling parent __init__()
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d (1, 32, 5)   # 1 input 32 outputs 5x5 kernel
        self.conv2 = nn.Conv2d (32, 64, 5)  # 32 inputs 64 outputs 5x5 kernel
        self.conv3 = nn.Conv2d (64, 128, 5) # 64 inputs 128 outputs 5x5 kernel

        x = torch.randn (50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs (x)

        # Linear layers
        self.fc1 = nn.Linear (self._to_linear, 512)
        self.fc2 = nn.Linear (512, 2)
    
    # perform convolutions
    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d (F.relu (self.conv1(x)), (2, 2))
        x = F.max_pool2d (F.relu (self.conv2(x)), (2, 2))
        x = F.max_pool2d (F.relu (self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x
    
    # forward pass
    def forward(self, x):
        x = self.convs (x)
        x = x.view (-1, self._to_linear)
        x = F.relu (self.fc1(x))
        x = self.fc2 (x)    # last layer so no activation here
        return F.softmax (x, dim=1)

# %%
# Create network
net = Net()
print (net)

# Build data once
if REBUILD_DATA:
    buildData = BuildData()
    buildData.make_training_data()

# %%
# Load training data
training_data = numpy.load ("training_data.npy", allow_pickle=True)
print (len(training_data))

optimizer = optim.Adam (net.parameters(), lr=0.001)
loss_function = nn.MSELoss ()

# Load data into X
X = torch.Tensor ([i[0] for i in training_data]).view(-1, 50, 50)
# Normalize
X = X / 255
# Load labels into y
y = torch.Tensor ([i[1] for i in training_data])

VALIDATION_PCT = 0.1
val_size = int (len(X) * VALIDATION_PCT)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size]
test_y = y[-val_size]

BATCH_SIZE = 100
EPOCHS = 5

# Training
# def train (net):
for epoch in range (EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
        batch_y = train_y[i:i+BATCH_SIZE]

        # reset gradients
        net.zero_grad()

        outputs = net(batch_X)
        loss = loss_function (outputs, batch_y)
        # back propagation
        loss.backward()
        # update
        optimizer.step()
    
    print (f"Epoch: {epoch}. Loss: {loss}")

# Testing
# def test (net):
correct = 0
total = 0

with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax (test_y[i])
        net_out = net(test_X[i].view(-1, 1, 50, 50))[0]
        predicted_class = torch.argmax (net_out)

        if predicted_class == real_class:
            correct += 1
        
        total += 1
    
print ("Accuracy: ", round(correct/total, 3))
# %%
