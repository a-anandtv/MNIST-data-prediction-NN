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
REBUILD_DATA = True

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
                        img = cv2.imread (pat, cv2.IMREAD_GRAYSCALE)
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

# Build data once
if REBUILD_DATA:
    buildData = BuildData()
    buildData.make_training_data()
# %%
