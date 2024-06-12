import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
from PIL import Image
from CTGNeuralNetwork import CTGNeuralNetwork, preprocess

import os
from PIL import Image

import os


directory = "datasets/train/"

types = ["Charmander", "Gastly", "Snorlax", "Pikachu"]
num_classes = len(types)

np.set_printoptions(suppress=True)

n = CTGNeuralNetwork()

enough_images = 1000

file_lists = []


for i in range(num_classes):
    dir = directory + types[i] + '/'
    files = []
    for file in os.listdir(dir):
        x = preprocess(dir + file)
        if x is not None:
            files.append(file)
        if (len(files) == enough_images):
            break
    file_lists.append(files)


for i in range(num_classes):
    print(len(file_lists[i]))

print("Start: ", datetime.now())

epochs = 1

trained_on = 0

stop_at = 150

for epoch in range(epochs):
    for i in range(stop_at):
        for label in range(num_classes):
            dir = directory + types[label] + '/'
            files = file_lists[label]

            if i >= len(files):
                continue

            f = files[i]
            f = dir + f
            if i % 100 == 0:
                print(i, f)

            img = preprocess(f)
 
            # print(label)
            target = np.zeros(len(types))
            target[label] = 1.0
            
            n.train(img, target)

            

torch.save(n.state_dict(), "NN.pth")
print("End: ", datetime.now())