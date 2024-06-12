import torch
import numpy as np
import os
from PIL import Image, UnidentifiedImageError
from CTGNeuralNetwork import CTGNeuralNetwork, preprocess

def is_image_file(filename):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    return any(filename.lower().endswith(ext) for ext in image_extensions)



directory = "datasets/test/"

types = ["Charmander", "Gastly", "Snorlax", "Pikachu"]
num_classes = len(types)

np.set_printoptions(suppress=True)

n = CTGNeuralNetwork()
n.load_state_dict(torch.load('NN.pth'))

stop_at = 10

file_lists = []

for i in range(num_classes):
    dir = directory + types[i] + '/'
    files = []
    for file in os.listdir(dir):
        x = preprocess(dir + file)
        if x is not None:
            files.append(file)
        if (len(files) == stop_at):
            break
    file_lists.append(files)

for i in range(num_classes):
    print(len(file_lists[i]))

correct = 0
tested_on = 0

test_labels = [0, 0, 0, 0]
test_correct= [0, 0, 0, 0]

for i in range(stop_at):
    for label in range(num_classes):
        dir = directory + types[label] + '/'
        files = file_lists[label]
        if i >= len(files):
            continue
        f = files[i]
        f = dir + f
        #print(i, stop_at, f)
        img = preprocess(f)
 

        img_tensor = torch.tensor(img).float()
        output = n.forward(img_tensor).detach().cpu().numpy()
        #print(output)

        test_labels[label] += 1

        tested_on += 1

        guess = np.argmax(output)
        if guess == label:
            test_correct[label] += 1
            correct += 1

print('Tested On:', tested_on)
print("Accuracy:", correct / tested_on)

print(test_labels)
print(test_correct)