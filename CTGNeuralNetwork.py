import torch
import torch.nn as nn

from PIL import Image
import numpy as np

TARGET_WIDTH = 64
TARGET_HEIGHT = 64

LAYER_1 = 2000
LR = 0.0001


device = torch.device('mps')

def preprocess(f):
    if f.endswith('.DS_Store'):
        return None
    
    image = Image.open(f)
    image = image.resize((TARGET_WIDTH, TARGET_HEIGHT))
    a = np.array(image) / 255.0

    #print(a.shape)
    if len(a.shape) != 3:
        #print("Skipping image", f, "due to incorrect number of channels")
        return None
    
    if a.shape[2] == 3:
        flattened_size = a.shape[0] * a.shape[1] * a.shape[2]  # Calculate the size
        a = a.reshape(flattened_size)  # Reshape accordingly
        return a

    if a.shape[2] == 4:
        image = image.convert("RGB")
        a = np.array(image) / 255.0
        flattened_size = a.shape[0] * a.shape[1] * a.shape[2]  # Calculate the size
        a = a.reshape(flattened_size)  # Reshape accordingly
        return a

    return None
    


class CTGNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(TARGET_HEIGHT*TARGET_WIDTH*3, LAYER_1),
            nn.Sigmoid(),
            nn.Linear(LAYER_1, 4),
            nn.Sigmoid()
        )

        self.loss_function = nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.parameters(), lr=LR)

        self.to(device)

    def forward(self, inputs):
        inputs = torch.FloatTensor(inputs).to(device)
        return self.model(inputs)
    
    def train(self, inputs, targets):
        targets = torch.FloatTensor(targets).to(device)
        outputs = self.forward(inputs)

        loss = self.loss_function(outputs, targets)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()