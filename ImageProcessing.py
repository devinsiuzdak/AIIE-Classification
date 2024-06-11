import os
import numpy as np
from PIL import Image

def preprocess(f):
    image = Image.open(f)
    image = image.resize((128, 128))
    a = np.array(image) / 255.0
    
    if len(a.shape) != 3 or a.shape[2] != 3:
        print("Skipping image", f, "due to incorrect number of channels")
        return None  # Skip preprocessing for images with incorrect number of channels
    
    flattened_size = a.shape[0] * a.shape[1] * a.shape[2]  # Calculate the size
    a = a.reshape(flattened_size)  # Reshape accordingly
    return a


train = "Datasets/Train/"

#What they are
types = ["Charmander", "Gastly", "Hypno", "Snorlax","Pikachu"]

stop_at = 100

for i in range(len(types)):
    directory = train + types[i]
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        if not os.path.isfile(f):
            continue

        print(f)
        img = preprocess(f)
        label = i
        target = np.zeros(len(types))
        target[label] = 1.0 