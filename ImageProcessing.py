import os
import numpy as np
from PIL import Image

def preprocess(f):
    image = Image.open(f)
    image = image.resize((128, 128))
    a = np.array(image) / 255.0
    a = a.reshape(128 * 128 * 3)
    return a

train = "datasets/train/"

#What they are
types = ["Charmander", "Gastly", "Hypno", "Snorlax","Pikachu"]

stop_at = 

for i in range(len(types)):
    directory = train + types[i]
    counter = 0
    for filename in os.listdir(directory):
        counter = counter + 1
        if counter == stop_at:
            break
        f = os.path.join(directory, filename)

        if not os.path.isfile(f):
            continue

        print(f)
        img = preprocess(f)
        label = i
        # If you are doing a choice from categories
        target = np.zeros(len(types))
        target[label] = 1.0 