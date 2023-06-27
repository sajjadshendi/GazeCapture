import pandas as pd
from PIL import Image
import numpy as np
from numpy import asarray

class Calibr_file():
    def __init__(self, Calibr_path):
        self.path = Calibr_path
        self.output = "calibr_file.npz"

    def collect_data_and_convert(self):
        raw_data = pd.read_csv(self.path)
        images = []
        Y = []
        for index, row in raw_data.iterrows():
            img = Image.open(row["Path"])
            num_img = asarray(img)
            images.append(num_img)
            coordinates = []
            coordinates.append(row["X"])
            coordinates.append(row["Y"])
            Y.append(coordinates)
        np.savez(self.output, images = images, Y = Y)
        return self.output
