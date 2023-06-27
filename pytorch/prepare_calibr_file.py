import pandas as pd
from PIL import Image
import numpy as np
from numpy import asarray

class calibr_file():
    def __init__(self, Calibr_path):
        self.path = Calibr_path

    def collect_data_and_convert(self):
        raw_data = pd.read_csv(self.path)
        images = []
        Y = []
        for index, row in raw_data.iterrows():
            img = Image.open(row["path"])
            num_img = asarray(img)
            images.append(num_img)
            coordinates = []
            coordinates.append(row["X"])
            coordinates.append(row["Y"])
            Y.append(coordinates)
        np.savez("calibr_file", images = images, Y = Y)
