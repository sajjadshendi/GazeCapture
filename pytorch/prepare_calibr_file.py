import pandas as pd
from PIL import Image
import numpy as np
import cv2

class Calibr_file():
    def __init__(self, Calibr_path):
        self.path = Calibr_path
        self.output = "calibr_file.npz"

    def collect_data_and_convert(self):
        raw_data = pd.read_csv(self.path)
        images = []
        Y = []
        for index, row in raw_data.iterrows():
            img = cv2.imread(row["Path"])
            images.append(img)
            coordinates = []
            coordinates.append(row["X"])
            coordinates.append(row["Y"])
            Y.append(coordinates)
        np.savez(self.output, images = images, y = Y)
        return self.output
