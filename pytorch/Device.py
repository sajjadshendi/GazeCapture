import pandas as pd
import math

class Device():
    def __init__(self, screenW, screenH, dX, dY, devices_information_path):
        self.screenW = screenW
        self.screenH = screenH
        self.dX = dX
        self.dY = dY
        self.path = devices_information_path
        self.device = ""

    def pick_device(self):
        table = pd.read_csv(self.path)
        count = len(table["DeviceName"])
        min = 0
        i_min = 0
        for i in range(count):
            row = table.iloc[i]
            indicator = math.fabs(row["DeviceScreenWidthMm"] - (self.screenW*10)) + math.fabs(row["DeviceScreenHeightMm"] - (self.screenH*10)) + math.fabs(row["DeviceCameraToScreenXMm"] - (self.dX*10))*10 + math.fabs(row["DeviceCameraToScreenYMm"] - (self.dY*10))*10
            if(i == 0):
                min = indicator
            else:
                if(min > indicator):
                    i_min = i
                    min = indicator
        key_row = table.iloc[i_min]
        self.device = key_row["DeviceName"]
        return self.device
