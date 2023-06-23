import pandas as pd
import math

class Device():
    def __init__(screenW, screenH, dX, dY, devices_information_path):
        self.screenW = screenW
        self.screenH = screenH
        self.dX = dX
        self.dY = dY
        self.path = devices_information_path
        self.devce = ""

    def pick_device():
        table = pd.read_csv(path)
        print(table)
        count = len(table["DeviceName"])
        print(count)
        min = 0
        i_min = 0
        for i in range(count):
            row = table.iloc[i]
            indicator = math.abs(row["DeviceScreenWidthMm"] - (screenW*10)) + math.abs(row["DeviceScreenHeightMm"] - (screenH*10)) + math.abs(row["DeviceCameraToScreenXMm"] - (dX*10))*10 + math.abs(row["DeviceCameraToScreenYMm"] - (dY*10)*10)
            if(i == 0):
                min = indicator
            else:
                if(min > indicator):
                    i_min = i
                    min = indicator
                    self.device = row["DeviceName"]
        return self.device
