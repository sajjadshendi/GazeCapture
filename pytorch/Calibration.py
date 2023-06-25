from ITrackerData import ITrackerData
class Calibration():
    def __init__(calibr_path, model, imSize):
        self.model = model
        self.calibr_pah = calibr_path
        self.imSize = imSize

    def Calibr(self):
        dataTrain = ITrackerData(dataPath = self.calibr_path, split='train', imSize = self.imSize)
