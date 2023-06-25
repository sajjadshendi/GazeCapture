import torch
from ITrackerData import ITrackerData

class Calibration():
    def __init__(calibr_path, model, imSize):
        self.model = model
        self.calibr_pah = calibr_path
        self.imSize = imSize
        self.workers = 2
        self.epochs = 25
        self.batch_size = torch.cuda.device_count()*100 # Change if out of cuda memory
        self.base_lr = 0.0001
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.lr = base_lr

    def Calibr(self):
        dataTrain = ITrackerData(dataPath = self.calibr_path, split='train', imSize = self.imSize)
        
        train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size=self.batch_size, shuffle=True,
        num_workers=self.workers, pin_memory=True)

      val_loader = torch.utils.data.DataLoader(
        dataVal,
        batch_size=self.batch_size, shuffle=False,
        num_workers=self.workers, pin_memory=True)

      criterion = nn.MSELoss().cuda()

      optimizer = torch.optim.SGD(self.model.parameters(), self.lr,
                                momentum=self.momentum,
                                weight_decay=self.weight_decay)
        
