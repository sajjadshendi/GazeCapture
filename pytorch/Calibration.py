import torch
from ITrackerData import ITrackerData

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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
        epoch = 0

        for epoch in range(0, epoch):
            self.adjust_learning_rate(optimizer, epoch)
        
        for epoch in range(epoch, self.epochs):
            self.adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train(train_loader, self.model, criterion, optimizer, epoch)

        return self.model
        
    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        self.lr = self.base_lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.state_dict()['param_groups']:
            param_group['lr'] = self.lr
