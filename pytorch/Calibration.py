import torch
import time
import torch.nn as nn
from ITrackerData_Calibr import ITrackerData_Calibr
from prepare_calibr_file import Calibr_file

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
    def __init__(self, calibr_path, model, imSize):
        self.model = model
        self.calibr_path = calibr_path
        self.imSize = imSize
        self.workers = 2
        self.epochs = 25
        self.batch_size = torch.cuda.device_count()*100 # Change if out of cuda memory
        self.base_lr = 0.0001
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.lr = self.base_lr
        self.count = 0

    def Calibr(self):
        obj = Calibr_file(self.calibr_path)
        calibr_file = obj.collect_data_and_convert()
        dataTrain = ITrackerData_Calibr(dataPath = calibr_file, imSize = self.imSize)
        train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size=self.batch_size, shuffle=True,
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
            self.train(train_loader, criterion, optimizer, epoch)

        return self.model

    def train(self, train_loader, criterion,optimizer, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        # switch to train mode
        self.model.train()

        end = time.time()

        for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze) in enumerate(train_loader):
        
            # measure data loading time
            data_time.update(time.time() - end)
            imFace = imFace.cuda()
            imEyeL = imEyeL.cuda()
            imEyeR = imEyeR.cuda()
            faceGrid = faceGrid.cuda()
            gaze = gaze.cuda()
        
            imFace = torch.autograd.Variable(imFace, requires_grad = True)
            imEyeL = torch.autograd.Variable(imEyeL, requires_grad = True)
            imEyeR = torch.autograd.Variable(imEyeR, requires_grad = True)
            faceGrid = torch.autograd.Variable(faceGrid, requires_grad = True)
            gaze = torch.autograd.Variable(gaze, requires_grad = False)

            # compute output
            output = self.model(imFace, imEyeL, imEyeR, faceGrid)

            loss = criterion(output, gaze)
        
            losses.update(loss.data.item(), imFace.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.count=self.count+1

            print('Epoch (train): [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
        
    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        self.lr = self.base_lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.state_dict()['param_groups']:
            param_group['lr'] = self.lr
