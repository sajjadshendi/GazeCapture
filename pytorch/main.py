import math, shutil, os, time, argparse
import numpy as np
import pandas as pd
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.models as models

from ITrackerData import ITrackerData
from ITrackerModel import ITrackerModel
from ITracker_Prediction_Data import ITracker_Prediction_Data

'''
Train/test code for iTracker.

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='iTracker-pytorch-Trainer.')
parser.add_argument('--data_path', help="Path to processed dataset. It should contain metadata.mat. Use prepareDataset.py.")
parser.add_argument('--sink', type=str2bool, nargs='?', const=True, default=False, help="Just sink and terminate.")
parser.add_argument('--reset', type=str2bool, nargs='?', const=True, default=False, help="Start from scratch (do not load).")
parser.add_argument('--predict', type=str2bool, nargs='?', const=True, default=False, help="use model for predicting")
parser.add_argument('--orientation', help='''Orientation: The orientation of the interface, as described by the enumeration UIInterfaceOrientation, where:
                                                1: portrait
                                                2: portrait, upside down (iPad only)
                                                3: landscape, with home button on the right
                                                4: landscape, with home button on the left''')
parser.add_argument('--screenW')
parser.add_argument('--screenH')
args = parser.parse_args()

# Change there flags to control what happens.
doLoad = not args.reset # Load checkpoint at the beginning
doTest = args.sink # Only run test, no training
doPredict = args.predict # predict gaze estimation

workers = 2
epochs = 25
batch_size = torch.cuda.device_count()*100 # Change if out of cuda memory

base_lr = 0.0001
momentum = 0.9
weight_decay = 1e-4
print_freq = 10
prec1 = 0
best_prec1 = 1e20
lr = base_lr

count_test = 0
count = 0

def transform_predicts(xCam, yCam, orientation, device, screenW, screenH):
    apple_device_data = pd.read_csv("apple_device_data.csv")
    xCam = xCam * 10
    yCam = yCam * 10
    xCurr = xCam
    yCurr = yCam
    oCurr = orientation
    screenWCurr = screenW * 10
    screenHCurr = screenH * 10
    row = apple_device_data.loc[apple_device_data["DeviceName"] == device]
    dX = row["DeviceCameraToScreenXMm"]
    dY = row["DeviceCameraToScreenYMm"]
    dW = row["DeviceScreenWidthMm"]
    dH = row["DeviceScreenHeightMm"]
    if(oCurr == 1):
        xCurr = xCurr + dX
        yCurr =  -yCurr - dY
    elif(oCurr == 2):
        xCurr = xCurr - dX + dW
        yCurr = -yCurr + dY + dH
    elif(oCurr == 3):
        xCurr = xCurr - dY
        yCurr = -yCurr - dX + dW
    else:
        xCurr = xCurr + dY + dH
        yCurr = -yCurr + dX
    
    if(oCurr == 1 or oCurr == 2):
        xCurr = xCurr * (screenWCurr / dW)
        yCurr = yCurr * (screenHCurr / dH)
    
    if(oCurr == 3 or oCurr == 4):
        xCurr = xCurr * (screenWCurr / dH);
        yCurr = yCurr * (screenHCurr / dW);

    xScreen = xCurr
    yScreen = yCurr
    xScreen = xScreen / 10
    yScreen = yScreen / 10
    return xScreen, yScreen

def draw(predicts, screenW, screenH, orientation):
    x = []
    y = []
    n = []
    for point in predicts:
      x.append(point[1])
      y.append(screenH - point[0])
    for i in range(len(x)):
      n.append(str(i))
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_aspect('equal', adjustable='box')
    if(orientation == 1 or orientation == 2):
        ax.set_xlim((0, screenW))
        ax.set_ylim((0, screenH))
    else:
        ax.set_xlim((0, screenH))
        ax.set_ylim((0, screenW))
    ax.grid(True)
    
    for j, txt in enumerate(n):
      ax.annotate(txt, (x[j], y[j]))
    plt.savefig("result")

def main():
    global args, best_prec1, weight_decay, momentum

    model = ITrackerModel()
    model = torch.nn.DataParallel(model)
    model.cuda()
    imSize=(224,224)
    cudnn.benchmark = True   

    epoch = 0
    if doLoad:
        saved = load_checkpoint()
        if saved:
            print('Loading checkpoint for epoch %05d with loss %.5f (which is the mean squared error not the actual linear error)...' % (saved['epoch'], saved['best_prec1']))
            state = saved['state_dict']
            try:
                model.module.load_state_dict(state)
            except:
                model.load_state_dict(state)
            epoch = saved['epoch']
            best_prec1 = saved['best_prec1']
        else:
            print('Warning: Could not read checkpoint!')

    if(not doPredict):
      dataTrain = ITrackerData(dataPath = args.data_path, split='train', imSize = imSize)
      dataVal = ITrackerData(dataPath = args.data_path, split='test', imSize = imSize)

      train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

      val_loader = torch.utils.data.DataLoader(
        dataVal,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

      criterion = nn.MSELoss().cuda()

      optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    else:
      dataPredict = ITracker_Prediction_Data(dataPath = args.data_path, imSize = imSize)



    # Quick test
    if doTest:
        validate(val_loader, model, criterion, epoch)
        return
    
    if doPredict:
        raw_predicts = dataPredict.process(model)
        print(raw_predicts)
        predicts = []
        for raw_predict in raw_predicts:
            predict = []
            x, y = transform_predicts(raw_predict[0][0].item(), raw_predict[0][1].item(), int(args.orientation), "iPhone 6s Plus", float(args.screenW), float(args.screenH))
            predict.append(x.item())
            predict.append(y.item())
            predicts.append(predict)
        print(predicts)
        draw(predicts, float(args.screenW), float(args.screenH), int(args.orientation))
        return

    for epoch in range(0, epoch):
        adjust_learning_rate(optimizer, epoch)
        
    for epoch in range(epoch, epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)



def train(train_loader, model, criterion,optimizer, epoch):
    global count
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

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
        output = model(imFace, imEyeL, imEyeR, faceGrid)

        loss = criterion(output, gaze)
        
        losses.update(loss.data.item(), imFace.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        count=count+1

        print('Epoch (train): [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

def validate(val_loader, model, criterion, epoch):
    global count_test
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()


    oIndex = 0
    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        imFace = imFace.cuda()
        imEyeL = imEyeL.cuda()
        imEyeR = imEyeR.cuda()
        faceGrid = faceGrid.cuda()
        gaze = gaze.cuda()
        
        imFace = torch.autograd.Variable(imFace, requires_grad = False)
        imEyeL = torch.autograd.Variable(imEyeL, requires_grad = False)
        imEyeR = torch.autograd.Variable(imEyeR, requires_grad = False)
        faceGrid = torch.autograd.Variable(faceGrid, requires_grad = False)
        gaze = torch.autograd.Variable(gaze, requires_grad = False)

        # compute output
        with torch.no_grad():
            output = model(imFace, imEyeL, imEyeR, faceGrid)

        loss = criterion(output, gaze)
        
        lossLin = output - gaze
        lossLin = torch.mul(lossLin,lossLin)
        lossLin = torch.sum(lossLin,1)
        lossLin = torch.mean(torch.sqrt(lossLin))

        losses.update(loss.data.item(), imFace.size(0))
        lossesLin.update(lossLin.item(), imFace.size(0))
     
        # compute gradient and do SGD step
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        print('Epoch (val): [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error L2 {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
                    epoch, i, len(val_loader), batch_time=batch_time,
                   loss=losses,lossLin=lossesLin))

    return lossesLin.avg

CHECKPOINTS_PATH = '.'

def load_checkpoint(filename='checkpoint.pth.tar'):
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if not os.path.isdir(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH, 0o777)
    bestFilename = os.path.join(CHECKPOINTS_PATH, 'best_' + filename)
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestFilename)


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


if __name__ == "__main__":
    main()
    print('DONE')
