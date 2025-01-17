import torch.utils.data as data
import scipy.io as sio
from PIL import Image
from matplotlib import cm
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
import re

'''
Data loader for the iTracker.
Use prepareDataset.py to convert the dataset from http://gazecapture.csail.mit.edu/ to proper format.

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

MEAN_PATH = './'

def loadMetadata(filename, silent = False):
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print('\tReading metadata from %s...' % filename)
        metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata


class SubtractMean(object):
    """Normalize an tensor image with mean.
    """

    def __init__(self, meanImg):
        self.meanImg = transforms.ToTensor()(meanImg / 255)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """       
        return tensor.sub(self.meanImg)



class ITrackerData(data.Dataset):
    def __init__(self, dataPath, split = "train", imSize=(224,224), gridSize=(25, 25)):
        
        self.split = split
        self.dataPath = dataPath
        self.imSize = imSize
        self.gridSize = gridSize
        self.npzfile = np.load(dataPath)
        self.train_eye_left = self.npzfile["train_eye_left"]
        self.train_eye_right = self.npzfile["train_eye_right"]
        self.train_face = self.npzfile["train_face"]
        self.train_face_mask = self.npzfile["train_face_mask"]
        self.train_y = self.npzfile["train_y"]
        self.val_eye_left = self.npzfile["val_eye_left"]
        self.val_eye_right = self.npzfile["val_eye_right"]
        self.val_face = self.npzfile["val_face"]
        self.val_face_mask = self.npzfile["val_face_mask"]
        self.val_y = self.npzfile["val_y"]

        self.faceMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_face_224.mat'))['image_mean']
        self.eyeLeftMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_left_224.mat'))['image_mean']
        self.eyeRightMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_right_224.mat'))['image_mean']

        self.transformFace = transforms.Compose([
             transforms.ToTensor(),
             transforms.Resize(self.imSize, antialias=True),
             SubtractMean(meanImg=self.faceMean)
        ])
        self.transformEyeL = transforms.Compose([
             transforms.ToTensor(),
             transforms.Resize(self.imSize, antialias=True),
             SubtractMean(meanImg=self.eyeLeftMean)
        ])
        self.transformEyeR = transforms.Compose([
             transforms.ToTensor(),
             transforms.Resize(self.imSize, antialias=True),
             SubtractMean(meanImg=self.eyeRightMean)
        ])
        

        if(split == "train"):
            self.indices = np.arange(len(self.train_y))
        else:
            self.indices = np.arange(len(self.val_y))

    #def loadImage(self, image_array):
        #try:
            #im = Image.open(image_array).convert('RGB')
        #except OSError:
            #raise RuntimeError('Could not read image: ')
            #im = Image.new("RGB", self.imSize, "white")

        #return im


    def makeGrid(self, params):
        gridLen = self.gridSize[0] * self.gridSize[1]
        grid = np.zeros([gridLen,], np.float32)
        
        place = 0
        for i in range(self.gridSize[0]):
            for j in range(self.gridSize[1]):
                grid[place] = params[i][j]
                place += 1
        return grid

    def __getitem__(self, index):
        index = self.indices[index]

        if(self.split == "train"):
            
            imFace = self.transformFace(Image.fromarray(self.train_face[index]))
            imEyeL = self.transformEyeL(Image.fromarray(self.train_eye_left[index]))
            imEyeR = self.transformEyeR(Image.fromarray(self.train_eye_right[index]))
            gaze = np.array([self.train_y[index][0], self.train_y[index][1]], np.float32)

            faceGrid = self.makeGrid(self.train_face_mask[index])
        
        else:


            imFace = self.transformFace(Image.fromarray(self.val_face[index]))
            imEyeL = self.transformEyeL(Image.fromarray(self.val_eye_left[index]))
            imEyeR = self.transformEyeR(Image.fromarray(self.val_eye_right[index]))
            gaze = np.array([self.train_y[index][0], self.train_y[index][1]], np.float32)

            faceGrid = self.makeGrid(self.val_face_mask[index])
        

        # to tensor
        row = torch.LongTensor([int(index)])
        faceGrid = torch.FloatTensor(faceGrid)
        gaze = torch.FloatTensor(gaze)

        return row, imFace, imEyeL, imEyeR, faceGrid, gaze
    
        
    def __len__(self):
        return len(self.indices)
