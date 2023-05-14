import torch.utils.data as data
import scipy.io as sio
from PIL import Image
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


def normalize(data):
    shape = data.shape
    data = np.reshape(data, (shape[0], -1))
    data = data.astype('float32') / 255. # scaling
    data = data - np.mean(data, axis=0) # normalizing
    return np.reshape(data, shape)



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

        if(split == "train"):
            self.transformFace = transforms.Compose([
                transforms.Resize(self.imSize),
                normalize(self.train_face),
                transforms.ToTensor(),
            ])
            self.transformEyeL = transforms.Compose([
                transforms.Resize(self.imSize),
                normalize(self.train_eye_left),
                transforms.ToTensor(),
            ])
            self.transformEyeR = transforms.Compose([
                transforms.Resize(self.imSize),
                normalize(self.train_eye_right),
                transforms.ToTensor(),
            ])
        else:
            self.transformFace = transforms.Compose([
                transforms.Resize(self.imSize),
                normalize(self.val_face),
                transforms.ToTensor(),
            ])
            self.transformEyeL = transforms.Compose([
                transforms.Resize(self.imSize),
                normalize(self.val_eye_left),
                transforms.ToTensor(),
            ])
            self.transformEyeR = transforms.Compose([
                transforms.Resize(self.imSize),
                normalize(self.val_eye_right),
                transforms.ToTensor(),
            ])


        if(split == "train"):
            self.indices = np.arange(len(self.train_y))
        else:
            self.indices = np.arange(len(self.val_y))

    def loadImage(self, image_array):
        try:
            im = Image.open(image_array).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ')
            #im = Image.new("RGB", self.imSize, "white")

        return im


    def makeGrid(self, params):
        gridLen = self.gridSize[0] * self.gridSize[1]
        grid = np.zeros([gridLen,], np.float32)
        
        indsY = np.array([i // self.gridSize[0] for i in range(gridLen)])
        indsX = np.array([i % self.gridSize[0] for i in range(gridLen)])
        condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2]) 
        condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3]) 
        cond = np.logical_and(condX, condY)

        grid[cond] = 1
        return grid

    def __getitem__(self, index):
        index = self.indices[index]

        if(self.split == "train"):
            imFace = self.loadImage(self.train_face)
            imEyeL = self.loadImage(self.train_eye_left)
            imEyeR = self.loadImage(self.train_eye_right)

            imFace = self.transformFace(imFace)
            imEyeL = self.transformEyeL(imEyeL)
            imEyeR = self.transformEyeR(imEyeR)
            gaze = np.array(self.train_y, np.float32)

            faceGrid = self.makeGrid(self.train_face_mask)
        
        else:
            imFace = self.loadImage(self.val_face)
            imEyeL = self.loadImage(self.val_eye_left)
            imEyeR = self.loadImage(self.val_eye_right)

            imFace = self.transformFace(imFace)
            imEyeL = self.transformEyeL(imEyeL)
            imEyeR = self.transformEyeR(imEyeR)
            gaze = np.array(self.val_y, np.float32)

            faceGrid = self.makeGrid(self.val_face_mask)
        

        # to tensor
        row = torch.LongTensor([int(index)])
        faceGrid = torch.FloatTensor(faceGrid)
        gaze = torch.FloatTensor(gaze)

        return row, imFace, imEyeL, imEyeR, faceGrid, gaze
    
        
    def __len__(self):
        return len(self.indices)
