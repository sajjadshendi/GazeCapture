import torch.utils.data as data
import cv2
import numpy as np
from PIL import Image
import scipy.io as sio
from PIL import Image
from matplotlib import cm
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
import re
import copy

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



class ITrackerData_Calibr(data.Dataset):
    def __init__(self, dataPath, imSize=(224,224), gridSize=(25, 25)):
        
        self.dataPath = dataPath
        self.imSize = imSize
        self.gridSize = gridSize
        self.npzfile = np.load(dataPath)
        self.images = self.npzfile["images"]
        self.y = self.npzfile["y"]

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
        

        self.data = self.prepare()
        self.indices = np.arange(len(self.data))

    def loadImage(self, array):
        try:
            im = Image.fromarray(array).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ' + array)
            #im = Image.new("RGB", self.imSize, "white")

        return im

    def Frame_Process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
        face_coordinates = face_cascade.detectMultiScale(gray, 1.1, 10)
        face_coordinates = np.asarray(face_coordinates)
        if(face_coordinates.shape == (1,4)):
            face = frame[0:1, 0:1]
            face_gray = gray[0:1, 0:1]
            for (x, y, w, h) in face_coordinates:
                face_gray = gray[y:y + h, x:x + w]
                face = frame[y:y + h, x:x + w]
            eye_coordinates_unsorted = eye_cascade.detectMultiScale(face_gray)
            eye_coordinates_unsorted = np.asarray(eye_coordinates_unsorted)
            if(eye_coordinates_unsorted.shape == (2,4)):
                eye_coordinates = []
                if(eye_coordinates_unsorted[0][0] > eye_coordinates_unsorted[1][0]):
                    eye_coordinates.append(eye_coordinates_unsorted[1])
                    eye_coordinates.append(eye_coordinates_unsorted[0])
                else:
                    eye_coordinates.append(eye_coordinates_unsorted[0])
                    eye_coordinates.append(eye_coordinates_unsorted[1])
                count = 0
                left_eye = face[0:1, 0:1]
                right_eye = face[0:1, 0:1]
                for (ex,ey,ew,eh) in eye_coordinates:
                    if(count == 0):
                        left_eye = face[ey:ey + eh, ex:ex + ew]
                    else:
                        right_eye = face[ey:ey + eh, ex:ex + ew]
                    count += 1
                    
                
                gridLen = self.gridSize[0] * self.gridSize[1]
                grid = np.zeros([gridLen,], np.float32)
                height = frame.shape[0]
                width = frame.shape[1]
                for (x, y, w, h) in face_coordinates:
                    x = self.gridSize[0] * ((x+1) / width)
                    y = self.gridSize[0] * ((y+1) / height)
                    w = w * (self.gridSize[0] / width)
                    h = h * (self.gridSize[0] / height)
                for i in range(int(x-1), int(x+w)):
                    for j in range(int(y-1), int(y+h)):
                        grid[((j-1) * self.gridSize[0]) + (i)] = 1
                
                return face, left_eye, right_eye, grid, True
            else:
              return False, False, False, False, False
        else:
          return False, False, False, False, False

    def prepare(self):
        data = []
        for image in self.images:
            per_image = []
            face, left_eye, right_eye, grid, flag = self.Frame_Process(image)
            print(flag)
            if(not flag):
                continue
            imFace = self.loadImage(face)
            imEyeL = self.loadImage(left_eye)
            imEyeR = self.loadImage(right_eye)

            imFace = self.transformFace(imFace)
            imEyeL = self.transformEyeL(imEyeL)
            imEyeR = self.transformEyeR(imEyeR)
            per_image.append(imFace)
            per_image.append(imEyeL)
            per_image.append(imEyeR)
            per_image.append(grid)
            data.append(per_image)
            print("aa")
        return data


    def __getitem__(self, index):
        index = self.indices[index]

            
        imFace = self.data[index][0]
        imEyeL = self.data[index][1]
        imEyeR = self.data[index][2]
        gaze = np.array([self.y[index][0], self.y[index][1]], np.float32)

        faceGrid = self.data[index][3]
        

        # to tensor
        row = torch.LongTensor([int(index)])
        faceGrid = torch.FloatTensor(faceGrid)
        gaze = torch.FloatTensor(gaze)

        return row, imFace, imEyeL, imEyeR, faceGrid, gaze
    
        
    def __len__(self):
        return len(self.indices)
