import cv2
import numpy as np
from PIL import Image
import scipy.io as sio
import torchvision.transforms as transforms
import torch
import os

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


class ITracker_Prediction_Data():
  
    def __init__(self, dataPath, imSize = (224, 224), gridSize=(25, 25)):
      self.datapath = dataPath
      self.imSize = imSize
      self.gridSize = gridSize
      self.images = []
      self.faceMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_face_224.mat'))['image_mean']
      self.eyeLeftMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_left_224.mat'))['image_mean']
      self.eyeRightMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_right_224.mat'))['image_mean']
      
      self.transformFace = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.imSize, antialias=True),
            SubtractMean(meanImg=self.eyeRightMean)
      ])
      self.transformEyeL = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.imSize, antialias=True),
            SubtractMean(meanImg=self.eyeRightMean)
      ])
      self.transformEyeR = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.imSize, antialias=True),
            SubtractMean(meanImg=self.eyeRightMean)
      ])

    
    def loadImage(self, array):
        try:
            im = Image.fromarray(array).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ' + array)
            #im = Image.new("RGB", self.imSize, "white")

        return im
    
    
    def FrameCapture(self):
  
      # Path to video file
      vidObj = cv2.VideoCapture(self.datapath)
  
      # Used as counter variable
      count = 0
  
      # checks whether frames were extracted
      success = 1
     
    
      while success:
  
          # vidObj object calls read
          # function extract frames
          success, image = vidObj.read()
  
          # Saves the frames with frame-count
          if(success):
            self.images.append(image)
  
          count += 1
      return
    
    def Frame_Process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
        face_coordinates = face_cascade.detectMultiScale(gray, 1.1, 10)
        face = frame[0:1, 0:1]
        face_gray = gray[0:1, 0:1]
        for (x, y, w, h) in face_coordinates:
            face_gray = gray[y:y + h, x:x + w]
            face = frame[y:y + h, x:x + w]
        eye_coordinates_unsorted = eye_cascade.detectMultiScale(face_gray)
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
    
  
    
    def prepare(self):
        self.FrameCapture()
        data = []
        for image in self.images:
            per_image = []
            face, left_eye, right_eye, grid, flag = self.Frame_Process(image)
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
            data.apppend(per_image)
        return data
     
    def process(self, model):
        data = self.prepare()
        estimations = []
        for image in data:
          imFace = data[0]
          imEyeL = data[1]
          imEyeR = data[2]
          faceGrid = data[3]
          imFace = imFace.cuda()
          imEyeL = imEyeL.cuda()
          imEyeR = imEyeR.cuda()
          faceGrid = faceGrid.cuda()
          imFace = torch.autograd.Variable(imFace, requires_grad = True)
          imEyeL = torch.autograd.Variable(imEyeL, requires_grad = True)
          imEyeR = torch.autograd.Variable(imEyeR, requires_grad = True)
          faceGrid = torch.autograd.Variable(faceGrid, requires_grad = True)
          output = model(imFace, imEyeL, imEyeR, faceGrid)
          print(output)
          estimations.append(output)
        return estimations
    
        
