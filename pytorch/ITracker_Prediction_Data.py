import cv2
class ITracker_Prediction_Data():
  
    def __init__(self, dataPath, imSize = (224, 224)):
      self.datapath = dataPath
      self.imSize = imSize
    
    def FrameCapture(self):
  
      # Path to video file
      vidObj = cv2.VideoCapture(self.datapath)
  
      # Used as counter variable
      count = 0
  
      # checks whether frames were extracted
      success = 1
      
      images = []
      while success:
  
          # vidObj object calls read
          # function extract frames
          success, image = vidObj.read()
  
          # Saves the frames with frame-count
          if(success):
            images.append(image)
  
          count += 1
      return images
    
