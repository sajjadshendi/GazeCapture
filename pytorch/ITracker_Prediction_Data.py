import cv2
class ITracker_Prediction_Data():
  
    def __init__(self, dataPath, imSize = (224, 224)):
      self.datapath = dataPath
      self.imSize = imSize
      self.images = []
    
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
        eye_coordinates = []
        if(eye_coordinates_unsorted[0][0] > eye_coordinates_unsorted[1][1]):
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
        print(face, left_eye, right_eye)
    
    def prepare(self):
        self.FrameCapture()
        self.Frame_Process(self.images[0])
