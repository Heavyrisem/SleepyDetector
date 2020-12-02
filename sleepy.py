import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import winsound as ws
import time as t
import threading

# import imutils

playingbeep = False
def predict(image):

    # convert to grayscale
    image = cv2.cvtColor(cropedimg, cv2.COLOR_BGR2RGB)
    predict_image = Image.fromarray(image)

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    predict_image = ImageOps.fit(predict_image, size, Image.ANTIALIAS)

    predict_image_array = np.asarray(predict_image)


    # Normalize the image
    normalized_image_array = (predict_image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict
    prediction = model.predict(data)
    # print(prediction)    


    if prediction[0][0] > prediction[0][1]:
        return str(int(prediction[0][0]*100)) + "% not sleepy"
    else:
        # a = threading.Thread(target=beep)
        # a.start()
        return str(int(prediction[0][1]*100)) + "% sleepy"

def beep():
    global playingbeep
    if playingbeep == False:
        playingbeep = True
        print('beep')
        ws.Beep(4000, 50)
        t.sleep(0.1)
        ws.Beep(4000, 50)
        t.sleep(0.1)
        ws.Beep(4000, 50)
        t.sleep(1)
        playingbeep = False

if __name__ == "__main__":


    # Load the cascade for detecte face
    face_cascade = cv2.CascadeClassifier('./facedetection/haarcascade_frontalface_default.xml')

    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (20, 40)
    font_size = 0.6
    color = (255, 255, 255)
    thickness = 2


    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # # Load the model
    model_path = "models/"
    versions = []
    for version in os.listdir(model_path):
        version = version.replace("v", "")
        versions.append(int(version))
    model = tensorflow.keras.models.load_model(model_path+'v'+str(max(versions))+'/keras_model.h5')
    print("model version is " + 'v' + str(max(versions)))

    # # Create the array of the right shape to feed into the keras model
    # # The 'length' or number of images you can put into the array is
    # # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("39835179454-offset-2516.mp4")
    
    # cap.set(3,640)
    # cap.set(4,480)

    print('camera Ready')
    while True:
        ret, frame = cap.read()
        
        if frame is None:
            print('No Frame')
            continue

        frame = cv2.resize(frame, (640, 480))
        image = frame
        screen = frame
        
        faces = face_cascade.detectMultiScale(frame, 1.1, 4)
        
        if faces is None:
            print('face not detected')

        for (x, y, w, h) in faces:
            cropedimg = frame[y: int((h)*60/100)+y, x+int(w*15/100): int(w*85/100)+x]
            # cropedimg = frame[y :h+y, x: w+x]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            prediction = predict(cropedimg)
            screen = cv2.putText(frame, prediction, (x, y), font, font_size, color, thickness, cv2.LINE_AA)
            cv2.imshow('Croped face', cropedimg)

        
        
        
        
        # frame = frame[564: 299, 352: 299]
        cv2.imshow('Face Detection', screen)

        # run the inference
        # prediction = model.predict(data)
        # print(prediction)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        



    # When everything done, release the capture
    quit()
    cap.release()
    cv2.destroyAllWindows()

