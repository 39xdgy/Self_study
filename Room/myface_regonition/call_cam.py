import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import face_recognition as face
import os
global frame


cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
clf = joblib.load("gender_MLP_model.pkl")

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False
first_pic = True
count = 0
while rval:
    cv2.imshow("Preview", frame)
    rval, frame = vc.read()
    
    ##print(frame)
    
    ##encode = face.face_encodings(frame)
    
    ##if()

    if first_pic:
        cv2.imwrite("test.jpg", frame)
        first_pic = False
    elif count == 20:
        count = 0
        os.remove("test.jpg")
        cv2.imwrite("test.jpg", frame)




    count = count + 1
    
    key = cv2.waitKey(20)
    if key == 27:
        os.remove("test.jpg")
        rval = False


cv2.destroyWindow("preview")

