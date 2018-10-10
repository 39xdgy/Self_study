import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import face_recognition as face
import os
global frame
from sklearn import metrics
import pickle
from sklearn.externals import joblib



cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
clf = joblib.load("gender_MLP_model.pkl")

shot_count = 0;


if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False
first_pic = True
count = 0;
while rval and shot_count < 100:
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

    img = face.load_image_file("test.jpg")
    encode = face.face_encodings(img)
    if len(encode) > 0:
        name = "my_face" + str(shot_count) + ".jpg"
        cv2.imwrite(name, frame)
        shot_count = shot_count + 1


    count = count + 1

    key = cv2.waitKey(20)
    if key == 27:
        os.remove("test.jpg")
        rval = False


cv2.destroyWindow("preview")
