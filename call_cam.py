import cv2
import numpy as np
#from sklearn.neural_network import MLPClassifier
#from sklearn.externals import joblib
#import face_recognition as face
from PIL import Image, ImageChops
import os
global frame
import itchat


def send_move_danger():
    itchat.send("Someone is in the room!", toUserName = 'filehelper')
    print('Success')

def send_move_save():
    itchat.send("He left, we are safe!", toUserName = 'filehelper')
    print('Success')

    
itchat.auto_login(hotReload = True)
    
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
#clf = joblib.load("gender_MLP_model.pkl")

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False


count_same = 0
has_same = False
count_diff = 0
has_diff = False
while rval:
    
    rval, now_frame = vc.read()

    cv2.imshow("preview", now_frame)    
    img1 = Image.fromarray(frame, 'RGB')
    img2 = Image.fromarray(now_frame, 'RGB')

    
    diff = ImageChops.difference(img1, img2)
    array = np.array(diff)
    out = np.where(array > 40)[0].size
    if(out == 0):
        print("Same")
        count_same += 1
    else:
        print("Different")
        count_diff += 1
        count_same = 0

    if(count_diff >= 10 and (not has_diff)):
        send_move_danger()
        has_diff = True
    if(count_same > 100 and count_diff >= 10):
        send_move_save()
        count_diff = 0
    
    frame = now_frame

    key = cv2.waitKey(20)
    if key == 27:
        rval = False


cv2.destroyWindow("preview")

