import cv2
import os
global frame


cv2.namedWindow("preview")
vc = cv2.VideoCapture(1)

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

