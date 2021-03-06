import itchat
import datetime, os, platform, time
import cv2
import face_recognition as face
from sklearn.externals import joblib

def send_move(friends_name, text):
    users = itchat.search_friends(name = friends_name)
    print(users)
    userName = users[0]['UserName']
    itchat.send(text, toUserName = userName)
    print('Success')

def send_move_danger():
    users = itchat.search_friends(name = 'Boss')
    userName = users[0]['UserName']
    itchat.send("Someone is in the room!", toUserName = userName)
    itchat.send_image("breaker.jpg", toUserName = userName)
    print('Success')

def send_move_save():
    users = itchat.search_friends(name = 'Boss')
    userName = users[0]['UserName']
    itchat.send("He left, we are safe!", toUserName = userName)
    print('Success')
    
@itchat.msg_register(itchat.content.TEXT)
def print_content(msg):
    print(msg['User']['NickName'] + 'said: ' + msg['Text'])

def is_my_face(clf, image_name):
    img = face.load_image_file(image_name)
    encode = face.face_encodings(img)
    if(len(encode) != 0):
        out = clf.predict(encode)
        if(out[0] == 1):
            return True
    return False

clf = joblib.load("./face_regonition/my_face.pkl")

itchat.auto_login(hotReload = True)


is_break_in = False
key = cv2.waitKey(20)
me = False
try:
    while(True):
        if(not me):
            if(os.path.isfile("./breaker.jpg") and (not is_break_in)):
                if(is_my_face(clf, "./breaker.jpg")):
                    print("Welcome back my lord")
                    me = True
                    
                send_move_danger()
                is_break_in = True
        
            elif((not os.path.isfile("./breaker.jpg")) and is_break_in):
                send_move_save()
                is_break_in = False
except KeyboardInterrupt:
    print("Finish")
    
itchat.run()

print("Finish")
