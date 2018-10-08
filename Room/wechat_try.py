import itchat
import datetime, os, platform, time
import cv2
def send_move(friends_name, text):
    users = itchat.search_friends(name = friends_name)
    print(users)
    userName = users[0]['UserName']
    itchat.send(text, toUserName = userName)
    print('Success')

def send_move_danger():
    itchat.send("Someone is in the room!", toUserName = 'filehelper')
    print('Success')

def send_move_save():
    itchat.send("He left, we are safe!", toUserName = 'filehelper')
    print('Success')
    
@itchat.msg_register(itchat.content.TEXT)
def print_content(msg):
    print(msg['User']['NickName'] + 'said: ' + msg['Text'])


itchat.auto_login(hotReload = True)
is_break_in = False
key = cv2.waitKey(20)
while(not (key == 27)):
    if(os.path.isfile("./breaker.jpg") and (not is_break_in)):
        is_break_in = True
        send_move_danger()
        itchat.send_image("breaker.jpg", toUserName = 'filehelper')
    if((not os.path.isfile("./breaker.jpg")) and is_break_in):
        send_move_save()
        is_break_in = False
        
    #itchat.run()
    
