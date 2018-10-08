import itchat
import datetime, os, platform, time
from call_cam import rval
def send_move():
    users = itchat.search_friends(name = 'Boss')
    print(users)
    #userName = users[0]['UserName']
    itchat.send("test", toUserName = 'filehelper')
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
    
while(rval):
    if(os.path.isfile("breaker.jpg")):
        is_break_in = True
        send_move_danger()
    if((not os.path.isfile("breaker.jpg")) and is_break_in):
        send_move_safe()
        is_break_in = False
        
    #itchat.run()
    
