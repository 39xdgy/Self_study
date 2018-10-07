import itchat
import datetime, os, platform, time

def send_move():
    users = itchat.search_friends(name = 'Boss')
    print(users)
    #userName = users[0]['UserName']
    itchat.send("test", toUserName = 'filehelper')
    print('Success')


@itchat.msg_register(itchat.content.TEXT)
def print_content(msg):
    print(msg['User']['NickName'] + '说: ' + msg['Text'])

if __name__ == '__main__':
    itchat.auto_login(hotReload = True)
    #send_move()
    itchat.run()
    
