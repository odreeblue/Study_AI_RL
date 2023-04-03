'''
#key input 예제
import struct
a = input()
print("입력받은 문자 : "+a)
    # up : ^[[A, down : ^[[B , right : ^[[C, left : ^[[D
print(type(a)) # type : str
if a == "w":
    aa = 1
    print("up key")
    b = a.encode()
    print(len(b))
    print(type(b))
    c = struct.pack('iii',aa,aa,aa)
    print(len(c))
'''


# 클라이언트
import socket
import struct
from pynput import keyboard
import time
# pynput 을 사용하려면 맥OS의 "보안 및 개인 정보 보호"->"손쉬운사용" -> "터미널.app"체크("시스템/응용프로그램/유틸리티/터미널.app")
                                                                    #-> "visual studio code" 체크
#server_ip = '192.168.200.179' # 위에서 설정한 서버 ip
#server_ip = '192.168.200.108'
#server_ip = '172.20.10.10'
#server_ip = '118.235.3.203'
server_ip = '127.0.0.1'
server_port = 50001 # 위에서 설정한 서버 포트번호

socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.connect((server_ip, server_port))
'''
#typeOfService = 1
#displayId = 0
#payloadLength = 4
def on_press(key):
    if key.char == "w":
        direction = 0
        senddata = struct.pack('i',direction)
        socket.sendall(senddata)
        data = socket.recv(12)
        data = struct.unpack('ffi',data)
        print(data)
    if key.char == "s":
        direction = 1
        senddata = struct.pack('i',direction)
        socket.sendall(senddata)
        time.sleep(1)
        data = socket.recv(12)
        data = struct.unpack('ffi',data)
        print(data)
    if key.char == "d":
        direction = 2
        senddata = struct.pack('i',direction)
        socket.sendall(senddata)
        data = socket.recv(12)
        data = struct.unpack('ffi',data)
        print(data)
    if key.char == "a":
        direction = 3
        senddata = struct.pack('i',direction)
        socket.sendall(senddata)
        
        data = socket.recv(12)
        data = struct.unpack('ffi',data)
        print(data)
    if key == keyboard.Key.esc:
        print("esc 눌림")
        return False
    
def on_release(key):
    if key.char == "w":
        
        print("위쪽 눌림")
        #time.sleep(0.3)
        #data = socket.recv(9)
        #msg = data.decode()
        #print("msg : ", msg)

    if key.char == "s":
        print("아래쪽 눌림")
        #time.sleep(0.3)
        #data = socket.recv(9)
        #msg = data.decode()
        #print("msg : ", msg)
        
    if key.char == "d":
        print("오른쪽 눌림")
        #time.sleep(0.3)
        #data = socket.recv(9)
        #msg = data.decode()
        #print("msg : ", msg)
        
        
    if key.char == "a":
        print("왼쪽 눌림")
        #time.sleep(0.3)
        #data = socket.recv(9)
        #msg = data.decode()
        #print("msg : ", msg)

    #data = socket.recv(4)
    #print(data.decode())
    #data = socket.recv(4)
    #print(data.decode())
    
    
    #print('Key released: {0}'.format(key))
    #if key == keyboard.Key.esc:
        # Stop listener
    #    return False

with keyboard.Listener(on_press = on_press, on_release=on_release, IS_TRUSTED = True) as listener:
    listener.join()
socket.close()
'''

# /end 입력될 때 까지 계속해서 서버에 패킷을 보냄
while True:
    
    direction_input = input()
    if direction_input == "w":
        direction = 0
        senddata = struct.pack('i',direction)
        socket.sendall(senddata)
    elif direction_input == "s":
        direction = 1
        senddata = struct.pack('i',direction)
        socket.sendall(senddata)
    elif direction_input == "d":
        direction = 2
        senddata = struct.pack('i',direction)
        socket.sendall(senddata)
    elif direction_input == "a":
        direction = 3
        senddata = struct.pack('i',direction)
        socket.sendall(senddata)
    elif direction_input == "x":
        break
    recvdata = socket.recv(12)
    recvdata = struct.unpack('ffi',recvdata)
    print(recvdata)
    '''
    if keyboard.read_key() == "w":
        direction = 0
        senddata = struct.pack('i',direction)
        socket.sendall(senddata)
    if keyboard.read_key() == "s":
        direction = 1
        senddata = struct.pack('i',direction)
        socket.sendall(senddata)
    if keyboard.read_key() == "d":
        direction = 2
        senddata = struct.pack('i',direction)
        socket.sendall(senddata)
    if keyboard.read_key() == "a":
        direction = 3
        senddata = struct.pack('i',direction)
        socket.sendall(senddata)
    if keyboard.read_key() == "x":
        break
    
    #socket.sendall(msg.encode(encoding='utf-8'))
    
    #data = socket.recv(100)
    #msg = data.decode() 
    #print('echo msg:', msg)
    
    #if msg == '/end':
    #    break
    '''
socket.close()