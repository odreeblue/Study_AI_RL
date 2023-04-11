# 클라이언트
import socket
import struct
from pynput import keyboard
import time
import io
from PIL import Image
import binascii
#import cv2
import numpy as np
#import simplejpeg
from io import StringIO
from io import BytesIO
import base64
# pynput 을 사용하려면 맥OS의 "보안 및 개인 정보 보호"->"손쉬운사용" -> "터미널.app"체크("시스템/응용프로그램/유틸리티/터미널.app")
                                                                    #-> "visual studio code" 체크
#server_ip = '192.168.200.179' # 위에서 설정한 서버 ip
#server_ip = '192.168.200.108'
server_ip = '127.0.0.1'
#server_ip = '118.235.3.203'
server_port = 50001 # 위에서 설정한 서버 포트번호

socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.connect((server_ip, server_port))
socket.setblocking(False)       

socket.setblocking(True)

# /end 입력될 때 까지 계속해서 서버에 패킷을 보냄
#def image_recv(sock, default_size,size):
#    data = b''
#    to_receive = default_size
#    while to_receive > 0:
#        data += sock.recv(default_size)
#        to_receive = default_size - len(data)
#    return data[0:size]


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
    pos_x = struct.unpack('f',socket.recv(4))[0]
    pos_z = struct.unpack('f',socket.recv(4))[0]
    reward = struct.unpack('f',socket.recv(4))[0]
    is_episode_end = struct.unpack('f',socket.recv(4))[0]
    image_size = struct.unpack('f',socket.recv(4))[0]
    print("x: "+str(pos_x)+", \
           z: "+str(pos_z)+", \
           reward: "+str(reward)+", \
           is_episode_end: "+str(is_episode_end)+", \
           image_size : "+str(image_size))

    #image_data = image_recv(socket,8000,int(image_size))
    
    
    data = b''
    #data= ""
    to_receive = int(image_size)
    while to_receive > 0:
        data += socket.recv(int(image_size))
        to_receive = int(image_size) - len(data)
    #return data
    image_data = data
    print("image_data : "+str(len(image_data))) # 39500

    x = base64.b64decode(image_data)

    filename = 'some_image1.png'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(x)
    #print(x)
    stream = BytesIO(x)
    image = Image.open(stream).convert('L')
    image = image.resize((64,64))
    print(np.asarray(image))
    print(np.asarray(image).shape)
    #print(type(image))
    stream.close()
    image.show()

    tempdata = b''
    #tempdata = ""
    to_receive2 = 51000-int(image_size)
    while to_receive2 > 0:
        tempdata += socket.recv(51000-int(image_size))
        to_receive2 = int(51000-int(image_size)) - len(tempdata)

    image_data2 = tempdata
    print("image_data2 : "+str(len(image_data2)))
    x2 = base64.b64decode(image_data2)
    # print(len(x2))
    
    
    filename = 'some_image2.png'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(x2)
socket.close()
