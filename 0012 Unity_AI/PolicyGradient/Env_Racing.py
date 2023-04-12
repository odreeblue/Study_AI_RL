import os
import socket
import struct
import subprocess
import time
#subprocess.call("./MiroGame.app")
import base64
from io import StringIO
from io import BytesIO
from PIL import Image
import tensorflow as tf
import numpy as np
class Env1():
    def __init__(self):
        '''게임 환경 실행'''
        #os.system("Racing.exe")
        time.sleep(5)
        self.server_ip = '127.0.0.1'
        self.server_port = 50001
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
    def connect(self):
        self.socket.connect((self.server_ip, self.server_port))


    def step(self, action):
        next_state_position = 0 #다음의 공의 위치 
        reward = 0 # 다음의 공의 위치에서 받을 리워드
        done = False # 다음 공의 위치에 가게 되었을 때 게임이 종료되는지에 대한 플래그

        # 진행 방향 명령(action) -> Unity에 전달
        senddata = struct.pack('i',action) # 바이너리 데이터로 패킹
        self.socket.sendall(senddata) # Unity 로 패킹한 데이터 송신
        
        # 데이터 수신
        pos_x = struct.unpack('f',self.socket.recv(4))[0] # 다음 공의 x 좌표
        pos_y = struct.unpack('f',self.socket.recv(4))[0] # 다음 공의 z 좌표
        next_state_position = np.array([[pos_x,pos_y]])

        reward = struct.unpack('f',self.socket.recv(4))[0] # 다음 공의 x,z에서 받을 리워드
        done_ = struct.unpack('f',self.socket.recv(4))[0] # episode가 끝났는지에 대한 플래그 0 -> False, 1 -> True
        if int(done_) == 0:
            done = False
        elif int(done_) == 1:
            done = True
        # 이미지 데이터 수신
        image_size = struct.unpack('f',self.socket.recv(4))[0] # image size 크기 받기
        print("x: "+str(pos_x)+", \
           z: "+str(pos_y)+", \
           reward: "+str(reward)+", \
           is_episode_end: "+str(done)+", \
           image_size : "+str(image_size))
        data = b''
        to_receive = int(image_size)
        while to_receive > 0:
            data += self.socket.recv(int(image_size))
            to_receive = int(image_size) - len(data)
        img = base64.b64decode(data) #base64데이터 -> 문자열데이터
        stream = BytesIO(img) 
        image = Image.open(stream).convert('L')
        image = image.resize((64,64))
        #image.show()
        #print(np.asarray(image))
        #print(np.asarray(image).shape)
        stream.close()
        next_state_image = np.asarray(image).reshape(1,64,64,1).astype(np.float32)
        #next_state_image = tf.image.convert_image_dtype(img,tf.)
        
        next_state = {'image':next_state_image,'position':next_state_position}

        #####################
        #### 버퍼 비우기 ####
        tempdata = b''
        to_receive2 = 51000-int(image_size)
        while to_receive2 > 0:
            tempdata += self.socket.recv(51000-int(image_size))
            to_receive2 = int(51000-int(image_size))-len(tempdata) 
        #####################
        
        return next_state, reward, done 
    

