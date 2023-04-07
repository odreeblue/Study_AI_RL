import os
import socket
import struct
import subprocess
import time
#subprocess.call("./MiroGame.app")

class Env():
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
        next_state = 0 #다음의 공의 위치 
        reward = 0 # 다음의 공의 위치에서 받을 리워드
        done = False # 다음 공의 위치에 가게 되었을 때 게임이 종료되는지에 대한 플래그

        # 진행 방향 명령(action) -> Unity에 전달
        senddata = struct.pack('i',action) # 바이너리 데이터로 패킹
        self.socket.sendall(senddata) # Unity 로 패킹한 데이터 송신
        
        # 데이터 수신
        pos_x = struct.unpack('f',self.socket.recv(4))[0] # 다음 공의 x 좌표
        pos_y = struct.unpack('f',self.socket.recv(4))[0] # 다음 공의 z 좌표
        next_state = [pos_x,pos_y]

        reward = struct.unpack('f',self.socket.recv(4))[0] # 다음 공의 x,z에서 받을 리워드
        done_ = struct.unpack('f',self.socket.recv(4))[0] # episode가 끝났는지에 대한 플래그
                                                          # 0 -> False, 1 -> True
        if done_ == 0.0:
            done = False
        elif done_ == 1.0:
            done = True
        return next_state, reward, done 