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

    def step(self, direction,done):
        # 진행 방향 명령(direction), 게임이 끝났는지 여부(done) 
        senddata = struct.pack('ii',direction,done) # 바이너리 데이터로 패킹
        self.socket.sendall(senddata) # Unity 로 패킹한 데이터 송신
        
        # 데이터 수신
        pos_x = struct.unpack('f',socket.recv(4))[0] # 공의 x 좌표
        pos_z = struct.unpack('f',socket.recv(4))[0] # 공의 z 좌표
        


        
        
        recvdata = self.socket.recv(12) # 
        #time.sleep(0.2)



        recvdata = struct.unpack('fff', recvdata)
        



        return recvdata 