#서버
import socket, time
#외부(client)에서 내 컴퓨터(내부,server,공유기 환경)로 접속하려할때
# : 1. 우선 공유기를 내부ip, 포트번호 설정하여 포트포워딩 해준다
#   2. 내부 프로그램의 host는 공유기 외부 ip 말고, 컴퓨터의 192~ 와 같은 내부 ip 설정해준다
#       잘 모르겠으면, ipconfig(window), 네트워크 환경(mac) 가서 확인한다.
#   3. 외부에서 server ip는 공유기 ip 설정하고 포트번호는 포트포워딩 한 번호를 넣어준다
#   4. 연결하면된다
#host = 'localhost' 
host = '192.168.200.179'
port = 3333 

# 서버소켓 오픈
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((host, port))

# 클라이언트 접속 준비 완료
server_socket.listen()

print('echo server start')

#  클라이언트 접속 기다리며 대기 
client_soc, addr = server_socket.accept()

print('connected client addr:', addr)

# 클라이언트가 보낸 패킷 계속 받아 에코메세지 돌려줌
while True:
    data = client_soc.recv(100)
    msg = data.decode() 
    print('recv msg:', msg)
    client_soc.sendall(msg.encode(encoding='utf-8')) 
    if msg == '/end':
        break

time.sleep(5)
print('서버 종료')
server_socket.close() # 사용했던 서버 소켓을 닫아줌