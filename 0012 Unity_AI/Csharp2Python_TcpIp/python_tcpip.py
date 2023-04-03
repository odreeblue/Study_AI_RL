import socket 


HOST = '192.168.200.108'
PORT = 3000

client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM) 

client_socket.connect((HOST, PORT)) 


import struct
# 키보드로 입력한 문자열을 서버로 전송하고 
# 서버에서 에코되어 돌아오는 메시지를 받으면 화면에 출력합니다. 
# quit를 입력할 때 까지 반복합니다. 
while True: 

    #message = input('Enter Message : ')
    #if message == 'quit':
    #    break
    #message += "\n"
    #client_socket.send(message.encode()) 
    recvdata = client_socket.recv(144) 
    recvdata = struct.unpack('20s20si100s',recvdata)
    #data2 = struct.unpack(data)
    print('Received from the server :',recvdata)

    name = input("Name : ")
    name = bytes(name,'utf-8')
    subject = input("Subject : ")
    subject = bytes(subject,'utf-8')
    grade = input("Grade : ")
    grade = int(grade)
    memo = input("Memo : ")
    memo = bytes(memo,'utf-8')
    
    senddata = struct.pack('20s20si100s',name,subject,grade,memo)
    client_socket.sendall(senddata)

    if grade == 999:
        break


client_socket.close()