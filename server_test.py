import socket
import keyboard
import json

HOST = '127.0.0.1'
PORT = 7000

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT))
s.listen(5)

print('server start at: %s:%s' % (HOST, PORT))
print('wait for connection...')


conn, addr = s.accept()

while True:
    
    print('addr: ' + str(addr))
    print('conn: ' + str(conn))

    indata = conn.recv(1024)
    rs = indata.decode()

    if rs == q:
        break
    else:
        conn.send(rs.encode())
    


