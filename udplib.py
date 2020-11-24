import socket
# msgFromClient       = "Hello UDP Server"
# bytesToSend         = str.encode(msgFromClient)
# serverAddressPort   = ("192.168.43.33", 20001)
# bufferSize          = 1024
# UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
# UDPClientSocket.sendto(bytesToSend, serverAddressPort)
# msgFromServer = UDPClientSocket.recvfrom(bufferSize)
# msg = "Message from Server {}".format(msgFromServer[0])
# print(msg)
def Attend_send(msgFromClient,host_ip='192.168.1.9',port_no=1111,bufferSize = 1024):
    bytesToSend         = str.encode(msgFromClient)
    serverAddressPort   = (host_ip, port_no)
    UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    UDPClientSocket.settimeout(4)
    UDPClientSocket.sendto(bytesToSend, serverAddressPort)
    msgFromServer = UDPClientSocket.recvfrom(bufferSize)
    msg = "Message from Server {}".format(msgFromServer[0])
    return msgFromServer[0].decode()
if __name__=="__main__":
    print(Attend_send("20200206~97.9",host_ip="192.168.43.250",port_no=1111,bufferSize = 1024))
    