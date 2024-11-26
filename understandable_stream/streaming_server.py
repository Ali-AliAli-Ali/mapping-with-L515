# libraries for sending stream
import socket
import pickle
import struct


class RealsenseCameraServer():
    def __init__(self):
        # host_ip = get_camera_ip()
        # print(f"Server IP Address: {camera_ip}")
        camera_ip = "172.20.10.10"
        local_ip = "172.20.10.2"
        port = 1024

        # Socket creation
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (camera_ip, port)
        server_sock.bind(server_address)
        server_sock.listen(5)  # RS use 10

        print(f"Server started at socket {server_sock}")

        # def get_camera_ip():
        #     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        #     sock.connect(("8.8.8.8", 80))   # -> change to needed ip. 
        #     return sock.getsockname()[0]